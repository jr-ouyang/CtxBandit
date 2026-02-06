import numpy as np
import scipy
import cvxpy as cp

import time 

from dataclasses import dataclass
from .utils import ReadableStrMixin
from collections import defaultdict

from .policy_inference import PolicyInference
from .policy_inference_mixins import (
    ComputeHPDIntervalMixin, 
    PrepareGridMixin, 
)


class PolicyInferenceSingle(
    ComputeHPDIntervalMixin, 
    PrepareGridMixin, 
    PolicyInference
):
    @dataclass
    class _MLEResult(ReadableStrMixin):
        mle: float
        unique: bool
        mle_observed: float
        interval: tuple[float, float]
        max_log_elr: float
        beta_star: float
        runtime: float

    @dataclass
    class _WilksResult(ReadableStrMixin):
        @dataclass
        class _Wilks(ReadableStrMixin):
            interval: tuple[float, float]
            cover: bool | None
            position: bool | None
            width: float
            elr_cutoff: float
            level: float
            solver_index: tuple[int, int]
            runtime: float
        wilks_95: _Wilks | None = None
        wilks_90: _Wilks | None = None
        wilks_others: dict[float, _Wilks] | None = None

    @dataclass
    class _HPDResult(ReadableStrMixin):
        @dataclass
        class _HPD(ReadableStrMixin):
            interval: tuple[float, float]
            cover: bool | None
            position: bool | None
            width: float
            elr_cutoff: float
            level: float
            runtime: float
        hpd_95: _HPD | None = None
        hpd_90: _HPD | None = None
        hpd_others: dict[float, _HPD] | None = None

    @dataclass
    class _GridResult(ReadableStrMixin):
        @dataclass
        class _AdaptiveSupport(ReadableStrMixin):
            bound: tuple[float, float]
            width: float
            elr_cutoff: float
            wilks_level: float
            solver_index: tuple[int, int]
            runtime: float
        @dataclass
        class _Runtime(ReadableStrMixin):
            wilks_bound: float
            elr_on_grid: float
            total: float
        grid: np.ndarray
        elr_on_grid: np.ndarray | None
        solver_index_on_grid: np.ndarray | None
        step_size: float
        adaptive_support: _AdaptiveSupport
        runtime: _Runtime | None

    @dataclass
    class _MCResult(ReadableStrMixin):
        @dataclass
        class _Parameters(ReadableStrMixin):
            mc_sample_size: int
            repeat: int
            size_per_repeat: int
            seed: int
        @property
        def true_value(self) -> float:
            return self.policy_value
        policy_value: float
        policy_value_oracle: float
        runtime: float
        parameters: _Parameters

    @dataclass
    class _ImportanceDataset(ReadableStrMixin):
        c: np.ndarray
        w: np.ndarray
        r: np.ndarray

    @dataclass
    class _Extreme(ReadableStrMixin):
        w: tuple[float, float]
        r: tuple[float, float]

    def __init__(
            self, 
            num_arms, context_dim, 
            size_inference, seed_inference, 
            size_learning, seed_learning, policy_param, 
    ):
        super().__init__(
            num_arms, context_dim, 
            size_inference, seed_inference, 
        )
        self.size_learning = size_learning
        self.seed_learning = seed_learning
        self.policy_param = policy_param

        self.rng_learning = np.random.default_rng(seed_learning)
        self.dataset_learning = self._generate_offline_dataset(
            size_learning, self.rng_learning, 
        )
        self.logistic_models = self._fit_logistic_models(self.dataset_learning)

        covariate = self.dataset_inference[0]
        self.pmf = self._compute_pmf_by_policy_rule(
            self.logistic_models, covariate, policy_param, 
        )

        self.extreme = self._Extreme(
            w = (0, num_arms), 
            r = (0, 1),          # can be any interval encompassing [0,1]
        )

        self.mc_result = None
        self.importance_dataset_by_size = {}
        self.mle_result_by_size = {}
        self.grid_result_by_size = {}
        self.hpd_result_by_size = {}
        self.wilks_result_by_size = {}
        self.results_by_size = [
            self.importance_dataset_by_size, 
            self.mle_result_by_size, 
            self.grid_result_by_size, 
            self.hpd_result_by_size, 
            self.wilks_result_by_size, 
        ]

    def compute_policy_value_by_mc_integration(
            self, 
            repeat=1000, size_per_repeat=1000, 
            seed=31415926, verbose=False, 
    ):
        t0 = time.perf_counter()
        rng = np.random.default_rng(seed)
        mc_policy_value        = np.zeros(repeat)
        mc_policy_value_oracle = np.zeros(repeat)
        for i in range(repeat):
            context_arm  = rng.standard_normal((size_per_repeat, self.num_arms, self.context_dim))
            context_user = rng.dirichlet(alpha=np.ones(self.context_dim), size=size_per_repeat)[:,:,None]
            covariate = np.matmul(context_arm, context_user).squeeze(2)
            p_reward = scipy.special.expit(self.logistic_beta_true*covariate)
            pmf = self._compute_pmf_by_policy_rule(
                self.logistic_models, covariate, self.policy_param, 
            )
            mc_policy_value[i]        = np.mean(np.sum(p_reward * pmf, axis=1))
            mc_policy_value_oracle[i] = np.mean(np.max(p_reward, axis=1))

        policy_value = np.mean(mc_policy_value)
        policy_value_oracle = np.mean(mc_policy_value_oracle)
        mc_sample_size=repeat*size_per_repeat
        t1 = time.perf_counter()
        runtime = t1 - t0

        self.mc_result = self._MCResult(
            policy_value=float(policy_value), 
            policy_value_oracle=float(policy_value_oracle), 
            runtime=runtime, 
            parameters=self._MCResult._Parameters(
                mc_sample_size=mc_sample_size, 
                repeat=repeat, 
                size_per_repeat=size_per_repeat, 
                seed=seed, 
            )
        )

        if verbose:
            formatted_time = self._format_runtime(runtime)    # H:MM:SS.x
            print(
                f"Policy value by Monte Carlo integration\n"
                f"    {policy_value:.7g}\n"
                f"    {policy_value_oracle:.7g} (oracle)\n"
                f"Parameters\n"
                f"    MC sample size: {mc_sample_size:,} ({repeat:,} repetitions of subsample size {size_per_repeat:,} each)\n"
                f"    Random seed: {seed}\n"
                f"Runtime\n"
                f"    {formatted_time}\n"
            )

    def _prepare_importance_dataset(self, size):
        assert size <= self.size_inference
        arm = self.dataset_inference[1][:size]
        reward = self.dataset_inference[2][:size]
        pmf = self.pmf[:size]

        # compute importance weights for the observed arms
        weight = pmf[np.arange(size), arm] * self.num_arms

        # count the (weight, reward) pair
        dct = defaultdict(int)
        for i in range(size):
            if weight[i] == 0:
                # the value of reward is immaterial when the weight is 0
                # NB: this handling is different from the other two classes
                dct[(weight[i], 0)] += 1
            else:
                dct[(weight[i], reward[i])] += 1
        c_w_r = np.array([(c, w, r) for (w,r), c in dct.items()])

        self.importance_dataset_by_size[size] = self._ImportanceDataset(
            c = c_w_r[:,0], 
            w = c_w_r[:,1], 
            r = c_w_r[:,2], 
        )

    def compute_mle(self, size):
        t0 = time.perf_counter()

        if size not in self.importance_dataset_by_size:
            self._prepare_importance_dataset(size)
        importance_dataset = self.importance_dataset_by_size[size]
        c = importance_dataset.c
        w = importance_dataset.w
        r = importance_dataset.r

        n = np.sum(c)
        gr = w > 1
        le = w < 1
        beta_lb = np.append(
            (c[gr] - n) / n / (w[gr] - 1),
            - 1 / (self.extreme.w[1] - 1),
        )
        beta_ub = np.append(
            (c[le] - n) / n / (w[le] - 1),
            - 1 / (self.extreme.w[0] - 1),
        )
        beta_min = np.max(beta_lb)
        beta_max = np.min(beta_ub)

        def grad_obj(beta):
            return np.sum( c * (w-1) / ( 1 + beta*(w-1) ) )
        grad_left  = grad_obj(beta_min)
        grad_right = grad_obj(beta_max)        

        if grad_left * grad_right < 0:
            beta_star = scipy.optimize.brentq(f=grad_obj, a=beta_min, b=beta_max)
        elif grad_right > 0:
            beta_star = beta_max
        else:
            beta_star = beta_min

        log_elr = - np.sum(c*np.log(1 + (w-1)*beta_star))
        max_log_elr = self._validate_log_elr(log_elr)

        Qi = c/(1 + (w-1)*beta_star)/n       # probability on the i-th point
        mle_observed = np.sum(w*r*Qi)

        resid_wQ = 1 - np.sum(w*Qi)          # Residual wQ = 1 - \sum_i wi*Qi
        resid_wQ = np.clip(resid_wQ, 0, 1)   # remove numerical error

        # if resid_wQ > 0, then the MLE is not unique and forms an interval
        r_min, r_max = self.extreme.r
        interval = (
            mle_observed + resid_wQ * r_min, 
            mle_observed + resid_wQ * r_max, 
        )
        mle = sum(interval)/2
        unique = (resid_wQ < self.mle_uniqueness_check_TOLERANCE)

        mle          = self._trim_fp_error_for_mle(mle)
        mle_observed = self._trim_fp_error_for_mle(mle_observed)
        interval     = self._trim_fp_error_for_mle(interval)

        t1 = time.perf_counter()
        runtime = t1 - t0

        self.mle_result_by_size[size] = self._MLEResult(
            mle=float(mle), 
            unique=bool(unique), 
            mle_observed=float(mle_observed), 
            interval=tuple(float(x) for x in interval), 
            max_log_elr=float(max_log_elr), 
            beta_star=float(beta_star), 
            runtime=runtime, 
        )

    def compute_wilks_interval(self, size, level=0.95):
        t0 = time.perf_counter()
        importance_dataset = self.importance_dataset_by_size[size]
        c = importance_dataset.c
        w = importance_dataset.w
        r = importance_dataset.r
        max_log_elr = self.mle_result_by_size[size].max_log_elr

        n = np.sum(c)
        log_elr_cutoff = max_log_elr - scipy.stats.chi2.ppf(q=level, df=1)/2
        const = np.exp(log_elr_cutoff/n)

        def compute_lower_bound(c, w, r, const):
            TOLERANCE = self.wilks_cvx_constraint_TOLERANCE
            MAX_DENOM = self.wilks_cvx_geo_mean_MAX_DENOM
            beta = cp.Variable()
            gamma = cp.Variable()
            r_min = self.extreme.r[0]
            constraint = [
                gamma + w*beta + w*r_min >= TOLERANCE
                for w in self.extreme.w
            ]

            base = gamma + w*beta + w*r
            geo_mean = cp.geo_mean(x=base, p=c, max_denom=MAX_DENOM) if c.shape[0] > 1 else base
            objective = - gamma - beta + const * geo_mean
            problem = cp.Problem(cp.Maximize(objective), constraint)

            wilks_bound, solver_index = self._solve_cvx_problem(problem)
            wilks_bound = self._validate_wilks_bound(wilks_bound)
            return wilks_bound, solver_index

        r_min, r_max = self.extreme.r
        shift = r_min + r_max
        lower, solver_index_lower = compute_lower_bound(c, w,    r,    const)
        upper, solver_index_upper = compute_lower_bound(c, w, shift-r, const)
        upper = shift - upper
        lower = max(r_min, lower)   # remove numerical error; if nan, return the boundary value
        upper = min(r_max, upper)   # remove numerical error; if nan, return the boundary value

        cover, position = self._check_interval_cover_and_position(lower, upper)

        elr_cutoff = np.exp(log_elr_cutoff)
        t1 = time.perf_counter()
        runtime = t1 - t0

        wilks = self._WilksResult._Wilks(
            interval=(float(lower), float(upper)), 
            cover=cover, 
            position=position, 
            width=float(upper-lower), 
            elr_cutoff=float(elr_cutoff), 
            level=level, 
            solver_index=(solver_index_lower, solver_index_upper), 
            runtime=runtime, 
        )

        if size not in self.wilks_result_by_size:
            self.wilks_result_by_size[size] = self._WilksResult()

        if level in [0.95, 0.90]:
            attr_name = f"wilks_{int(level*100)}"
            setattr(self.wilks_result_by_size[size], attr_name, wilks)
        else:
            if self.wilks_result_by_size[size].wilks_others is None:
                self.wilks_result_by_size[size].wilks_others = {}
            self.wilks_result_by_size[size].wilks_others[level] = wilks

    def _define_log_elr_cvx_problem(self, size):
        importance_dataset = self.importance_dataset_by_size[size]
        c = importance_dataset.c
        w = importance_dataset.w
        r = importance_dataset.r

        beta = cp.Variable()
        tau = cp.Variable()
        v = cp.Parameter()
        ex = list(set( [
            (w, w*r) 
            for w in self.extreme.w 
            for r in self.extreme.r 
        ] ))
        constraint = [ 1 + beta*(w-1) + tau*(wr-v) >= 0 for (w, wr) in ex]
        objective = - cp.sum(cp.multiply(c, cp.log1p( beta*(w-1) + tau*(w*r-v) ))) 
        self.log_elr_cvx_problem = cp.Problem(cp.Minimize(objective), constraint)

