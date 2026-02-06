import numpy as np
import scipy
import cvxpy as cp

import time 

from dataclasses import dataclass, field
from .utils import ReadableStrMixin
from collections import defaultdict

from .policy_inference import PolicyInference
from .policy_inference_mixins import (
    ComputePolicyValueByMCIntegration2DMixin, 
    ComputeHPDIntervalMixin, 
    ComputeMLE2DMixin, 
    PrepareGridMixin, 
)


class PolicyInferenceDifference(
    ComputePolicyValueByMCIntegration2DMixin, 
    ComputeHPDIntervalMixin, 
    ComputeMLE2DMixin, 
    PrepareGridMixin, 
    PolicyInference, 
):
    @dataclass
    class _MLEResult(ReadableStrMixin):
        type Point = list[float]
        @dataclass
        class _Joint(ReadableStrMixin):
            type Point = list[float]
            mle: Point
            unique: bool
            mle_observed: Point
            interval: tuple[Point, Point]
            region: tuple[Point, Point, Point, Point]
        @dataclass
        class _Difference(ReadableStrMixin):
            mle: float
            unique: bool
            mle_observed: float
            interval: tuple[float, float]
        joint: _Joint
        difference: _Difference
        max_log_elr: float
        beta_star: Point
        unobserved_extreme: tuple[Point, ...]
        unobserved_extreme_with_weight: tuple[Point, ...]
        unobserved_extreme_with_weight_excluding_0: tuple[Point, ...]
        solver_index: tuple[int, int]
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
    class _ProbImprovementResult(ReadableStrMixin):
        margin_to_prob: dict[float, float] = field(default_factory=dict)

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
            return self.policy_value_difference
        policy_value_vector: tuple[float, float]
        policy_value_difference: float
        policy_value_baseline: float
        policy_value_new: float
        policy_value_oracle: float
        runtime: float
        parameters: _Parameters

    @dataclass
    class _ImportanceDataset(ReadableStrMixin):
        c : np.ndarray
        ww: np.ndarray
        r : np.ndarray

    @dataclass
    class _Extreme(ReadableStrMixin):
        ww: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        r : tuple[float, float]

    def __init__(
            self, 
            num_arms, context_dim, 
            size_inference, seed_inference, 
            size_learning_baseline, seed_learning_baseline, policy_param_baseline, 
            size_learning_new, seed_learning_new, policy_param_new, 
    ):
        super().__init__(
            num_arms, context_dim, 
            size_inference, seed_inference, 
        )
        self.size_learning_baseline = size_learning_baseline
        self.seed_learning_baseline = seed_learning_baseline
        self.policy_param_baseline = policy_param_baseline

        self.size_learning_new = size_learning_new
        self.seed_learning_new = seed_learning_new
        self.policy_param_new = policy_param_new

        # logistic models - baseline
        self.rng_learning_baseline = np.random.default_rng(seed_learning_baseline)
        self.dataset_learning_baseline = self._generate_offline_dataset(
            size_learning_baseline, self.rng_learning_baseline, 
        )
        self.logistic_models_baseline = self._fit_logistic_models(
            self.dataset_learning_baseline, 
        )

        # logistic models - new
        self.rng_learning_new = np.random.default_rng(seed_learning_new)
        self.dataset_learning_new = self._generate_offline_dataset(
            size_learning_new, self.rng_learning_new, 
        )
        self.logistic_models_new = self._fit_logistic_models(
            self.dataset_learning_new, 
        )

        # compute pmf for baseline and new policies
        covariate = self.dataset_inference[0]
        self.pmf_baseline = self._compute_pmf_by_policy_rule(
            self.logistic_models_baseline, covariate, policy_param_baseline, 
        )
        self.pmf_new = self._compute_pmf_by_policy_rule(
            self.logistic_models_new, covariate, policy_param_new, 
        )

        self.extreme = self._Extreme(
            ww = tuple(
                np.array([w0, w1]) 
                for w0 in [0, num_arms] 
                for w1 in [0, num_arms] 
            ), 
            r = (0, 1),          # can be any interval encompassing [0,1]
        )

        self.mc_result = None
        self.importance_dataset_by_size = {}
        self.mle_result_by_size = {}
        self.grid_result_by_size = {}
        self.hpd_result_by_size = {}
        self.wilks_result_by_size = {}
        self.prob_improvement_result_by_size = {}
        self.results_by_size = [
            self.importance_dataset_by_size, 
            self.mle_result_by_size, 
            self.grid_result_by_size, 
            self.hpd_result_by_size, 
            self.wilks_result_by_size, 
            self.prob_improvement_result_by_size, 
        ]

    def _prepare_importance_dataset(self, size):
        assert size <= self.size_inference
        arm = self.dataset_inference[1][:size]
        reward = self.dataset_inference[2][:size]
        pmf_baseline = self.pmf_baseline[:size]
        pmf_new      = self.pmf_new[:size]

        # compute importance weights for the observed arms
        weight_baseline = pmf_baseline[np.arange(size), arm] * self.num_arms
        weight_new      = pmf_new     [np.arange(size), arm] * self.num_arms

        # count the (weight_baseline, weight_new, reward) trio
        dct = defaultdict(int)
        for i in range(size):
            if weight_baseline[i] == weight_new[i]:
                # the value of reward is immaterial when two weights are equal
                # NB: this handling is different from the other two classes
                dct[(weight_baseline[i], weight_new[i], 0)] += 1
            else:
                dct[(weight_baseline[i], weight_new[i], reward[i])] += 1
        c_ww_r = np.array([(c, w0, w1, r) for (w0, w1, r), c in dct.items()])

        self.importance_dataset_by_size[size] = self._ImportanceDataset(
            c  = c_ww_r[:,0], 
            ww = c_ww_r[:,1:3], 
            r  = c_ww_r[:,3], 
        )

    def compute_wilks_interval(self, size, level=0.95):
        t0 = time.perf_counter()
        importance_dataset = self.importance_dataset_by_size[size]
        c  = importance_dataset.c
        ww = importance_dataset.ww
        r  = importance_dataset.r
        max_log_elr = self.mle_result_by_size[size].max_log_elr

        n = np.sum(c)
        log_elr_cutoff = max_log_elr - scipy.stats.chi2.ppf(q=level, df=1)/2
        const = np.exp(log_elr_cutoff/n)

        def compute_lower_bound(c, ww, r, const):
            TOLERANCE = self.wilks_cvx_constraint_TOLERANCE
            MAX_DENOM = self.wilks_cvx_geo_mean_MAX_DENOM
            beta = cp.Variable(2)
            gamma = cp.Variable()
            r_min, r_max = self.extreme.r
            constraint = [
                gamma + ww@beta + min((ww[1]-ww[0])*r_min, (ww[1]-ww[0])*r_max) >= TOLERANCE
                for ww in self.extreme.ww
            ]

            diff_wr = (ww[:,1] - ww[:,0]) * r
            base = gamma + ww@beta + diff_wr
            geo_mean = cp.geo_mean(x=base, p=c, max_denom=MAX_DENOM) if c.shape[0] > 1 else base
            objective = - gamma - cp.sum(beta) + const * geo_mean
            problem = cp.Problem(cp.Maximize(objective), constraint)

            wilks_bound, solver_index = self._solve_cvx_problem(problem)
            wilks_bound = self._validate_wilks_bound(wilks_bound)
            return wilks_bound, solver_index

        r_min, r_max = self.extreme.r
        shift = r_min + r_max
        lower, solver_index_lower = compute_lower_bound(c, ww,    r,    const)
        upper, solver_index_upper = compute_lower_bound(c, ww, shift-r, const)
        upper = - upper

        d_max = r_max - r_min
        d_min = - d_max
        lower = max(d_min, lower)   # remove numerical error; if nan, return the boundary value
        upper = min(d_max, upper)   # remove numerical error; if nan, return the boundary value

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
        c  = importance_dataset.c
        ww = importance_dataset.ww
        r  = importance_dataset.r

        beta = cp.Variable(2)
        delta = cp.Variable()
        d = cp.Parameter()

        # remove duplicated (ww, ww*r) pair
        #     numpy.ndarray is unhashable, so cannot use a `set` directly
        #     convert numpy.ndarray to tuple, then use `set`, then convert back
        ex_tuple = list(set( [
            (tuple(ww), (ww[1]-ww[0])*r)
            for ww in self.extreme.ww
            for r in self.extreme.r
        ] ))
        ex = [
            (np.array(ww), diff_wr) 
            for (ww, diff_wr) in ex_tuple
        ]

        constraint = [
            1 + (ww-1)@beta + (diff_wr - d)*delta >= 0 
            for (ww, diff_wr) in ex
        ]
        diff_wr = (ww[:,1] - ww[:,0]) * r
        objective = - cp.sum(cp.multiply(c, cp.log1p( (ww-1)@beta + (diff_wr - d)*delta )))
        self.log_elr_cvx_problem = cp.Problem(cp.Minimize(objective), constraint)

    def compute_prob_improvement(self, size, margin):
        # assume flat prior => posterior is proportional to ELR 
        grid_result = self.grid_result_by_size[size]
        posterior = grid_result.elr_on_grid / grid_result.elr_on_grid.sum()
        grid = grid_result.grid

        if size not in self.prob_improvement_result_by_size:
            self.prob_improvement_result_by_size[size] = self._ProbImprovementResult()

        greater = grid >  margin
        equal   = grid == margin
        prob = posterior[greater].sum() + posterior[equal].sum()/2
        self.prob_improvement_result_by_size[size].margin_to_prob[margin] = float(prob)

