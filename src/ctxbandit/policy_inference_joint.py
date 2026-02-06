import numpy as np
import scipy
import cvxpy as cp

import time 

from dataclasses import dataclass, field
from .utils import ReadableStrMixin
from typing import Literal
from collections import defaultdict

from .policy_inference import PolicyInference
from .policy_inference_mixins import (
    ComputePolicyValueByMCIntegration2DMixin, 
    ComputeMLE2DMixin, 
)


class PolicyInferenceJoint(
    ComputePolicyValueByMCIntegration2DMixin, 
    ComputeMLE2DMixin, 
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
        solver_index: int
        runtime: float

    @dataclass
    class _ProbImprovementResult(ReadableStrMixin):
        abs_margin_to_prob: dict[float, float] = field(default_factory=dict)
        rel_margin_to_prob: dict[float, float] = field(default_factory=dict)

    @dataclass
    class _GridResult(ReadableStrMixin):
        @dataclass
        class _AdaptiveSupport(ReadableStrMixin):
            bound_0: tuple[float, float]
            bound_1: tuple[float, float]
            width_0: float
            width_1: float
            elr_cutoff: float
            wilks_level: float
            solver_index_0: tuple[int, int]
            solver_index_1: tuple[int, int]
            runtime: float
        @dataclass
        class _Runtime(ReadableStrMixin):
            wilks_bound: float
            elr_on_grid: float
            total: float
        grid_0: np.ndarray
        grid_1: np.ndarray
        elr_on_grid: np.ndarray | None
        solver_index_on_grid: np.ndarray | None
        grid: np.ndarray
        step_size_0: float
        step_size_1: float
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
            return self.policy_value_vector
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
        self.prob_improvement_result_by_size = {}
        self.results_by_size = [
            self.importance_dataset_by_size, 
            self.mle_result_by_size, 
            self.grid_result_by_size, 
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
            if weight_baseline[i] == weight_new[i] == 0:
                # the value of reward is immaterial when both weights are 0
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

    def _define_log_elr_cvx_problem(self, size):
        importance_dataset = self.importance_dataset_by_size[size]
        c  = importance_dataset.c
        ww = importance_dataset.ww
        r  = importance_dataset.r

        beta = cp.Variable(2)
        tau = cp.Variable(2)
        vv = cp.Parameter(2)

        # remove duplicated (ww, ww*r) pair
        #     numpy.ndarray is unhashable, so cannot use a `set` directly
        #     convert numpy.ndarray to tuple, then use `set`, then convert back
        ex_tuple = list(set( [
            (tuple(ww), tuple(ww*r))
            for ww in self.extreme.ww
            for r in self.extreme.r
        ] ))
        ex = [
            (np.array(ww), np.array(wwr)) 
            for (ww, wwr) in ex_tuple
        ]

        constraint = [1 + (ww-1)@beta + (wwr - vv)@tau >= 0 for (ww, wwr) in ex]
        wwr = ww * r[:, None]
        objective = - cp.sum(cp.multiply(c, cp.log1p( (ww-1)@beta + (wwr-vv[None])@tau )))
        self.log_elr_cvx_problem = cp.Problem(cp.Minimize(objective), constraint)

    def compute_wilks_interval(self, size, level):
        raise NotImplementedError(
            "Check out a related method instead:\n"
            "    self._compute_wilks_bound()\n",
        )

    def _compute_wilks_bound(self, size, wilks_level):
        t0 = time.perf_counter()
        importance_dataset = self.importance_dataset_by_size[size]
        c  = importance_dataset.c
        ww = importance_dataset.ww
        r  = importance_dataset.r
        max_log_elr = self.mle_result_by_size[size].max_log_elr

        n = np.sum(c)
        log_elr_cutoff = max_log_elr - scipy.stats.chi2.ppf(q=wilks_level, df=2)/2
        const = np.exp(log_elr_cutoff/n)

        def compute_lower_bound(c, ww, r, const, which_component):
            TOLERANCE = self.wilks_cvx_constraint_TOLERANCE
            MAX_DENOM = self.wilks_cvx_geo_mean_MAX_DENOM
            beta = cp.Variable(2)
            gamma = cp.Variable()
            r_min = self.extreme.r[0]
            constraint = [
                gamma + ww@beta + ww[which_component]*r_min >= TOLERANCE
                for ww in self.extreme.ww
            ]

            wr = ww[:,which_component] * r
            base = gamma + ww@beta + wr
            geo_mean = cp.geo_mean(x=base, p=c, max_denom=MAX_DENOM) if c.shape[0] > 1 else base
            objective = - gamma - cp.sum(beta) + const * geo_mean
            problem = cp.Problem(cp.Maximize(objective), constraint)

            wilks_bound, solver_index = self._solve_cvx_problem(problem)
            wilks_bound = self._validate_wilks_bound(wilks_bound)
            return wilks_bound, solver_index

        bounds = []
        solver_indices = []
        for which_component in [0, 1]:
            r_min, r_max = self.extreme.r
            shift = r_min + r_max
            lower, solver_index_lower = compute_lower_bound(c, ww,    r,    const, which_component)
            upper, solver_index_upper = compute_lower_bound(c, ww, shift-r, const, which_component)
            upper = shift - upper
            lower = max(r_min, lower)   # remove numerical error; if nan, return the boundary value
            upper = min(r_max, upper)   # remove numerical error; if nan, return the boundary value
            bounds.append((lower, upper))
            solver_indices.append((solver_index_lower, solver_index_upper))

        t1 = time.perf_counter()
        runtime = t1 - t0

        adaptive_support = self._GridResult._AdaptiveSupport(
            bound_0=tuple(float(x) for x in bounds[0]), 
            bound_1=tuple(float(x) for x in bounds[1]), 
            width_0=float(bounds[0][1] - bounds[0][0]), 
            width_1=float(bounds[1][1] - bounds[1][0]), 
            elr_cutoff=float(np.exp(log_elr_cutoff)), 
            wilks_level=wilks_level, 
            solver_index_0=solver_indices[0], 
            solver_index_1=solver_indices[1], 
            runtime=runtime, 
        )
        return adaptive_support

    def _prepare_grid(self, size, num_points, wilks_level):
        adaptive_support = self._compute_wilks_bound(size, wilks_level)

        # 1-dimensional grid
        grid_1d_0, step_size_0 = np.linspace(
            *adaptive_support.bound_0, num=num_points, 
            endpoint=False, retstep=True
        )
        grid_1d_1, step_size_1 = np.linspace(
            *adaptive_support.bound_1, num=num_points, 
            endpoint=False, retstep=True
        )
        grid_1d_0 = grid_1d_0 + step_size_0 / 2
        grid_1d_1 = grid_1d_1 + step_size_1 / 2

        # 2-dimensional grid
        #     grid_0: shape (n, n)
        #     grid_1: shape (n, n)
        #     grid  : shape (n, n, 2)
        grid_0, grid_1 = np.meshgrid(grid_1d_0, grid_1d_1, indexing='ij')
        grid = np.stack((grid_0, grid_1), axis=2)

        self.grid_result_by_size[size] = self._GridResult(
            grid_0=grid_0,       #  shape (n, n)
            grid_1=grid_1,       #  shape (n, n)
            elr_on_grid=None,    #  shape (n, n)
            solver_index_on_grid=None,  # (n, n)
            grid=grid,           #  shape (n, n, 2)
            step_size_0=step_size_0, 
            step_size_1=step_size_1, 
            adaptive_support=adaptive_support, 
            runtime=None, 
        )

    def compute_prob_improvement(
            self, 
            size, 
            mode: Literal["abs", "rel"], 
            margin: float, 
    ):
        # assume flat prior => posterior is proportional to ELR 
        grid_result = self.grid_result_by_size[size]
        posterior = grid_result.elr_on_grid / grid_result.elr_on_grid.sum()
        grid_0 = grid_result.grid_0
        grid_1 = grid_result.grid_1

        if size not in self.prob_improvement_result_by_size:
            self.prob_improvement_result_by_size[size] = self._ProbImprovementResult()

        if mode == "abs":
            greater = grid_1 >  grid_0 + margin
            equal   = grid_1 == grid_0 + margin
            prob = posterior[greater].sum() + posterior[equal].sum()/2
            self.prob_improvement_result_by_size[size].abs_margin_to_prob[margin] = float(prob)
        else:
            greater = grid_1 >  grid_0 * (1 + margin)
            equal   = grid_1 == grid_0 * (1 + margin)
            prob = posterior[greater].sum() + posterior[equal].sum()/2
            self.prob_improvement_result_by_size[size].rel_margin_to_prob[margin] = float(prob)

