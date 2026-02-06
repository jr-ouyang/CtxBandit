import numpy as np
import scipy
import cvxpy as cp
import statsmodels.api as sm

import time 
from multiprocessing import Pool
from .utils import FormatRuntimeMixin

import warnings


class PolicyInference(FormatRuntimeMixin):
    """
    Base class for PolicyInferenceSingle, PolicyInferenceJoint, PolicyInferenceDifference
    """
    def __init__(
            self, 
            num_arms, context_dim, 
            size_inference, seed_inference, 
    ):
        self.num_arms = num_arms
        self.context_dim = context_dim
        self.size_inference = size_inference
        self.seed_inference = seed_inference

        self.logistic_beta_true = 3
        self.rng_inference = np.random.default_rng(seed_inference)
        self.dataset_inference = self._generate_offline_dataset(
            size_inference, self.rng_inference, 
        )

        self.log_elr_cvx_problem = None
        self.solver_path = [cp.CLARABEL, cp.ECOS, cp.SCS]
        self.solver_verbose = False

        # MLE: for checking the uniqueness of mle
        self.mle_uniqueness_check_TOLERANCE = 1e-08

        # MLE: for determining if an unobserved extreme point gets positive weight
        self.unobserved_extreme_with_weight_TOLERANCE = 1e-6

        # Wilks: for avoiding "invalid scalar power" error in `cp.geo_mean`
        self.wilks_cvx_constraint_TOLERANCE = 1e-08
        self.wilks_cvx_geo_mean_MAX_DENOM = 1<<13

        # Runtime in print messgae: for controlling the number of decimal places shown
        self.num_decimal_places_for_runtime = 1

    def _generate_offline_dataset(self, size, rng):
        covariate = np.zeros((size, self.num_arms))
        arm = np.zeros(size, dtype=int)
        reward = np.zeros(size)
        for i in range(size):
            context_arm  = rng.standard_normal((self.num_arms, self.context_dim))
            context_user = rng.dirichlet(alpha=np.ones(self.context_dim))
            covariate[i] = np.matmul(context_arm, context_user)
            arm[i] = i % self.num_arms
            p_reward = scipy.special.expit(self.logistic_beta_true*covariate[i, arm[i]])
            reward[i] = rng.binomial(n=1, p=p_reward)
        return covariate, arm, reward

    def _fit_logistic_models(self, dataset):
        covariate, arm, reward = dataset
        logistic_models = {}
        for a in range(self.num_arms):
            match = (arm==a)
            r = reward[match]
            y = np.column_stack([r, 1-r])
            X = sm.add_constant(covariate[match, a])
            logistic_models[a] = sm.GLM(y, X, family=sm.families.Binomial()).fit() 
        return logistic_models

    def _compute_pmf_by_policy_rule(self, logistic_models, covariate, policy_param):
        scale_se, power_ub, num_arms_top = policy_param
        assert num_arms_top <= self.num_arms

        # for each arm, compute the upper bounds of all rounds
        ub = np.zeros((covariate.shape[0], self.num_arms))
        for a in range(self.num_arms):
            model = logistic_models[a]
            X = sm.add_constant(covariate[:,a])
            se = np.sqrt(model.cov_params(r_matrix=X).diagonal())
            ub[:,a] = scipy.special.expit(X@model.params + scale_se*se)

        # for each round, set ub=0 for arms ranked below `num_arms_top`
        ub = ub/np.max(ub, axis=1, keepdims=True)
        ub = ub * (np.argsort(np.argsort(ub)) >= self.num_arms-num_arms_top)

        # compute pmf
        ub_power = np.round(ub**power_ub, 2)
        pmf = ub_power/np.sum(ub_power, axis=1, keepdims=True)
        return pmf

    def clear_results_for_all_sizes(self):
        for result in self.results_by_size:
            result.clear()

    def compute_policy_value_by_mc_integration(self, repeat, size_per_repeat, seed, verbose):
        raise NotImplementedError()

    def _prepare_importance_dataset(self, size):
        raise NotImplementedError()

    def compute_mle(self, size):
        raise NotImplementedError()

    def _trim_fp_error_for_mle(self, obj):
        if isinstance(obj, tuple):
            return tuple(self._trim_fp_error_for_mle(mle) for mle in obj)
        r_min, r_max = self.extreme.r
        return np.clip(obj, r_min, r_max)

    def _trim_fp_error_for_diff_mle(self, obj):
        if isinstance(obj, tuple):
            return tuple(self._trim_fp_error_for_diff_mle(mle) for mle in obj)
        r_min, r_max = self.extreme.r
        d_max = r_max - r_min
        d_min = - d_max
        return np.clip(obj, d_min, d_max)

    def compute_wilks_interval(self, size, level):
        raise NotImplementedError()

    def _check_interval_cover_and_position(self, lower, upper):
        if self.mc_result is not None:
            # Check if the interval covers the true value
            # `position` represents the interval's position relative to the true value
            target = self.mc_result.true_value
            if lower < target < upper:
                cover, position = True, 0
            elif target < lower:
                cover, position = False, 1    # interval is on the right of the true value
            else:
                cover, position = False, -1   # interval is on the left of the true value
        else:
            cover, position = None, None
        return cover, position

    def _validate_wilks_bound(self, value):
        if value is None:
            warnings.warn(
                "\n"
                f"All solvers failed. Resetting:\n"
                "   Wilks bound = np.nan\n"
                "It will be automatically corrected later based on the bound type\n",
                stacklevel=2,
            )
            return np.nan
        elif not np.isfinite(value):
            warnings.warn(
                "\n"
                "A solver error occurred, but CVXPY did not raise an error:\n"
                f"   Wilks bound = {value}\n"
                "Resetting:\n"
                "   Wilks bound = np.nan\n"
                "It will be automatically corrected later based on the bound type\n",
                stacklevel=2,
            )
            return np.nan
        else:
            return value

    def _define_log_elr_cvx_problem(self, size):
        raise NotImplementedError()

    def _solve_log_elr_cvx_problem(self, x):
        problem = self.log_elr_cvx_problem
        problem.parameters()[0].value = x
        log_elr, solver_index = self._solve_cvx_problem(problem)
        log_elr = self._validate_log_elr(log_elr)
        return log_elr, solver_index

    def _solve_cvx_problem(self, problem):
        """NB:
        If the problem involves `geo_mean()`, 
            solvers cp.ECOS and cp.SCS may return NaN 
            with a RuntimeWarning (invalid scalar power) 
            instead of raising an error. 
        In all cases, the returned value will be passed to a validation function 
            to ensure it is meaningful.

        We define `self.solver_path` and `self.solver_verbose` as instance attributes 
            because we typically use the same solver path and keep verbosity off. 
        To use a different solver path or adjust verbosity, 
            modify these attributes before calling this function. 

        Return (problem.value, solver_index), 
            where solver_index is also the number of solvers that failed.
        """
        for solver_index, solver in enumerate(self.solver_path):
            try:
                problem.solve(solver=solver, verbose=self.solver_verbose)
                return problem.value, solver_index
            except cp.error.SolverError as e:
                if self.solver_verbose:
                    cp.settings.LOGGER.info("Solver %s failed: %s", solver, e)
        # all solvers failed
        solver_index += 1
        return None, solver_index

    def _validate_log_elr(self, value):
        if value is None:
            warnings.warn(
                "\n"
                f"All solvers failed. Resetting:\n"
                "   log(elr) = -np.inf\n",
                stacklevel=2,
            )
            return -np.inf
        elif np.isnan(value):
            warnings.warn(
                "\n"
                "A solver error occurred, but CVXPY did not raise an error:\n"
                f"   log(elr) = {value}\n"
                "Resetting:\n"
                "   log(elr) = -np.inf\n",
                stacklevel=2,
            )
            return -np.inf
        elif np.isposinf(value):
            warnings.warn(
                "\n"
                "The convex problem is infeasible:\n"
                f"   log(elr) = {value}\n"
                "Resetting:\n"
                "   log(elr) = -np.inf\n",
                stacklevel=2,
            )
            return -np.inf
        elif value > 0:
            warnings.warn(
                "\n"
                "A positive value encountered in log(elr), likely due to floating-point error:\n"
                f"   log(elr) = {value}\n"
                "Resetting:\n"
                "   log(elr) = 0\n",
                stacklevel=2,
            )
            return 0.
        else:
            return value

    def _prepare_grid(self, size, num_points, wilks_level):
        raise NotImplementedError()

    def compute_elr_over_grid(
            self, 
            size, 
            num_points=1000, wilks_level=0.999_9, 
            parallel=True, 
            num_processes=4, maxtasksperchild=None, chunksize=None, 
            verbose=False, 
    ):
        # prepare an adaptive grid based Wilks' interval or region
        #     scipy.stats.chi2.ppf(q=0.999_9, df=1)/2 = 7.568
        #     scipy.stats.chi2.ppf(q=0.999_9, df=2)/2 = 9.210
        self._prepare_grid(size, num_points, wilks_level)
        grid_result = self.grid_result_by_size[size]
        runtime_wilks_bound = grid_result.adaptive_support.runtime
        grid = grid_result.grid
        if grid.shape == (num_points, num_points, 2):
            grid = grid.reshape(-1,2)

        # define the cvx problem for evaluating the log_elr function, where
        #     log(elr) = log(el) + n * log(n) - np.sum(c*np.log(c))
        self._define_log_elr_cvx_problem(size)

        t0 = time.perf_counter()
        if parallel:
            # compute chunksize if None is given
            #     the default chunksize from multiprocessing is given by
            #     https://github.com/python/cpython/blob/main/Lib/multiprocessing/pool.py#L480
            if chunksize is None:
                chunksize, extra = divmod(len(grid), num_processes * 4)
                if extra:
                    chunksize += 1

            # compute num_chunks
            num_chunks, extra  = divmod(len(grid), chunksize)
            if extra:
                num_chunks += 1

            # compute log(elr) in parallel with multiprocessing.Pool
            with Pool(
                processes=num_processes, 
                maxtasksperchild=maxtasksperchild, 
            ) as pool:
                temp = pool.map(
                    self._solve_log_elr_cvx_problem, 
                    grid, 
                    chunksize=chunksize, 
                )
                temp_0, temp_1 = zip(*temp)
                log_elr_on_grid = np.array(temp_0)        # np.float64
                solver_index_on_grid = np.array(temp_1)   # np.int64
        else:
            # compute log(elr) sequentially
            log_elr_on_grid = np.empty(len(grid))
            solver_index_on_grid = np.empty(len(grid), dtype=np.int64)
            for i, x in enumerate(grid):
                log_elr, solver_index = self._solve_log_elr_cvx_problem(x)
                log_elr_on_grid[i] = log_elr
                solver_index_on_grid[i] = solver_index
        t1 = time.perf_counter()
        runtime_elr_on_grid = t1 - t0

        # remove the solved problem so that this class can be pickled
        self.log_elr_cvx_problem = None

        # save `elr_on_grid` to `_GridResult`
        elr_on_grid = np.exp(log_elr_on_grid)
        if grid.shape == (num_points * num_points, 2):
            elr_on_grid = elr_on_grid.reshape(num_points, num_points)
        self.grid_result_by_size[size].elr_on_grid = elr_on_grid
        self.grid_result_by_size[size].solver_index_on_grid = solver_index_on_grid

        runtime = self._GridResult._Runtime(
            wilks_bound = runtime_wilks_bound, 
            elr_on_grid = runtime_elr_on_grid, 
            total       = runtime_wilks_bound + runtime_elr_on_grid, 
        )
        self.grid_result_by_size[size].runtime = runtime

        if verbose:
            # format time as H:MM:SS
            runtime_wilks_bound = self._format_runtime(runtime.wilks_bound)
            runtime_elr_on_grid = self._format_runtime(runtime.elr_on_grid)
            runtime_total       = self._format_runtime(runtime.total)

            lines = [
                f"Completed computing the empirical likelihood ratio over a grid",
                f"    Parameters",
                f"        Importance dataset size: {size:,}",
                f"        Number of grid points: {len(grid):,}",
                f"        Wilks level for adaptive support: {wilks_level:,}",
                f"        Parallel computing: {parallel}",
            ]
            if parallel:
                lines.extend([
                    f"            num_processes: {num_processes:,}",
                    f"            maxtasksperchild: {maxtasksperchild}",
                    f"            chunksize: {chunksize:,}",
                    f"            num_chunks: {num_chunks:,}",
                ])
            lines.extend([
                f"    Runtime",
                f"        wilks_bound: {runtime_wilks_bound}",
                f"        elr_on_grid: {runtime_elr_on_grid}",
                f"        total:       {runtime_total}",
            ])
            print("\n".join(lines) + "\n")

