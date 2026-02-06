import numpy as np
import scipy
import cvxpy as cp

import time 


class PrepareGridMixin:
    def _prepare_grid(self, size, num_points, wilks_level):
        self.compute_wilks_interval(size, level=wilks_level)
        wilks_result = self.wilks_result_by_size[size]
        if wilks_level in [0.95, 0.90]:
            attr_name = f"wilks_{int(wilks_level*100)}"
            wilks = getattr(wilks_result, attr_name)
        else:
            wilks = wilks_result.wilks_others[wilks_level]

        grid_lower, grid_upper = wilks.interval
        grid, step_size = np.linspace(
            grid_lower, grid_upper, num=num_points, 
            endpoint=False, retstep=True, 
        )
        grid = grid + step_size/2

        adaptive_support = self._GridResult._AdaptiveSupport(
            bound=wilks.interval, 
            width=wilks.width, 
            elr_cutoff=wilks.elr_cutoff, 
            wilks_level=wilks.level, 
            solver_index=wilks.solver_index, 
            runtime=wilks.runtime, 
        )

        self.grid_result_by_size[size] = self._GridResult(
            grid=grid,
            elr_on_grid=None, 
            solver_index_on_grid=None, 
            step_size=step_size, 
            adaptive_support=adaptive_support, 
            runtime=None, 
        )


class ComputeHPDIntervalMixin:
    def compute_hpd_interval(self, size, level=0.95):
        t0 = time.perf_counter()
        if size not in self.grid_result_by_size:
            raise ValueError("Need to compute elr over a grid first\n")

        # assume flat prior => posterior is proportional to ELR 
        grid_result = self.grid_result_by_size[size]
        posterior = grid_result.elr_on_grid / grid_result.elr_on_grid.sum()

        grid = grid_result.grid
        step_size = grid_result.step_size

        bin_star = np.argmax(posterior)
        bin_left, bin_right = bin_star, bin_star
        bin_rightmost = grid.shape[0] - 1

        # start with the bin with the highest value
        curr_lvl = posterior[bin_star]
        is_last_bin_on_right = True # whether the last bin added is on the right

        # keep adding bins until current level reaches target
        while curr_lvl < level:
            # if reach the leftmost end, add a bin on the right
            if bin_left == 0:
                bin_right += 1
                curr_lvl += posterior[bin_right]
                is_last_bin_on_right = True
            # if reach the rightmost end, add a bin on the left
            elif bin_right==bin_rightmost:
                bin_left -= 1
                curr_lvl += posterior[bin_left]
                is_last_bin_on_right = False
            # otherwise, add a bin that has higher value
            elif posterior[bin_left-1] < posterior[bin_right+1]:
                bin_right += 1
                curr_lvl += posterior[bin_right]
                is_last_bin_on_right = True
            else:
                bin_left -= 1
                curr_lvl += posterior[bin_left]
                is_last_bin_on_right = False

        # extend interval by half a bin width to include boundary bins
        lower = grid[bin_left]  - step_size/2
        upper = grid[bin_right] + step_size/2

        # after the while loop, `curr_lvl >= level`, so need to remove the extra length
        if is_last_bin_on_right:
            extra_length = (curr_lvl - level)/posterior[bin_right] * step_size
            upper -= extra_length
        else:
            extra_length = (curr_lvl - level)/posterior[bin_left]  * step_size
            lower += extra_length

        support_lower, support_upper = grid_result.adaptive_support.bound
        lower = max(support_lower, lower)   # remove numerical error
        upper = min(support_upper, upper)   # remove numerical error

        cover, position = self._check_interval_cover_and_position(lower, upper)

        elr_cutoff = min(grid_result.elr_on_grid[[bin_left, bin_right]])
        t1 = time.perf_counter()
        runtime = t1 - t0

        hpd = self._HPDResult._HPD(
            interval=(float(lower), float(upper)), 
            cover=cover, 
            position=position, 
            width=float(upper-lower), 
            elr_cutoff=float(elr_cutoff), 
            level=level, 
            runtime=runtime, 
        )

        if size not in self.hpd_result_by_size:
            self.hpd_result_by_size[size] = self._HPDResult()

        if level in [0.95, 0.90]:
            attr_name = f"hpd_{int(level*100)}"
            setattr(self.hpd_result_by_size[size], attr_name, hpd)
        else:
            if self.hpd_result_by_size[size].hpd_others is None:
                self.hpd_result_by_size[size].hpd_others = {}
            self.hpd_result_by_size[size].hpd_others[level] = hpd


class ComputeMLE2DMixin:
    def compute_mle(self, size):
        t0 = time.perf_counter()

        if size not in self.importance_dataset_by_size:
            self._prepare_importance_dataset(size)
        importance_dataset = self.importance_dataset_by_size[size]
        c  = importance_dataset.c
        ww = importance_dataset.ww
        r  = importance_dataset.r

        n  = np.sum(c)
        beta = cp.Variable(2)
        constraint = [1 + (ww-1)@beta >= 0 for ww in self.extreme.ww]
        objective = - cp.sum(cp.multiply(c, cp.log1p( (ww-1)@beta )))
        problem = cp.Problem(cp.Minimize(objective), constraint)

        log_elr, solver_index = self._solve_cvx_problem(problem)
        max_log_elr = self._validate_log_elr(log_elr)
        beta_star = beta.value

        Qi = c/(1 + (ww-1)@beta.value)/n     # probability on the i-th point
        mle_observed = Qi*r@ww

        resid_wQ = 1 - Qi@ww
        resid_wQ = np.clip(resid_wQ, 0, 1)   # remove numerical error

        # if resid_wQ > (0,0), then the MLE is not unique;
        #     it forms either an interval or a parallelogram
        r_min, r_max = self.extreme.r
        interval = (
            mle_observed + resid_wQ * r_min, 
            mle_observed + resid_wQ * r_max, 
        )
        mle = sum(interval)/2
        unique = np.all(resid_wQ < self.mle_uniqueness_check_TOLERANCE)

        # if two extreme points of the support of ww are unobserved in the dataset, 
        #     and the optimization assigns them positive probability weight, 
        #     and (0,0) is not among them, 
        # then the MLE is not unique and forms a parallelogram
        TOLERANCE = self.unobserved_extreme_with_weight_TOLERANCE
        unobserved_extreme = tuple(
            point 
            for point in self.extreme.ww 
            if not np.any(np.all(point==ww, axis=1))
        )
        unobserved_extreme_with_weight = tuple(
            point
            for point in unobserved_extreme
            if 1+(point-1)@beta.value < TOLERANCE
        )
        unobserved_extreme_with_weight_excluding_0 = tuple(
            point
            for point in unobserved_extreme_with_weight
            if not np.all(point==0)
        )
        if len(unobserved_extreme_with_weight_excluding_0) == 2:
            W = np.array(unobserved_extreme_with_weight_excluding_0).T
            Q = np.linalg.solve(W, resid_wQ)
            resid_wQ_0 = W[:,0] * Q[0]
            resid_wQ_1 = W[:,1] * Q[1]
            region = (
                mle_observed + resid_wQ_0 * r_min + resid_wQ_1 * r_min, 
                mle_observed + resid_wQ_0 * r_min + resid_wQ_1 * r_max, 
                mle_observed + resid_wQ_0 * r_max + resid_wQ_1 * r_max, 
                mle_observed + resid_wQ_0 * r_max + resid_wQ_1 * r_min, 
            )
        else:
            region = (
                mle_observed + resid_wQ * r_min, 
                mle_observed + resid_wQ * r_max, 
                mle_observed + resid_wQ * r_max, 
                mle_observed + resid_wQ * r_min, 
            )

        mle          = self._trim_fp_error_for_mle(mle)
        mle_observed = self._trim_fp_error_for_mle(mle_observed)
        interval     = self._trim_fp_error_for_mle(interval)
        region       = self._trim_fp_error_for_mle(region)

        # compute the mle of the policy value difference 
        #     based on the mle of the policy value vector
        diff_mle = mle[1] - mle[0]
        diff_mle_observed = mle_observed[1] - mle_observed[0]
        temp = [
            point[1] - point[0] 
            for point in region
        ]
        diff_interval = (min(temp), max(temp))

        diff_mle          = self._trim_fp_error_for_diff_mle(diff_mle)
        diff_mle_observed = self._trim_fp_error_for_diff_mle(diff_mle_observed)
        diff_interval     = self._trim_fp_error_for_diff_mle(diff_interval)
        diff_unique = (diff_interval[1]-diff_interval[0] < self.mle_uniqueness_check_TOLERANCE)

        t1 = time.perf_counter()
        runtime = t1 - t0

        self.mle_result_by_size[size] = self._MLEResult(
            joint=self._MLEResult._Joint(
                mle=[float(x) for x in mle], 
                unique=bool(unique), 
                mle_observed=[float(x) for x in mle_observed], 
                interval=tuple(
                    [float(x) for x in xx] 
                    for xx in interval
                ), 
                region=tuple(
                    [float(x) for x in xx] 
                    for xx in region
                ), 
            ), 
            difference=self._MLEResult._Difference(
                mle=float(diff_mle), 
                unique=bool(diff_unique), 
                mle_observed=float(diff_mle_observed), 
                interval=tuple(float(x) for x in diff_interval), 
            ), 
            max_log_elr=float(max_log_elr), 
            beta_star=[float(x) for x in beta_star], 
            unobserved_extreme=tuple(
                [float(x) for x in xx] 
                for xx in unobserved_extreme
            ), 
            unobserved_extreme_with_weight=tuple(
                [float(x) for x in xx] 
                for xx in unobserved_extreme_with_weight
            ), 
            unobserved_extreme_with_weight_excluding_0=tuple(
                [float(x) for x in xx] 
                for xx in unobserved_extreme_with_weight_excluding_0
            ), 
            solver_index=solver_index, 
            runtime=runtime, 
        )


class ComputePolicyValueByMCIntegration2DMixin:
    def compute_policy_value_by_mc_integration(
            self, 
            repeat=1000, size_per_repeat=1000, 
            seed=31415926, verbose=False, 
    ):
        t0 = time.perf_counter()
        rng = np.random.default_rng(seed)
        mc_policy_value_baseline = np.zeros(repeat)
        mc_policy_value_new      = np.zeros(repeat)
        mc_policy_value_oracle   = np.zeros(repeat)
        for i in range(repeat):
            context_arm  = rng.standard_normal((size_per_repeat, self.num_arms, self.context_dim))
            context_user = rng.dirichlet(alpha=np.ones(self.context_dim), size=size_per_repeat)[:,:,None]
            covariate = np.matmul(context_arm, context_user).squeeze(2)
            p_reward = scipy.special.expit(self.logistic_beta_true*covariate)
            pmf_baseline = self._compute_pmf_by_policy_rule(
                self.logistic_models_baseline, covariate, self.policy_param_baseline, 
            )
            pmf_new = self._compute_pmf_by_policy_rule(
                self.logistic_models_new, covariate, self.policy_param_new, 
            )
            mc_policy_value_baseline[i] = np.mean(np.sum(p_reward * pmf_baseline, axis=1))
            mc_policy_value_new[i]      = np.mean(np.sum(p_reward * pmf_new,      axis=1))
            mc_policy_value_oracle[i]   = np.mean(np.max(p_reward, axis=1))

        policy_value_baseline = np.mean(mc_policy_value_baseline)
        policy_value_new = np.mean(mc_policy_value_new)
        policy_value_oracle = np.mean(mc_policy_value_oracle)
        mc_sample_size=repeat*size_per_repeat
        t1 = time.perf_counter()
        runtime = t1 - t0

        policy_value_vector = (policy_value_baseline, policy_value_new)
        policy_value_difference = policy_value_new - policy_value_baseline

        self.mc_result = self._MCResult(
            policy_value_vector=tuple(float(x) for x in policy_value_vector), 
            policy_value_difference=float(policy_value_difference), 
            policy_value_baseline=float(policy_value_baseline), 
            policy_value_new=float(policy_value_new), 
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
                f"    {policy_value_difference:.7g} (difference)\n"
                f"    {policy_value_baseline:.7g} (baseline)\n"
                f"    {policy_value_new:.7g} (new)\n"
                f"    {policy_value_oracle:.7g} (oracle)\n"
                f"Parameters\n"
                f"    MC sample size: {mc_sample_size:,} ({repeat:,} repetitions of subsample size {size_per_repeat:,} each)\n"
                f"    Random seed: {seed}\n"
                f"Runtime\n"
                f"    {formatted_time}\n"
            )

