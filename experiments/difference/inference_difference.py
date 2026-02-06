import numpy as np
import pandas as pd
import argparse
import os
import time
from ctxbandit import PolicyInferenceDifference


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--replicate_start', type=int)
    parser.add_argument('--replicate_end',   type=int)
    parser.add_argument('--output_path',     type=str)
    args = parser.parse_args()

    for arg in vars(args):
        print(f'{arg} = {getattr(args, arg)}')
    print()

    # policy parameters - baseline
    size_learning_baseline = 256
    seed_learning_baseline = 31415926    # Pi
    scale_se, power_ub, num_arms_top = 1, 2, 3
    policy_param_baseline = scale_se, power_ub, num_arms_top

    # policy parameters - new
    size_learning_new = 1024
    seed_learning_new = 31415926    # Pi
    scale_se, power_ub, num_arms_top = 1, 1, 1
    policy_param_new = scale_se, power_ub, num_arms_top

    # contextual bandit problem setting
    num_arms = 10 
    context_dim = 12

    # importance dataset for inference
    size_inference = 1024
    seed_inference = 11235813    # Fibonacci

    replicate_lst = list(range(args.replicate_start, args.replicate_end+1))
    size_seq = np.rint(np.geomspace(8, size_inference, num=10)).astype(int)
    margin_lst = [0, 0.05, 0.1]

    prob_improvement = []
    for replicate in replicate_lst:
        t0 = time.perf_counter()

        # create inference model
        inference = PolicyInferenceDifference(
            num_arms, context_dim, 
            size_inference, seed_inference + replicate, 
            size_learning_baseline, seed_learning_baseline, policy_param_baseline, 
            size_learning_new, seed_learning_new, policy_param_new, 
        )

        # inference for each sample size
        for size in size_seq:
            inference.clear_results_for_all_sizes()
            inference.compute_mle(size)
            inference.compute_elr_over_grid(
                size, 
                num_points=10_000, wilks_level=0.999_9, 
                parallel=False, verbose=False, 
            )
            for margin in margin_lst:
                inference.compute_prob_improvement(size, margin)
                temp = inference.prob_improvement_result_by_size[size]
                prob_improvement.append(temp.margin_to_prob[margin])

        t1 = time.perf_counter()
        runtime = inference._format_runtime(t1-t0)
        print(f"Replicate {replicate:04d} Runtime {runtime}")
    print()

    num_replicates = len(replicate_lst)
    num_sizes = len(size_seq)
    num_margins = len(margin_lst)
    abs_margin_lst = [f"+{margin}" for margin in margin_lst]   # to distinguish from rel_margin
    df = pd.DataFrame(
        {
            "Replicate":   np.repeat(replicate_lst, num_sizes * num_margins), 
            "Sample Size": np.tile(np.repeat(size_seq, num_margins), num_replicates), 
            "Margin":      np.tile(abs_margin_lst, num_replicates * num_sizes), 
            "Probability": prob_improvement, 
        }
    )

    path = args.output_path
    dir = os.path.dirname(path)
    if dir:
        os.makedirs(dir, exist_ok=True)   # create folders if they don't exist

    df.to_csv(path, index=False)


if __name__ == '__main__':    
    main()

