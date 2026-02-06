import numpy as np
import pandas as pd
import argparse
import os
import time
from ctxbandit import PolicyInferenceSingle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy',          type=str)
    parser.add_argument('--replicate_start', type=int)
    parser.add_argument('--replicate_end',   type=int)
    parser.add_argument('--output_path',     type=str)
    args = parser.parse_args()

    for arg in vars(args):
        print(f'{arg} = {getattr(args, arg)}')
    print()

    if args.policy == "baseline":
        size_learning = 256
        seed_learning = 31415926    # Pi
        scale_se, power_ub, num_arms_top = 1, 2, 3
        policy_param = scale_se, power_ub, num_arms_top
    elif args.policy == "new":
        size_learning = 1024
        seed_learning = 31415926    # Pi
        scale_se, power_ub, num_arms_top = 1, 1, 1
        policy_param = scale_se, power_ub, num_arms_top
    else:
        raise ValueError

    # contextual bandit problem setting
    num_arms = 10 
    context_dim = 12

    # importance dataset for inference
    size_inference = 1024
    seed_inference = 11235813    # Fibonacci

    replicate_lst = list(range(args.replicate_start, args.replicate_end+1))
    size_seq = np.rint(np.geomspace(8, size_inference, num=10)).astype(int)
    level_lst = [0.95, 0.90]

    # compute policy value by mc integration once
    inference = PolicyInferenceSingle(
        num_arms, context_dim, 
        size_inference, seed_inference, 
        size_learning, seed_learning, policy_param, 
    )
    inference.compute_policy_value_by_mc_integration(
        repeat=1000, size_per_repeat=1000,
        seed=31415926, verbose=True, 
    )
    mc_result = inference.mc_result

    ci_cover    = []
    ci_position = []
    ci_width    = []
    for replicate in replicate_lst:
        t0 = time.perf_counter()

        # create inference model
        inference = PolicyInferenceSingle(
            num_arms, context_dim, 
            size_inference, seed_inference + replicate, 
            size_learning, seed_learning, policy_param, 
        )

        # use the pre-computed mc_result
        inference.mc_result = mc_result

        # inference for each sample size
        for size in size_seq:
            inference.clear_results_for_all_sizes()
            inference.compute_mle(size)
            inference.compute_elr_over_grid(
                size, 
                num_points=10_000, wilks_level=0.999_9, 
                parallel=False, verbose=False, 
            )
            for level in level_lst:
                inference.compute_wilks_interval(size, level)
                inference.compute_hpd_interval(size, level)
                wilks_result = inference.wilks_result_by_size[size]
                hpd_result = inference.hpd_result_by_size[size]
                wilks = getattr(wilks_result, f"wilks_{int(level*100)}")
                hpd   = getattr(hpd_result,   f"hpd_{int(level*100)}")
                ci_cover.extend([int(wilks.cover), int(hpd.cover)])
                ci_position.extend([wilks.position, hpd.position])
                ci_width.extend([wilks.width, hpd.width])

        t1 = time.perf_counter()
        runtime = inference._format_runtime(t1-t0)
        print(f"Replicate {replicate:04d} Runtime {runtime}")
    print()

    num_replicates = len(replicate_lst)
    num_sizes = len(size_seq)
    num_levels = len(level_lst)
    df = pd.DataFrame(
        {
            "Replicate":   np.repeat(replicate_lst, num_sizes * num_levels * 2), 
            "Sample Size": np.tile(np.repeat(size_seq, num_levels * 2), num_replicates), 
            "Level":       np.tile(np.repeat(level_lst, 2), num_replicates * num_sizes), 
            "Type":        np.tile(["Wilks", "HPD"], num_replicates * num_sizes * 2), 
            "Cover":       ci_cover, 
            "Position":    ci_position,  
            "Width":       ci_width, 
        }
    )

    path = args.output_path
    dir = os.path.dirname(path)
    if dir:
        os.makedirs(dir, exist_ok=True)   # create folders if they don't exist

    df.to_csv(path, index=False)


if __name__ == '__main__':    
    main()

