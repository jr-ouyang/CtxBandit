from ctxbandit import PolicyInferenceJoint
from ctxbandit import PolicyInferenceDifference
import argparse
import pickle
import os

def main():
    parser = argparse.ArgumentParser()
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
    seed_inference = 11235813 + 89   # Fibonacci + 89

    # create inference models
    joint = PolicyInferenceJoint(
        num_arms, context_dim, 
        size_inference, seed_inference, 
        size_learning_baseline, seed_learning_baseline, policy_param_baseline, 
        size_learning_new, seed_learning_new, policy_param_new, 
        )
    difference = PolicyInferenceDifference(
        num_arms, context_dim, 
        size_inference, seed_inference, 
        size_learning_baseline, seed_learning_baseline, policy_param_baseline, 
        size_learning_new, seed_learning_new, policy_param_new, 
    )

    # compute policy value by mc integration
    joint.compute_policy_value_by_mc_integration(
        repeat=1000, size_per_repeat=1000,
        seed=31415926, verbose=True, 
    )
    difference.compute_policy_value_by_mc_integration(
        repeat=1000, size_per_repeat=1000,
        seed=31415926, verbose=True, 
    )

    # compute mle
    size = 100
    joint.compute_mle(size)
    difference.compute_mle(size)

    # compute elr over a grid
    joint.compute_elr_over_grid(
        size, 
        num_points=1000, wilks_level=0.999_9, 
        parallel=True, num_processes=10, verbose=True, 
    )
    difference.compute_elr_over_grid(
        size, 
        num_points=10_000, wilks_level=0.999_9, 
        parallel=True, num_processes=10, verbose=True, 
    )

    # compute the probability of improvement from the baseline policy to the new policy
    joint.compute_prob_improvement(size, mode="abs", margin=0)
    difference.compute_prob_improvement(size, margin=0)

    # save the two inference models by pickle
    models = {
        "joint": joint, 
        "difference": difference, 
    }
    path = args.output_path
    dir = os.path.dirname(path)
    if dir:
        os.makedirs(dir, exist_ok=True)   # create folders if they don't exist

    with open(path, 'wb') as f:
        pickle.dump(models, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':    
    main()

