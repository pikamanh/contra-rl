#!/usr/bin/env python3
"""
Evaluate a trained PPO agent on Contra and print statistics.

Examples
--------
python eval.py --model-path models/best/best_model.zip
python eval.py --model-path models/best/best_model.zip --n-episodes 20 --render
python eval.py --model-path models/checkpoints/checkpoint_epoch_50.zip --action-set complex
"""
import argparse

import numpy as np
from tqdm import tqdm
from stable_baselines3 import PPO

from env_utils import make_vec_env


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a trained Contra PPO agent")
    p.add_argument("--model-path", type=str, required=True, help="Path to model .zip")
    p.add_argument("--n-episodes", type=int, default=10)
    p.add_argument("--action-set", choices=["simple", "complex", "right"], default="simple")
    p.add_argument("--n-stack", type=int, default=4)
    p.add_argument("--render", action="store_true", help="Render the game window")
    p.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic policy (default: deterministic)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    deterministic = not args.stochastic

    env = make_vec_env(action_set=args.action_set, n_envs=1, n_stack=args.n_stack)
    model = PPO.load(args.model_path, env=env)

    print(f"Model   : {args.model_path}")
    print(f"Policy  : {'deterministic' if deterministic else 'stochastic'}")
    print(f"Episodes: {args.n_episodes}\n")

    ep_rewards, ep_lengths, ep_x_pos, ep_scores = [], [], [], []

    for ep in tqdm(range(args.n_episodes), desc="Evaluating"):
        obs = env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        last_info = {}

        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, done, info = env.step(action)
            total_reward += float(reward[0])
            steps += 1
            if info and info[0]:
                last_info = info[0]
            if args.render:
                env.render()

        x_pos = last_info.get("x_pos", 0)
        score = last_info.get("score", 0)
        ep_rewards.append(total_reward)
        ep_lengths.append(steps)
        ep_x_pos.append(x_pos)
        ep_scores.append(score)

        tqdm.write(
            f"  Ep {ep + 1:>3} | reward={total_reward:8.1f} | "
            f"steps={steps:6d} | x_pos={x_pos:4} | score={score}"
        )

    env.close()

    print("\n" + "=" * 52)
    print("Evaluation Summary")
    print("=" * 52)
    print(f"{'Episodes':<20}: {args.n_episodes}")
    print(f"{'Mean reward':<20}: {np.mean(ep_rewards):.2f} ± {np.std(ep_rewards):.2f}")
    print(f"{'Best reward':<20}: {np.max(ep_rewards):.2f}")
    print(f"{'Worst reward':<20}: {np.min(ep_rewards):.2f}")
    print(f"{'Mean steps':<20}: {np.mean(ep_lengths):.0f}")
    print(f"{'Mean x_pos':<20}: {np.mean(ep_x_pos):.1f}")
    print(f"{'Max x_pos':<20}: {np.max(ep_x_pos):.1f}")
    print(f"{'Mean score':<20}: {np.mean(ep_scores):.1f}")
    print(f"{'Max score':<20}: {np.max(ep_scores)}")
    print("=" * 52)


if __name__ == "__main__":
    main()
