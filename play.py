#!/usr/bin/env python3
"""
Watch a trained PPO agent play Contra in real-time.

Examples
--------
python play.py --model-path models/best/best_model.zip
python play.py --model-path models/best/best_model.zip --n-episodes 5 --fps 30
python play.py --model-path models/best/best_model.zip --action-set complex --stochastic
"""
import argparse
import time

import numpy as np
from stable_baselines3 import PPO

from env_utils import make_vec_env


def parse_args():
    p = argparse.ArgumentParser(description="Watch a trained Contra PPO agent play in real-time")
    p.add_argument("--model-path", type=str, required=True, help="Path to model .zip")
    p.add_argument("--n-episodes", type=int, default=3)
    p.add_argument("--action-set", choices=["simple", "complex", "right"], default="simple")
    p.add_argument("--n-stack", type=int, default=4)
    p.add_argument("--fps", type=int, default=60, help="Target playback FPS (0 = unlimited)")
    p.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic policy (default: deterministic)",
    )
    return p.parse_args()


def _fmt_info(info: dict) -> str:
    return (
        f"x={info.get('x_pos', '?'):>3}  "
        f"y={info.get('y_pos', '?'):>3}  "
        f"lives={info.get('life', '?')}  "
        f"score={info.get('score', 0):>8}  "
        f"dead={info.get('dead', False)}"
    )


def main():
    args = parse_args()
    deterministic = not args.stochastic
    frame_time = (1.0 / args.fps) if args.fps > 0 else 0.0

    env = make_vec_env(action_set=args.action_set, n_envs=1, n_stack=args.n_stack)
    model = PPO.load(args.model_path, env=env)

    print("=" * 60)
    print("Contra – Real-time AI Play")
    print("=" * 60)
    print(f"  Model     : {args.model_path}")
    print(f"  Policy    : {'deterministic' if deterministic else 'stochastic'}")
    print(f"  Episodes  : {args.n_episodes}")
    print(f"  Target FPS: {args.fps if args.fps > 0 else 'unlimited'}")
    print("=" * 60)
    print("(Close the game window or press Ctrl+C to stop)\n")

    episode_rewards = []

    try:
        for ep in range(args.n_episodes):
            obs = env.reset()
            done = False
            total_reward = 0.0
            steps = 0
            last_info = {}

            print(f"── Episode {ep + 1}/{args.n_episodes} ──")

            while not done:
                t0 = time.perf_counter()

                action, _ = model.predict(obs, deterministic=deterministic)
                obs, reward, done, info = env.step(action)
                total_reward += float(reward[0])
                steps += 1

                env.render()

                if info and info[0]:
                    last_info = info[0]
                    print(f"\r  step={steps:6d} | reward={total_reward:9.2f} | {_fmt_info(last_info)}", end="", flush=True)

                # Rate-limit to target FPS
                elapsed = time.perf_counter() - t0
                if frame_time > 0 and elapsed < frame_time:
                    time.sleep(frame_time - elapsed)

            episode_rewards.append(total_reward)
            x_final = last_info.get("x_pos", 0)
            score_final = last_info.get("score", 0)
            defeated = last_info.get("defeated", False)

            print(
                f"\n  Episode {ep + 1} ended: "
                f"reward={total_reward:.2f}  steps={steps}  "
                f"x_pos={x_final}  score={score_final}  "
                f"boss_defeated={defeated}\n"
            )

    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")

    env.close()

    if episode_rewards:
        print("─" * 40)
        print(f"Played {len(episode_rewards)} episode(s)")
        print(f"Mean reward : {np.mean(episode_rewards):.2f}")
        print(f"Best reward : {np.max(episode_rewards):.2f}")


if __name__ == "__main__":
    main()
