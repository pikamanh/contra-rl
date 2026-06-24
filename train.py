#!/usr/bin/env python3
"""
Train a PPO agent to play Contra.

Examples
--------
# Basic run (4 envs, 1M steps, save every 25 epochs)
python train.py

# Resume from a previous checkpoint
python train.py --resume models/latest/latest_model.zip

# Complex action set, 8 envs, save zip archives every 25 epochs
python train.py --action-set complex --n-envs 8 --zip

# Quick smoke-test (50k steps)
python train.py --total-timesteps 50000 --n-envs 1
"""
import argparse
import os

from stable_baselines3 import PPO

from callbacks import ContraCallback
from env_utils import make_vec_env


def parse_args():
    p = argparse.ArgumentParser(description="Train PPO on Contra")

    # Environment
    p.add_argument("--action-set", choices=["simple", "complex", "right"], default="simple")
    p.add_argument("--n-envs", type=int, default=4, help="Number of parallel environments")
    p.add_argument("--n-stack", type=int, default=4, help="Frames to stack as observation")
    p.add_argument(
        "--multiprocess",
        action="store_true",
        help="Use SubprocVecEnv (true parallelism, faster on multi-core)",
    )

    # PPO hyperparameters
    p.add_argument("--total-timesteps", type=int, default=1_000_000)
    p.add_argument("--n-steps", type=int, default=512, help="Steps per rollout per env")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--n-epochs", type=int, default=4, help="Gradient epochs per PPO update")
    p.add_argument("--lr", type=float, default=2.5e-4, help="Learning rate")
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--ent-coef", type=float, default=0.01, help="Entropy coefficient")

    # Checkpointing
    p.add_argument("--save-dir", type=str, default="models")
    p.add_argument(
        "--save-freq-epochs",
        type=int,
        default=25,
        help="Save checkpoint every N PPO epochs (rollout iterations)",
    )
    p.add_argument(
        "--zip",
        action="store_true",
        help="Bundle saved models into a zip archive at each checkpoint and at the end",
    )

    # Resume
    p.add_argument("--resume", type=str, default=None, help="Path to model .zip to resume from")

    p.add_argument("--verbose", type=int, default=1, choices=[0, 1, 2])
    return p.parse_args()


def main():
    args = parse_args()

    # Validate batch size
    steps_per_epoch = args.n_steps * args.n_envs
    if steps_per_epoch % args.batch_size != 0:
        raise ValueError(
            f"n_steps ({args.n_steps}) × n_envs ({args.n_envs}) = {steps_per_epoch} "
            f"must be divisible by batch_size ({args.batch_size})."
        )

    os.makedirs(args.save_dir, exist_ok=True)
    log_dir = os.path.join(args.save_dir, "logs")
    monitor_dir = os.path.join(args.save_dir, "monitor")

    print("=" * 60)
    print("Contra PPO Training")
    print("=" * 60)
    print(f"  Action set      : {args.action_set}")
    print(f"  Environments    : {args.n_envs}  (multiprocess={args.multiprocess})")
    print(f"  Frame stack     : {args.n_stack}")
    print(f"  Total timesteps : {args.total_timesteps:,}")
    print(f"  n_steps/env     : {args.n_steps}  → {steps_per_epoch:,} steps/epoch")
    print(f"  Checkpoint every: {args.save_freq_epochs} epochs")
    print(f"  Save directory  : {args.save_dir}")
    print(f"  Zip archives    : {args.zip}")
    print("=" * 60)

    print("\nBuilding environments...")
    train_env = make_vec_env(
        action_set=args.action_set,
        n_envs=args.n_envs,
        monitor_dir=monitor_dir,
        multiprocess=args.multiprocess,
        n_stack=args.n_stack,
    )

    if args.resume:
        print(f"Resuming from: {args.resume}")
        model = PPO.load(
            args.resume,
            env=train_env,
            tensorboard_log=log_dir,
            verbose=args.verbose,
        )
    else:
        model = PPO(
            policy="CnnPolicy",
            env=train_env,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            learning_rate=args.lr,
            gamma=args.gamma,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=args.ent_coef,
            vf_coef=0.5,
            max_grad_norm=0.5,
            tensorboard_log=log_dir,
            verbose=args.verbose,
        )

    callback = ContraCallback(
        save_dir=args.save_dir,
        save_freq_epochs=args.save_freq_epochs,
        save_zip=args.zip,
        log_dir=log_dir,
        verbose=args.verbose,
    )

    print("\nStarting training...\n")
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callback,
        reset_num_timesteps=not bool(args.resume),
        tb_log_name="PPO_Contra",
        progress_bar=True,
    )

    train_env.close()
    print("\nDone. Models saved to:", args.save_dir)
    print("  Best model  :", os.path.join(args.save_dir, "best", "best_model.zip"))
    print("  Latest model:", os.path.join(args.save_dir, "latest", "latest_model.zip"))
    print(f"\nTo view training curves:\n  tensorboard --logdir {log_dir}")


if __name__ == "__main__":
    main()
