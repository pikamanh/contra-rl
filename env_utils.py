"""Shared environment creation utilities for training, eval, and play."""
import os

import cv2
import gym
import numpy as np
from gym import ObservationWrapper
from gym.spaces import Box
from nes_py.wrappers import JoypadSpace
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecFrameStack,
    VecTransposeImage,
)

import Contra  # noqa: F401 — triggers Contra-v0 registration
from Contra.actions import COMPLEX_MOVEMENT, RIGHT_ONLY, SIMPLE_MOVEMENT

ACTION_SETS = {
    "simple": SIMPLE_MOVEMENT,
    "complex": COMPLEX_MOVEMENT,
    "right": RIGHT_ONLY,
}


class GrayScaleResize(ObservationWrapper):
    """Convert RGB NES frame (240×256×3) → grayscale, resize to target shape."""

    def __init__(self, env, shape=(84, 84)):
        super().__init__(env)
        self.shape = shape
        self.observation_space = Box(
            low=0, high=255, shape=(shape[0], shape[1], 1), dtype=np.uint8
        )

    def observation(self, obs):
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (self.shape[1], self.shape[0]))
        return resized[:, :, np.newaxis]


def make_env(action_set: str = "simple", monitor_dir: str = None, rank: int = 0):
    """Return a factory for a single preprocessed Contra environment."""

    def _init():
        env = gym.make("Contra-v0")
        env = JoypadSpace(env, ACTION_SETS[action_set])
        env = GrayScaleResize(env)
        if monitor_dir:
            os.makedirs(monitor_dir, exist_ok=True)
            env = Monitor(env, os.path.join(monitor_dir, f"env_{rank}"))
        else:
            env = Monitor(env)
        return env

    return _init


def make_vec_env(
    action_set: str = "simple",
    n_envs: int = 1,
    monitor_dir: str = None,
    multiprocess: bool = False,
    n_stack: int = 4,
):
    """
    Create a vectorised, frame-stacked Contra environment.

    Pipeline:
        ContraEnv → JoypadSpace → GrayScaleResize(84×84) →
        VecFrameStack(n_stack) → VecTransposeImage  → (C, H, W) for CnnPolicy
    """
    fns = [make_env(action_set, monitor_dir, rank=i) for i in range(n_envs)]
    VecClass = SubprocVecEnv if (multiprocess and n_envs > 1) else DummyVecEnv
    vec_env = VecClass(fns)
    vec_env = VecFrameStack(vec_env, n_stack=n_stack)
    vec_env = VecTransposeImage(vec_env)
    return vec_env
