#!/usr/bin/env python3
"""Play Contra manually with keyboard input, without an RL model."""

import argparse
import time

import gym
import numpy as np
import pygame
import nes_py._rom as _nes_rom
from nes_py.wrappers import JoypadSpace

import Contra  # noqa: F401 - registers Contra-v0 with gym
from Contra.actions import COMPLEX_MOVEMENT, RIGHT_ONLY, SIMPLE_MOVEMENT


# nes-py 8.2.1 can keep ROM header bytes as numpy.uint8, causing overflow when
# multiplying by 1024. This matches the compatibility patch used in env_utils.py.
_nes_rom.ROM.prg_rom_size = property(lambda self: 16 * int(self.header[4]))
_nes_rom.ROM.chr_rom_size = property(lambda self: 8 * int(self.header[5]))

ACTION_SETS = {
    "simple": SIMPLE_MOVEMENT,
    "complex": COMPLEX_MOVEMENT,
    "right": RIGHT_ONLY,
}

KEY_BINDINGS = {
    pygame.K_RIGHT: "right",
    pygame.K_LEFT: "left",
    pygame.K_UP: "up",
    pygame.K_DOWN: "down",
    pygame.K_z: "A",
    pygame.K_x: "B",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Play Contra manually")
    parser.add_argument(
        "--action-set",
        choices=ACTION_SETS.keys(),
        default="complex",
        help="Allowed button combinations. Use complex to test most buttons.",
    )
    parser.add_argument("--scale", type=int, default=3, help="Window scale factor")
    parser.add_argument("--fps", type=int, default=60, help="Target FPS")
    parser.add_argument(
        "--no-hud",
        action="store_true",
        help="Hide reward/info text in the pygame window title",
    )
    return parser.parse_args()


def pressed_buttons():
    keys = pygame.key.get_pressed()
    buttons = {button for key, button in KEY_BINDINGS.items() if keys[key]}

    # Contra cannot use opposite directions at the same time.
    if "left" in buttons and "right" in buttons:
        buttons.discard("left")
        buttons.discard("right")
    if "up" in buttons and "down" in buttons:
        buttons.discard("up")
        buttons.discard("down")

    return buttons


def action_index_for(buttons, actions):
    """Return the best matching action index for the currently held buttons."""
    if not buttons:
        return 0

    exact = list(buttons)
    for index, action in enumerate(actions):
        if set(action) == buttons:
            return index

    # Some action sets do not include every combination. Prefer the valid action
    # that preserves the most currently held buttons.
    best_index = 0
    best_score = -1
    for index, action in enumerate(actions):
        action_buttons = set(action)
        if not action_buttons.issubset(buttons):
            continue

        score = len(action_buttons)
        if action_buttons == set(exact):
            score += 10
        if score > best_score:
            best_score = score
            best_index = index

    return best_index


def draw_frame(screen, frame, scale):
    surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
    if scale != 1:
        width, height = frame.shape[1] * scale, frame.shape[0] * scale
        surface = pygame.transform.scale(surface, (width, height))
    screen.blit(surface, (0, 0))
    pygame.display.flip()


def main():
    args = parse_args()
    actions = ACTION_SETS[args.action_set]

    pygame.init()
    pygame.display.set_caption("Contra manual play")

    env = gym.make("Contra-v0")
    env = JoypadSpace(env, actions)

    frame = env.reset()
    height, width = frame.shape[:2]
    screen = pygame.display.set_mode((width * args.scale, height * args.scale))
    clock = pygame.time.Clock()

    print("Manual Contra controls")
    print("  Arrow keys : move/aim")
    print("  Z          : A / jump")
    print("  X          : B / shoot")
    print("  R          : reset after game over")
    print("  Esc/Q      : quit")
    print(f"  Action set : {args.action_set} ({len(actions)} actions)")

    done = False
    running = True
    last_reward = 0.0
    info = {}

    try:
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        running = False
                    elif event.key == pygame.K_r and done:
                        frame = env.reset()
                        done = False
                        last_reward = 0.0
                        info = {}

            if not done:
                buttons = pressed_buttons()
                action = action_index_for(buttons, actions)
                frame, last_reward, done, info = env.step(action)

            draw_frame(screen, frame, args.scale)

            if not args.no_hud:
                action_name = "+".join(actions[action]) if not done else "DONE"
                pygame.display.set_caption(
                    "Contra manual play | "
                    f"action={action_name} reward={last_reward:.1f} "
                    f"x={info.get('x_pos', '?')} lives={info.get('life', '?')} "
                    f"score={info.get('score', '?')}"
                )

            if done:
                time.sleep(0.05)
            clock.tick(args.fps)
    finally:
        env.close()
        pygame.quit()


if __name__ == "__main__":
    main()
