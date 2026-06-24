#!/usr/bin/env python3
"""Inspect the exact vectorized training environment without loading a model."""

import argparse
import time

import numpy as np
import pygame

from env_utils import ACTION_SETS, make_vec_env


KEY_BINDINGS = {
    pygame.K_RIGHT: "right",
    pygame.K_LEFT: "left",
    pygame.K_UP: "up",
    pygame.K_DOWN: "down",
    pygame.K_z: "A",
    pygame.K_x: "B",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preview the observation pipeline used during PPO training"
    )
    parser.add_argument("--action-set", choices=ACTION_SETS.keys(), default="simple")
    parser.add_argument("--n-stack", type=int, default=2)
    parser.add_argument("--scale", type=int, default=5)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument(
        "--random",
        action="store_true",
        help="Sample random actions instead of using keyboard input",
    )
    parser.add_argument(
        "--no-overlay",
        action="store_true",
        help="Hide debug text overlay and only show the training frames",
    )
    return parser.parse_args()


def pressed_buttons():
    keys = pygame.key.get_pressed()
    buttons = {button for key, button in KEY_BINDINGS.items() if keys[key]}

    if "left" in buttons and "right" in buttons:
        buttons.discard("left")
        buttons.discard("right")
    if "up" in buttons and "down" in buttons:
        buttons.discard("up")
        buttons.discard("down")

    return buttons


def action_index_for(buttons, actions):
    if not buttons:
        return 0

    for index, action in enumerate(actions):
        if set(action) == buttons:
            return index

    best_index = 0
    best_score = -1
    for index, action in enumerate(actions):
        action_buttons = set(action)
        if not action_buttons.issubset(buttons):
            continue
        score = len(action_buttons)
        if score > best_score:
            best_score = score
            best_index = index

    return best_index


def latest_frame(obs):
    """Extract the newest 84x84 grayscale frame from VecTransposeImage output."""
    frame_stack = obs[0]
    if frame_stack.ndim == 3:
        return frame_stack[-1]
    if frame_stack.ndim == 2:
        return frame_stack
    raise ValueError(f"Unexpected observation shape: {obs.shape}")


def stack_grid(obs):
    """Return a horizontal grid of all stacked grayscale frames."""
    frame_stack = obs[0]
    if frame_stack.ndim != 3:
        return frame_stack

    frames = [frame_stack[i] for i in range(frame_stack.shape[0])]
    separator = np.full((frames[0].shape[0], 2), 180, dtype=np.uint8)
    grid = frames[0]
    for frame in frames[1:]:
        grid = np.concatenate((grid, separator, frame), axis=1)
    return grid


def reward_component_text(info):
    components = info.get("reward_components") or {}
    if not components:
        return "components: n/a"

    keys = ("x", "death", "boss", "score", "enemy_clear", "shoot_enemy")
    return "components: " + " ".join(
        f"{key}={components.get(key, 0)}" for key in keys
    )


def debug_lines(obs, action_name, reward, done, info):
    return [
        f"obs={obs.shape} action={action_name} reward={reward:.1f} done={bool(done[0])}",
        f"active_enemies={info.get('active_enemies', '?')} "
        f"visible_threats={info.get('visible_threat_sprites', '?')} "
        f"x={info.get('x_pos', '?')} y={info.get('y_pos', '?')} "
        f"lives={info.get('life', '?')} score={info.get('score', '?')}",
        f"raw_type_count={info.get('active_enemy_types', '?')} "
        f"enemy_slots={info.get('enemy_slots', '?')} "
        f"baseline={info.get('enemy_baseline', '?')}",
        reward_component_text(info),
        "keys: arrows move/aim, Z=A jump, X=B shoot, R=reset, Esc/Q=quit",
    ]


def draw_overlay(screen, font, lines):
    line_height = font.get_height() + 4
    panel_height = line_height * len(lines) + 12
    panel = pygame.Surface((screen.get_width(), panel_height), pygame.SRCALPHA)
    panel.fill((0, 0, 0, 180))
    screen.blit(panel, (0, 0))

    for index, line in enumerate(lines):
        text = font.render(line, True, (255, 255, 255))
        screen.blit(text, (8, 6 + index * line_height))


def draw_gray(screen, frame, scale, font=None, overlay_lines=None):
    rgb = np.repeat(frame[:, :, None], 3, axis=2)
    surface = pygame.surfarray.make_surface(np.transpose(rgb, (1, 0, 2)))
    if scale != 1:
        width, height = frame.shape[1] * scale, frame.shape[0] * scale
        surface = pygame.transform.scale(surface, (width, height))
    screen.blit(surface, (0, 0))
    if font is not None and overlay_lines:
        draw_overlay(screen, font, overlay_lines)
    pygame.display.flip()


def main():
    args = parse_args()
    actions = ACTION_SETS[args.action_set]

    env = make_vec_env(
        action_set=args.action_set,
        n_envs=1,
        n_stack=args.n_stack,
        multiprocess=False,
    )
    obs = env.reset()

    pygame.init()
    font = pygame.font.Font(None, 24)
    preview = stack_grid(obs)
    height, width = preview.shape[:2]
    screen = pygame.display.set_mode((width * args.scale, height * args.scale))
    clock = pygame.time.Clock()

    print("Training environment preview")
    print(f"  Observation shape : {obs.shape}")
    print(f"  Action set        : {args.action_set} ({len(actions)} actions)")
    print("  Arrow keys        : move/aim")
    print("  Z                 : A / jump")
    print("  X                 : B / shoot")
    print("  R                 : reset")
    print("  Esc/Q             : quit")

    running = True
    action = 0
    reward = 0.0
    info = {}
    done = np.array([False])

    try:
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        running = False
                    elif event.key == pygame.K_r:
                        obs = env.reset()
                        reward = 0.0
                        info = {}
                        done = np.array([False])

            if args.random:
                action = int(env.action_space.sample()[0])
            else:
                action = action_index_for(pressed_buttons(), actions)

            obs, reward_arr, done, info_arr = env.step([action])
            reward = float(reward_arr[0])
            info = info_arr[0]

            if bool(done[0]):
                time.sleep(0.05)

            preview = stack_grid(obs)
            action_name = "+".join(actions[action])
            overlay_lines = None
            if not args.no_overlay:
                overlay_lines = debug_lines(obs, action_name, reward, done, info)
            draw_gray(screen, preview, args.scale, font, overlay_lines)

            pygame.display.set_caption(
                "Train env preview | "
                f"obs={obs.shape} action={action_name} reward={reward:.1f} "
                f"active_enemies={info.get('active_enemies', '?')} "
                f"x={info.get('x_pos', '?')} lives={info.get('life', '?')} "
                f"score={info.get('score', '?')}"
            )
            clock.tick(args.fps)
    finally:
        env.close()
        pygame.quit()


if __name__ == "__main__":
    main()
