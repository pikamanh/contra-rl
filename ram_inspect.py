#!/usr/bin/env python3
"""Play manually while inspecting Contra RAM addresses."""

import argparse
import time

import gym
import nes_py._rom as _nes_rom
import numpy as np
import pygame
from nes_py.wrappers import JoypadSpace

import Contra  # noqa: F401 - registers Contra-v0
from Contra.actions import COMPLEX_MOVEMENT, RIGHT_ONLY, SIMPLE_MOVEMENT


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


def parse_int(value):
    return int(value, 16) if value.lower().startswith("0x") else int(value)


def parse_watch(values):
    addresses = []
    for value in values:
        if ":" in value:
            start, end = value.split(":", 1)
            addresses.extend(range(parse_int(start), parse_int(end) + 1))
        else:
            addresses.append(parse_int(value))
    return sorted(set(addresses))


def parse_args():
    parser = argparse.ArgumentParser(description="Inspect Contra RAM while playing")
    parser.add_argument("--action-set", choices=ACTION_SETS.keys(), default="complex")
    parser.add_argument("--scale", type=int, default=3)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument(
        "--watch",
        nargs="*",
        default=["0x0016:0x001A", "0x0200:0x021F"],
        help="RAM addresses/ranges to show, e.g. --watch 0x0016 0x0200:0x020F",
    )
    parser.add_argument(
        "--log-changes",
        action="store_true",
        help="Print watched RAM changes to the terminal",
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


def oam_entries(ram):
    entries = []
    for address in range(0x0200, 0x0300, 4):
        y = int(ram[address])
        tile = int(ram[address + 1])
        attr = int(ram[address + 2])
        x = int(ram[address + 3])
        if y < 240 and tile != 0:
            entries.append((address, y, tile, attr, x))
    return entries


def watched_lines(ram, addresses, baseline):
    lines = []
    row = []
    for address in addresses:
        value = int(ram[address])
        base = baseline[address]
        marker = "*" if value != base else " "
        row.append(f"{address:04X}:{value:03d}/{value:02X}{marker}")
        if len(row) == 4:
            lines.append("  ".join(row))
            row = []
    if row:
        lines.append("  ".join(row))
    return lines


def draw_overlay(screen, font, lines):
    line_height = font.get_height() + 4
    panel_height = min(screen.get_height(), line_height * len(lines) + 12)
    panel = pygame.Surface((screen.get_width(), panel_height), pygame.SRCALPHA)
    panel.fill((0, 0, 0, 185))
    screen.blit(panel, (0, 0))

    max_lines = (panel_height - 12) // line_height
    for index, line in enumerate(lines[:max_lines]):
        text = font.render(line, True, (255, 255, 255))
        screen.blit(text, (8, 6 + index * line_height))


def draw_frame(screen, frame, scale, font, overlay_lines):
    surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
    if scale != 1:
        width, height = frame.shape[1] * scale, frame.shape[0] * scale
        surface = pygame.transform.scale(surface, (width, height))
    screen.blit(surface, (0, 0))
    draw_overlay(screen, font, overlay_lines)
    pygame.display.flip()


def main():
    args = parse_args()
    actions = ACTION_SETS[args.action_set]
    addresses = parse_watch(args.watch)

    pygame.init()
    pygame.display.set_caption("Contra RAM inspect")

    env = gym.make("Contra-v0")
    env = JoypadSpace(env, actions)
    frame = env.reset()
    raw_env = env.unwrapped

    baseline = [int(value) for value in raw_env.ram]
    previous = {address: baseline[address] for address in addresses}

    height, width = frame.shape[:2]
    screen = pygame.display.set_mode((width * args.scale, height * args.scale))
    font = pygame.font.Font(None, 20)
    clock = pygame.time.Clock()

    print("RAM inspect controls")
    print("  Arrow keys : move/aim")
    print("  Z          : A / jump")
    print("  X          : B / shoot")
    print("  R          : reset baseline")
    print("  P          : print OAM snapshot")
    print("  Esc/Q      : quit")

    running = True
    done = False
    reward = 0.0
    info = {}
    action = 0

    try:
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        running = False
                    elif event.key == pygame.K_r:
                        frame = env.reset()
                        baseline = [int(value) for value in raw_env.ram]
                        previous = {address: baseline[address] for address in addresses}
                        done = False
                    elif event.key == pygame.K_p:
                        print("OAM", oam_entries(raw_env.ram)[:40])

            if not done:
                action = action_index_for(pressed_buttons(), actions)
                frame, reward, done, info = env.step(action)

            if args.log_changes:
                for address in addresses:
                    value = int(raw_env.ram[address])
                    if value != previous[address]:
                        print(
                            f"{address:04X}: {previous[address]:03d}/{previous[address]:02X}"
                            f" -> {value:03d}/{value:02X}"
                        )
                        previous[address] = value

            entries = oam_entries(raw_env.ram)
            overlay_lines = [
                f"action={'+'.join(actions[action]) if not done else 'DONE'} "
                f"reward={float(reward):.1f} x={info.get('x_pos', '?')} "
                f"score={info.get('score', '?')} lives={info.get('life', '?')}",
                "watched RAM: decimal/hex, * means changed from reset baseline",
                *watched_lines(raw_env.ram, addresses, baseline),
                f"OAM visible sprites={len(entries)} first={entries[:4]}",
                "R=reset baseline  P=print OAM  Esc/Q=quit",
            ]

            draw_frame(screen, frame, args.scale, font, overlay_lines)

            if done:
                time.sleep(0.05)
            clock.tick(args.fps)
    finally:
        env.close()
        pygame.quit()


if __name__ == "__main__":
    main()
