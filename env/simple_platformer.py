"""An original, tiny side-scrolling platformer, used in place of
gym-super-mario-bros where the real NES environment can't run (no `nes-py`
build toolchain here, and no ROM to load even if it could). This is not a
copy of Super Mario Bros -- it's a minimal original game with the same
*shape* of problem (run right, jump gaps and obstacles, avoid an enemy,
reach a flag) so the rest of the pipeline (interface/sensory.py's feature
extractor, interface/motor.py's action set, learning/actor.py's
actor-critic) can plug in unchanged.

Exposes the same `info` dict keys FeatureExtractor expects: x_pos, y_pos,
x_vel, y_vel, enemies, gap_ahead, obstacle_ahead, airborne, flag_get.
"""
from __future__ import annotations

import dataclasses

from interface.motor import SIMPLE_MOVEMENT


@dataclasses.dataclass
class Level:
    length: int = 60           # tiles
    gaps: tuple = ((14, 1), (34, 1))          # (start_tile, width) pits
    obstacles: tuple = (22, 44)                # tile x positions of a jump-or-die block
    enemy_start: int = 28                      # tile x of a single patrolling enemy
    enemy_range: int = 4                       # patrol half-width, tiles
    flag_tile: int = 58


TILE_PX = 16
GRAVITY = -1.6
JUMP_VY = 13.0   # tuned so a single jump's horizontal carry (~36px) clears a
                 # 1-tile (16px) gap with margin even from the earliest point
                 # in the gap_ahead lookahead window -- see _gap_ahead's note.
WALK_VX = 2.2
RUN_VX = 3.6


class SimplePlatformer:
    """Gym-like step/reset interface, tile-based physics, deterministic."""

    def __init__(self, level: Level | None = None, max_steps: int = 600):
        self.level = level or Level()
        self.max_steps = max_steps
        self.reset()

    def reset(self):
        self.x = 1.0 * TILE_PX
        self.y = 0.0
        self.vx = 0.0
        self.vy = 0.0
        self.on_ground = True
        self.enemy_x = self.level.enemy_start * TILE_PX
        self.enemy_dir = 1
        self.t = 0
        self.dead = False
        self.won = False
        return self._obs(), self._info(reward_dx=0)

    def _tile_at(self, px: float) -> int:
        # tiny epsilon avoids float-rounding flicker exactly at a tile
        # boundary (e.g. accumulated += steps landing on 368.0 vs
        # 367.999999996) changing which tile a collision check sees.
        return int((px + 1e-6) // TILE_PX)

    def _in_gap(self, tile: int) -> bool:
        for start, width in self.level.gaps:
            if start <= tile < start + width:
                return True
        return False

    def _obstacle_ahead(self) -> bool:
        tile = self._tile_at(self.x)
        return any(tile < o <= tile + 1 for o in self.level.obstacles)

    def _gap_ahead(self) -> bool:
        # Narrow lookahead by design: a wider window (e.g. "gap within the
        # next 3 tiles") caused the agent to start hopping as soon as it
        # became true and land mid-air-arc precisely back on the gap tile a
        # jump-arc-period later, from repeated re-triggering while grounded
        # between hops. Keeping this to "immediately adjacent" means at most
        # one or two jump-triggering steps right at the edge, so a single
        # jump's horizontal carry reliably clears the gap.
        tile = self._tile_at(self.x)
        for start, width in self.level.gaps:
            if tile < start <= tile + 1:
                return True
        return False

    def step(self, action_index: int):
        prev_x = self.x
        buttons = SIMPLE_MOVEMENT[action_index]
        want_right = "right" in buttons
        want_left = "left" in buttons
        want_run = "B" in buttons
        want_jump = "A" in buttons

        target_vx = 0.0
        if want_right:
            target_vx = RUN_VX if want_run else WALK_VX
        elif want_left:
            target_vx = -WALK_VX
        self.vx = target_vx

        if want_jump and self.on_ground:
            self.vy = JUMP_VY
            self.on_ground = False

        # physics
        self.x += self.vx
        self.x = max(self.x, 0.0)
        self.vy += GRAVITY
        self.y += self.vy
        if self.y <= 0:
            self.y = 0.0
            self.vy = 0.0
            self.on_ground = True
        else:
            self.on_ground = False

        # enemy patrol
        self.enemy_x += self.enemy_dir * 0.9
        lo = (self.level.enemy_start - self.level.enemy_range) * TILE_PX
        hi = (self.level.enemy_start + self.level.enemy_range) * TILE_PX
        if self.enemy_x < lo or self.enemy_x > hi:
            self.enemy_dir *= -1

        tile = self._tile_at(self.x)
        # fall in a gap while grounded-height => death
        if self._in_gap(tile) and self.y <= 0.5:
            self.dead = True
        # hit an obstacle block while not jumped high enough over it => death
        if tile in self.level.obstacles and self.y < TILE_PX * 0.6:
            self.dead = True
        # enemy collision (must be airborne with enough clearance to "avoid", else death)
        if abs(self.x - self.enemy_x) < TILE_PX * 0.6 and self.y < TILE_PX * 0.8:
            self.dead = True
        if tile >= self.level.flag_tile:
            self.won = True

        self.t += 1
        done = self.dead or self.won or self.t >= self.max_steps or tile >= self.level.length

        dx = self.x - prev_x
        return self._obs(), self._info(reward_dx=dx), done

    def _obs(self):
        return None  # no pixel observation in this env; features come from info

    def _info(self, reward_dx: float):
        return {
            "x_pos": self.x,
            "y_pos": self.y,
            "x_vel": self.vx,
            "y_vel": self.vy,
            "enemies": [(self.enemy_x, 0.0)],
            "gap_ahead": self._gap_ahead(),
            "obstacle_ahead": self._obstacle_ahead(),
            "airborne": not self.on_ground,
            "flag_get": self.won,
            "dead": self.dead,
            "dx": reward_dx,
            "tile_x": self._tile_at(self.x),
        }
