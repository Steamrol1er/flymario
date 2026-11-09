"""Thin wrapper around the real Atari Pong environment (`ale-py`), used
as-is: the same 128-byte RAM observation the network reads and the same
6-action controller interface the actual game exposes -- nothing
re-derived or renamed. See docs/PONG.md for why this game, how the ROM
was obtained, what was researched, and what was found along the way.

Native interface (unchanged from `gymnasium.make("ALE/Pong-v5", ...)`):
    action:  Discrete(6)  -- NOOP, FIRE, RIGHT, LEFT, RIGHTFIRE, LEFTFIRE
    reward:  +1 (you score), -1 (opponent scores), 0 otherwise
    ram:     uint8[128]   -- the raw Atari 2600 RAM, fed to the KC
                             projection directly, unmodified

Defaults to `obs_type="ram"` (no per-step RGB rendering cost) since the
live demo (live/game.html) draws its own crisp vector reconstruction from
RAM-derived paddle/ball positions rather than the emulator's raw pixel
frame -- that switch is also what fixed the live view's stutter (encoding
and transferring a re-rendered PNG over HTTP on every simulation step was
the actual bottleneck, not the emulator). Pass `render_rgb=True` if you
specifically want the real emulator frame too (e.g. to capture a
screenshot for documentation, as in `live/assets/sample_frame.png`).

`difficulty=0` (ALE's default, kept explicit here) is empirically the
HARDEST of the 4 available settings, not the easiest: a fixed random
policy scored 0/15 points against it, vs. up to 13% at difficulty 3 (see
docs/PONG.md). That's the closest thing to "the perfect bot" this game
exposes, and it's what's used throughout.

The RAM addresses read out for the replay/live views (paddle/ball
positions) were found empirically for this ale-py build by moving the
paddle and watching which bytes changed, not copied from a possibly-stale
published RAM map -- see the constants below.
"""
from __future__ import annotations

import gymnasium as gym
import ale_py

gym.register_envs(ale_py)

ACTION_MEANINGS = ["NOOP", "FIRE", "RIGHT", "LEFT", "RIGHTFIRE", "LEFTFIRE"]
N_ACTIONS = len(ACTION_MEANINGS)

RAM_PLAYER_Y = 51   # confirmed: moves directly with our own actions
RAM_BALL_Y = 54     # ball y (AtariARI annotation; also corr 0.99 with RAM[50], the CPU paddle that tracks it)
RAM_BALL_X = 49     # ball x (AtariARI annotation)
RAM_ENEMY_Y = 50    # CPU paddle y. These three were previously mislabeled (49/50/54 rotated); fixed, see docs/PONG.md


class PongEnv:
    def __init__(self, frame_skip: int = 1, difficulty: int = 0, render_rgb: bool = False):
        obs_type = "rgb" if render_rgb else "ram"
        self.render_rgb = render_rgb
        self.env = gym.make("ALE/Pong-v5", obs_type=obs_type, frameskip=frame_skip, difficulty=difficulty)
        assert self.env.action_space.n == N_ACTIONS
        assert list(self.env.unwrapped.get_action_meanings()) == ACTION_MEANINGS

    def _ram(self):
        return self.env.unwrapped.ale.getRAM()

    def reset(self, seed: int | None = None):
        # gymnasium seeds the underlying RNG stream on a seeded reset; every
        # later unseeded reset continues that same seeded stream, so callers
        # only need to pass `seed` once, at the very start of a run
        obs, info = self.env.reset(seed=seed)
        ram = obs if not self.render_rgb else self._ram()
        return ram, self._augment_info(info, ram, obs if self.render_rgb else None)

    def step(self, action_index: int):
        obs, reward, terminated, truncated, info = self.env.step(action_index)
        done = terminated or truncated
        ram = obs if not self.render_rgb else self._ram()
        info = self._augment_info(info, ram, obs if self.render_rgb else None)
        return ram, info, float(reward), done

    def _augment_info(self, info: dict, ram, frame) -> dict:
        info = dict(info)
        info["ram"] = ram
        if frame is not None:
            info["frame"] = frame  # real emulator RGB pixels, only when render_rgb=True
        info["player_y"] = int(ram[RAM_PLAYER_Y])
        info["ball_y"] = int(ram[RAM_BALL_Y])
        info["ball_x"] = int(ram[RAM_BALL_X])
        info["enemy_y"] = int(ram[RAM_ENEMY_Y])
        return info

    def close(self) -> None:
        self.env.close()
