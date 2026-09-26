"""Thin wrapper around the real Atari Pong environment (`ale-py`), used
as-is: the real rendered game frame for display, the same 128-byte RAM
observation the network reads, and the same 6-action controller interface
the actual game exposes -- nothing re-derived or renamed. See
docs/PONG.md for why this game, how the ROM was obtained, what was
researched, and what was found along the way.

Native interface (unchanged from `gymnasium.make("ALE/Pong-v5", ...)`):
    action:  Discrete(6)  -- NOOP, FIRE, RIGHT, LEFT, RIGHTFIRE, LEFTFIRE
    reward:  +1 (you score), -1 (opponent scores), 0 otherwise
    ram:     uint8[128]   -- the raw Atari 2600 RAM, fed to the KC
                             projection directly, unmodified
    frame:   uint8[210,160,3] -- the actual emulator screen, RGB, exactly
                             what a person playing the real game would see
                             (this is what earlier viz pages got wrong by
                             hand-drawing an approximation instead)

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
RAM_BALL_Y = 49     # confirmed: reverses sign at wall bounces
RAM_BALL_X = 50     # confirmed: moves across a full serve, reverses partway
RAM_ENEMY_Y = 54    # inferred by elimination, not independently confirmed


class PongEnv:
    def __init__(self, frame_skip: int = 1, difficulty: int = 0):
        self.env = gym.make("ALE/Pong-v5", obs_type="rgb", frameskip=frame_skip, difficulty=difficulty)
        assert self.env.action_space.n == N_ACTIONS
        assert list(self.env.unwrapped.get_action_meanings()) == ACTION_MEANINGS

    def _ram(self):
        return self.env.unwrapped.ale.getRAM()

    def reset(self):
        frame, info = self.env.reset()
        ram = self._ram()
        return ram, self._augment_info(info, ram, frame)

    def step(self, action_index: int):
        frame, reward, terminated, truncated, info = self.env.step(action_index)
        done = terminated or truncated
        ram = self._ram()
        info = self._augment_info(info, ram, frame)
        return ram, info, float(reward), done

    def _augment_info(self, info: dict, ram, frame) -> dict:
        info = dict(info)
        info["ram"] = ram
        info["frame"] = frame  # real emulator RGB pixels, for display only
        info["player_y"] = int(ram[RAM_PLAYER_Y])
        info["ball_y"] = int(ram[RAM_BALL_Y])
        info["ball_x"] = int(ram[RAM_BALL_X])
        info["enemy_y"] = int(ram[RAM_ENEMY_Y])
        return info

    def close(self) -> None:
        self.env.close()
