"""Super Mario Bros gym wrapper: frame-skip, reward shaping, and extraction
of the extra `info` fields our sensory feature extractor needs (see
interface/sensory.py).

Requires `gym-super-mario-bros` + `nes-py`, which need a C/C++ toolchain to
build `nes-py`'s C extension. This raises a clear, actionable ImportError if
those aren't installed rather than failing somewhere confusing downstream —
see README.md "Mario environment install notes" for how to get them
installed (WSL2 is the path of least resistance on Windows).
"""
from __future__ import annotations

import dataclasses

import numpy as np

try:
    import gym_super_mario_bros
    from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
    from nes_py.wrappers import JoypadSpace
    _IMPORT_ERROR: Exception | None = None
except Exception as e:  # pragma: no cover - exercised only when deps missing
    gym_super_mario_bros = None
    SIMPLE_MOVEMENT = None
    JoypadSpace = None
    _IMPORT_ERROR = e


@dataclasses.dataclass
class RewardConfig:
    dx_weight: float = 0.1
    time_penalty: float = 0.01
    death_penalty: float = 15.0
    flag_bonus: float = 50.0
    clip: float = 1.0


class MarioEnv:
    """Thin wrapper: `step(action_index)` -> (screen, info, reward, done).
    `info` includes the raw gym-super-mario-bros info dict plus derived
    fields (`enemies`, `gap_ahead`, `obstacle_ahead`, `airborne`) that
    `interface.sensory.FeatureExtractor` expects.
    """

    def __init__(
        self,
        world_stage: str = "SuperMarioBros-1-1-v0",
        frame_skip: int = 4,
        reward_config: RewardConfig | None = None,
    ):
        if _IMPORT_ERROR is not None:
            raise ImportError(
                "gym-super-mario-bros / nes-py are not installed or failed to "
                f"import ({_IMPORT_ERROR!r}). See README.md 'Mario environment "
                "install notes' - on Windows this usually means you need the "
                "Visual Studio C++ Build Tools, or to run under WSL2. The rest "
                "of flymario (connectome, sim, learning) does not depend on "
                "this package and can be developed/tested independently."
            )
        self.env = JoypadSpace(gym_super_mario_bros.make(world_stage), SIMPLE_MOVEMENT)
        self.frame_skip = frame_skip
        self.reward_config = reward_config or RewardConfig()
        self._prev_x = 0

    def reset(self) -> tuple[np.ndarray, dict]:
        obs = self.env.reset()
        self._prev_x = 0
        info = self._augment_info({}, obs)
        return obs, info

    def step(self, action_index: int) -> tuple[np.ndarray, dict, float, bool]:
        total_reward_raw = 0.0
        done = False
        info: dict = {}
        obs = None
        for _ in range(self.frame_skip):
            obs, _, done, info = self.env.step(action_index)
            if done:
                break

        reward = self._shape_reward(info, done)
        info = self._augment_info(info, obs)
        return obs, info, reward, done

    def _shape_reward(self, info: dict, done: bool) -> float:
        cfg = self.reward_config
        x = info.get("x_pos", self._prev_x)
        dx = x - self._prev_x
        self._prev_x = x

        r = cfg.dx_weight * dx - cfg.time_penalty
        if done and not info.get("flag_get", False):
            r -= cfg.death_penalty
        if info.get("flag_get", False):
            r += cfg.flag_bonus
        return float(np.clip(r, -cfg.clip, cfg.clip))

    def _augment_info(self, info: dict, obs) -> dict:
        # Placeholder heuristics from RAM-derived fields that
        # gym-super-mario-bros already exposes (x_pos, y_pos, status, etc.).
        # Enemy positions specifically require reading additional NES RAM
        # addresses (sprite tables) not exposed by the default `info` dict;
        # see docs/STATUS.md for the follow-up needed to fill this in
        # properly (reading env.unwrapped.ram directly).
        info = dict(info)
        info.setdefault("enemies", [])
        info.setdefault("gap_ahead", False)
        info.setdefault("obstacle_ahead", False)
        info.setdefault("airborne", info.get("y_pos", 0) > 79)  # rough ground-level heuristic
        info.setdefault("x_vel", 0.0)
        info.setdefault("y_vel", 0.0)
        return info

    def close(self) -> None:
        self.env.close()
