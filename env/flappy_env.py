"""Thin wrapper around `flappy_bird_gymnasium` for a consistent
reset()->(obs, info) / step(action)->(obs, info, reward, done) interface,
matching env/mario_env.py and env/simple_platformer.py.

Unlike Mario or our own platformer, FlappyBird's own observation vector
*is* already a compact, hand-scaled feature vector -- not pixels -- so
there's no separate feature-extraction step needed (see
interface/sensory.py's docstring for why that's normally the first thing
built for a new environment). The 12 values are, in order (from
flappy_bird_gymnasium's own source, `_get_observation_features`):

    [0:3]  last pipe:      (horizontal dist, top-of-gap y, bottom-of-gap y)
    [3:6]  next pipe:      (horizontal dist, top-of-gap y, bottom-of-gap y)
    [6:9]  next-next pipe: (horizontal dist, top-of-gap y, bottom-of-gap y)
    [9]    player's vertical position
    [10]   player's vertical velocity
    [11]   player's rotation

All values are pre-normalized by the package to roughly [-1, 1]. The
environment's own reward (+0.1/frame alive, +1 per pipe passed, -1 on
death) is already sensibly shaped, so we use it unchanged rather than
re-deriving reward shaping like env/mario_env.py does.

Only 2 actions: 0 = do nothing (fall), 1 = flap (jump up).
"""
from __future__ import annotations

import gymnasium as gym
import flappy_bird_gymnasium  # noqa: F401 -- registers FlappyBird-v0

N_ACTIONS = 2
ACTION_NAMES = ["noop", "flap"]


class FlappyEnv:
    def __init__(self, use_lidar: bool = False):
        self.env = gym.make("FlappyBird-v0", use_lidar=use_lidar)

    def reset(self):
        obs, info = self.env.reset()
        return obs, self._augment_info(info, obs)

    def step(self, action_index: int):
        obs, reward, terminated, truncated, info = self.env.step(action_index)
        done = terminated or truncated
        info = self._augment_info(info, obs)
        info["dead"] = terminated
        return obs, info, float(reward), done

    def _augment_info(self, info: dict, obs) -> dict:
        info = dict(info)
        info["obs"] = obs
        info["player_y"] = float(obs[9])
        info["player_vel_y"] = float(obs[10])
        info["player_rot"] = float(obs[11])
        info["next_pipe"] = (float(obs[3]), float(obs[4]), float(obs[5]))
        return info

    def close(self) -> None:
        self.env.close()
