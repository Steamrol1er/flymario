"""Game state -> sensory input for the fly brain (Stage A: feature-based).

Stage A (implemented here): hand-crafted features from Mario's RAM state,
converted to firing rates and injected as external current into a set of
"projection neuron" input channels feeding the mushroom body's Kenyon cells.
This is the recommended starting point (see docs/PLAN.md section 5): it lets
us use the real, connectome-derived KC coding (each KC randomly samples ~6
input channels, giving a sparse pseudo-random code) without first solving
the much harder problem of mapping pixels onto the compound eye.

Stage B (screen -> photoreceptors -> optic lobe) is a separate, harder
module — not implemented yet, see docs/STATUS.md.
"""
from __future__ import annotations

import dataclasses

import numpy as np


N_FEATURES = 8
FEATURE_NAMES = [
    "enemy_dx",       # signed horizontal distance to nearest enemy, clipped
    "enemy_dy",       # signed vertical distance to nearest enemy
    "enemy_present",  # 1.0 if an enemy is within sensing range, else 0.0
    "gap_ahead",      # 1.0 if there's a pit within a few tiles ahead
    "obstacle_ahead", # 1.0 if there's a solid obstacle (pipe/block) ahead
    "airborne",       # 1.0 if Mario is currently in the air
    "mario_vx",       # Mario's horizontal velocity, signed
    "mario_vy",       # Mario's vertical velocity, signed
]


@dataclasses.dataclass
class FeatureExtractorConfig:
    sense_range_px: float = 80.0   # how far ahead we look for enemies/obstacles
    max_rate_hz: float = 200.0     # firing rate for a "fully on" binary feature


class FeatureExtractor:
    """Turns `info` dict from the Mario gym env (or a hand-rolled state dict
    with the same keys) into an [N_FEATURES] float vector in [-1, 1], then
    into per-channel Poisson-like firing rates for injection into the brain
    model.

    Expected `info` keys (subset of what gym-super-mario-bros exposes via
    RAM, plus a couple we compute ourselves in env/mario_env.py):
        x_pos, y_pos, x_vel, y_vel : ints/floats, world pixel coords
        enemies : list of (x_pos, y_pos) for nearby enemies
        gap_ahead : bool
        obstacle_ahead : bool
        airborne : bool
    """

    def __init__(self, config: FeatureExtractorConfig | None = None):
        self.config = config or FeatureExtractorConfig()

    def extract(self, info: dict) -> np.ndarray:
        cfg = self.config
        feats = np.zeros(N_FEATURES, dtype=np.float32)

        enemies = info.get("enemies", [])
        if enemies:
            mx, my = info["x_pos"], info["y_pos"]
            dxs = [(ex - mx, ey - my) for ex, ey in enemies]
            dx, dy = min(dxs, key=lambda p: abs(p[0]))
            feats[0] = np.clip(dx / cfg.sense_range_px, -1.0, 1.0)
            feats[1] = np.clip(dy / cfg.sense_range_px, -1.0, 1.0)
            feats[2] = 1.0 if abs(dx) < cfg.sense_range_px else 0.0
        # else: leave enemy features at 0

        feats[3] = 1.0 if info.get("gap_ahead", False) else 0.0
        feats[4] = 1.0 if info.get("obstacle_ahead", False) else 0.0
        feats[5] = 1.0 if info.get("airborne", False) else 0.0
        feats[6] = np.clip(info.get("x_vel", 0.0) / 10.0, -1.0, 1.0)
        feats[7] = np.clip(info.get("y_vel", 0.0) / 10.0, -1.0, 1.0)
        return feats

    def to_rates(self, feats: np.ndarray) -> np.ndarray:
        """Map each feature in [-1, 1] to a non-negative firing rate (Hz),
        splitting sign into separate "positive" and "negative" sub-channels
        so downstream KCs see rate-coded, not signed, input (real neurons
        don't have negative firing rates)."""
        pos = np.clip(feats, 0, 1) * self.config.max_rate_hz
        neg = np.clip(-feats, 0, 1) * self.config.max_rate_hz
        return np.concatenate([pos, neg])  # [2 * N_FEATURES]


def poisson_spike_prob(rate_hz: np.ndarray, dt_ms: float) -> np.ndarray:
    """Per-timestep spike probability for a Poisson process at `rate_hz`,
    for use when driving individual simulator timesteps (dt in ms)."""
    return 1.0 - np.exp(-rate_hz * dt_ms / 1000.0)
