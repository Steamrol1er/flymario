"""Training entry point tying together: connectome subcircuit -> LIF
simulator -> sensory feature injection -> mushroom-body actor-critic ->
motor decoding -> Mario env -> reward -> dopamine.

This is the M2+ milestone script (docs/PLAN.md). It needs:
  - a real or toy connectome (connectome.loader)
  - resolved cell-type groups (connectome.cell_types) to know which neurons
    are KCs/MBONs/DANs/DNs -- for the real connectome this requires FlyWire
    Codex annotation files that can't be auto-downloaded (see
    connectome/cell_types.py and README.md). Until those are in place, use
    `connectome.source: toy` in the config, which has hand-built groups and
    runs end-to-end today.
  - gym-super-mario-bros + nes-py installed (see README.md install notes)

Usage:
    python -m experiments.train --config experiments/m2_config.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml

from connectome.loader import load_shiu_connectome, load_toy_connectome
from connectome.cell_types import resolve_groups
from connectome.subcircuit import build_subcircuit
from env.mario_env import MarioEnv, RewardConfig
from interface.sensory import FeatureExtractor, FeatureExtractorConfig
from interface.motor import SpikeToAction
from learning.actor import ActorCritic, ActorCriticParams
from learning.dopamine import DopamineParams


def build_connectome_and_groups(cfg: dict):
    c = cfg["connectome"]
    if c["source"] == "toy":
        conn, groups = load_toy_connectome()
        # toy groups double as our "KC"/"MBON" stand-ins directly; no DNs in
        # the toy circuit, so motor decoding falls back to a direct
        # actor-output -> action mapping (see TrainLoop.act below).
        return conn, groups
    elif c["source"] == "shiu":
        conn = load_shiu_connectome(c["data_dir"])
        groups = resolve_groups(conn.root_id_to_index, c["data_dir"])  # may raise
        sub = build_subcircuit(conn, groups, hops=c.get("subcircuit_hops", 2))
        return sub, groups
    else:
        raise ValueError(f"unknown connectome.source {c['source']!r}")


class TrainLoop:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.conn, self.groups = build_connectome_and_groups(cfg)

        n_kc = len(self.groups.get("kenyon_cell", [])) or self.conn.n_neurons // 4
        compartments = {"approach": +1, "avoid": -1}  # PAM-like / PPL1-like
        self.ac = ActorCritic(
            n_kc=n_kc,
            n_actions=7,  # len(interface.motor.SIMPLE_MOVEMENT)
            compartments=compartments,
            dopamine_params=DopamineParams(**cfg["learning"]["dopamine"]),
            actor_params=ActorCriticParams(**cfg["learning"]["actor"]),
        )
        self.n_kc = n_kc

        self.feat_extractor = FeatureExtractor(FeatureExtractorConfig(**cfg["sensory"]))
        self.motor = SpikeToAction()
        self.env = MarioEnv(
            world_stage=cfg["env"]["world_stage"],
            frame_skip=cfg["env"]["frame_skip"],
            reward_config=RewardConfig(**cfg["env"]["reward"]),
        )

        self.epsilon = cfg["decision"]["epsilon_start"]

    def feats_to_kc(self, feats: np.ndarray) -> torch.Tensor:
        """Placeholder feature -> KC encoding until the real connectome's
        projection-neuron -> KC wiring is hooked up (needs cell_types groups
        for `projection_neuron`, not yet in GROUP_PATTERNS). For now: project
        the small feature vector onto a fixed random sparse KC pattern, which
        at least reproduces the *qualitative* property we actually want here
        (a sparse, roughly-separable code per distinct situation) while we
        wait on the real wiring. See docs/STATUS.md."""
        rates = self.feat_extractor.to_rates(feats)  # [2*N_FEATURES]
        if not hasattr(self, "_proj"):
            rng = np.random.default_rng(0)
            self._proj = torch.tensor(
                rng.standard_normal((len(rates), self.n_kc)).astype(np.float32) * 0.1
            )
        raw = torch.tensor(rates, dtype=torch.float32) @ self._proj
        return torch.relu(raw)

    def run_episode(self, max_steps: int) -> dict:
        obs, info = self.env.reset()
        feats = self.feat_extractor.extract(info)
        kc = self.feats_to_kc(feats)

        total_reward = 0.0
        for step in range(max_steps):
            action, _ = self.ac.act(kc, epsilon=self.epsilon)
            obs, info, reward, done = self.env.step(action)
            feats = self.feat_extractor.extract(info)
            kc_next = self.feats_to_kc(feats)

            self.ac.learn(reward, kc_next, done=done)
            kc = kc_next
            total_reward += reward
            if done:
                break
        return {"steps": step + 1, "total_reward": total_reward, "x_pos": info.get("x_pos", 0)}

    def run(self):
        cfg = self.cfg["training"]
        out_dir = Path(cfg["out_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        eps_decay_n = self.cfg["decision"]["epsilon_decay_episodes"]
        eps0, eps1 = self.cfg["decision"]["epsilon_start"], self.cfg["decision"]["epsilon_end"]

        history = []
        for ep in range(cfg["n_episodes"]):
            self.epsilon = eps1 + (eps0 - eps1) * max(0.0, 1 - ep / eps_decay_n)
            stats = self.run_episode(cfg["max_steps_per_episode"])
            history.append(stats)
            if ep % cfg["log_every"] == 0:
                print(f"ep {ep:5d}  reward {stats['total_reward']:+.2f}  "
                      f"x_pos {stats['x_pos']:5d}  steps {stats['steps']:4d}  eps {self.epsilon:.3f}")
        self.env.close()
        return history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    loop = TrainLoop(cfg)
    loop.run()


if __name__ == "__main__":
    main()
