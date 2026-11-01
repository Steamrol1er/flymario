"""Evaluation-only Pong scoring: load a trained checkpoint, freeze learning,
and play N points with no random exploration. Reports the score rate for the
agent's own policy, comparable against a uniform-random baseline.

Run: python -m experiments.eval_pong --arch dopamine --seeds 0 1 2 --episodes 1000
     python -m experiments.eval_pong --arch hybrid --seeds 0 1 2
     python -m experiments.eval_pong --arch random --seeds 0 1 2
Writes: analysis/pong_eval_{arch}.json
"""
import argparse
import json

import numpy as np
import torch

from connectome.loader import load_toy_connectome
from env.pong_env import PongEnv, N_ACTIONS
from experiments.train_pong import RamToKC, FrameStack, N_FRAME_STACK, MAX_STEPS
from learning.actor import ActorCritic, ActorCriticParams
from learning.dopamine import DopamineParams
from learning.qhead import HybridAgent


def build(arch, seed, n_kc):
    if arch == "dopamine":
        ac = ActorCritic(
            n_kc=n_kc, n_actions=N_ACTIONS, compartments={"approach": +1, "avoid": -1},
            dopamine_params=DopamineParams(eta=0.02, trace_decay=0.9, w_min=0.0, w_max=3.0),
            actor_params=ActorCriticParams(gamma=0.97, eta_actor=0.05, trace_decay=0.9, da_gain=1.5,
                                           w_min=-3.0, w_max=3.0, normalize_advantage=True))
        ac.load_state_dict(torch.load(f"analysis/trace09_seed{seed}_checkpoint.pt"))
        return ac
    agent = HybridAgent(n_kc=n_kc, n_actions=N_ACTIONS)
    agent.load_state_dict(torch.load(f"analysis/pong_hybrid_s{seed}_checkpoint.pt"))
    return agent


def act(arch, agent, kc, rng, sample):
    if arch == "random":
        return int(rng.integers(N_ACTIONS))
    if arch == "hybrid":
        return agent.act(kc, epsilon=0.0)[0]
    if sample:
        return agent.act(kc, epsilon=0.0)[0]
    return int(torch.argmax(kc @ agent.actor_w).item())


def evaluate(arch, seed, episodes, sample=False):
    torch.manual_seed(1000 + seed)
    rng = np.random.default_rng(1000 + seed)
    conn, groups = load_toy_connectome()
    n_kc = len(groups["kc"])
    agent = None if arch == "random" else build(arch, seed, n_kc)
    ram_to_kc = RamToKC(n_in=128 * N_FRAME_STACK, n_kc=n_kc, seed=seed)
    env = PongEnv()
    env.reset(seed=1000 + seed)
    scored = []
    for _ in range(episodes):
        obs, info = env.reset()
        stack = FrameStack()
        kc = ram_to_kc(stack.reset(obs))
        total = 0.0
        for _ in range(MAX_STEPS):
            a = act(arch, agent, kc, rng, sample)
            obs, info, r, done = env.step(a)
            kc = ram_to_kc(stack.push(obs))
            total += r
            if done or r != 0.0:
                break
        scored.append(total > 0)
    return float(np.mean(scored)) * 100


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arch", choices=["dopamine", "hybrid", "random"], required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--episodes", type=int, default=1000)
    a = ap.parse_args()
    out = {"arch": a.arch, "episodes": a.episodes, "rates": {}}
    for s in a.seeds:
        modes = {"argmax": evaluate(a.arch, s, a.episodes, False)}
        if a.arch == "dopamine":
            modes["sampled"] = evaluate(a.arch, s, a.episodes, True)
        out["rates"][s] = modes
        print(a.arch, "seed", s, modes, flush=True)
    json.dump(out, open(f"analysis/pong_eval_{a.arch}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
