"""Hybrid architecture: the same connectome-derived KC encoding as
experiments/train_pong.py, but a trained (backprop + replay) Q-head
instead of the dopamine three-factor rule for the actual decision-making.
See learning/qhead.py's docstring for exactly which part is "the fly" and
which part isn't, and docs/PONG.md's research section for why this
specific split (real encoding, trained readout) is the one pattern that
reliably works in the fly-connectome-plays-games community, while the
"both stages biological" version (this project's own dopamine rule,
DOOMFLY's real 166,700-neuron connectome) reliably doesn't.

Same native Pong interface as experiments/train_pong.py: raw RAM,
Discrete(6) actions, the game's own +1/-1/0 reward. Same frame stacking
(2 consecutive RAM frames, for velocity). Same --seed/--out/--target-steps
CLI so results are directly comparable.

Run: python -m experiments.train_pong_hybrid [--target-steps N] [--seed S] [--out PATH]
Writes: analysis/pong_hybrid_run.json, analysis/pong_hybrid_checkpoint.pt
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from connectome.loader import load_toy_connectome
from env.pong_env import PongEnv, N_ACTIONS, ACTION_MEANINGS
from experiments.train_pong import RamToKC, FrameStack, N_FRAME_STACK
from learning.qhead import HybridAgent

DEFAULT_TARGET_STEPS = 2_000_000
QUICK_TARGET_STEPS = 100_000
MAX_STEPS = 1200


def run_episode(env, agent, ram_to_kc, epsilon, record=False):
    obs, info = env.reset()
    stack = FrameStack()
    kc = ram_to_kc(stack.reset(obs))

    trace = []
    total_reward = 0.0
    for step in range(MAX_STEPS):
        action, _ = agent.act(kc, epsilon=epsilon)
        obs, info, reward, done = env.step(action)
        kc_next = ram_to_kc(stack.push(obs))
        td_error = agent.learn(reward, kc_next, done=done)
        total_reward += reward

        if record:
            trace.append({
                "t": step,
                "player_y": info["player_y"], "ball_x": info["ball_x"],
                "ball_y": info["ball_y"], "enemy_y": info["enemy_y"],
                "action": ACTION_MEANINGS[action],
                "td_error": round(td_error, 4),
                "reward": reward,
            })

        kc = kc_next
        if done or reward != 0.0:
            break

    return {
        "steps": step + 1,
        "total_reward": total_reward,
        "scored": total_reward > 0,
        "trace": trace,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target-steps", type=int, default=DEFAULT_TARGET_STEPS)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="analysis/pong_hybrid_run.json")
    args = ap.parse_args()
    target_steps = QUICK_TARGET_STEPS if args.quick else args.target_steps
    checkpoint_path = args.out.replace(".json", "_checkpoint.pt")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    conn, groups = load_toy_connectome()
    n_kc = len(groups["kc"])

    agent = HybridAgent(n_kc=n_kc, n_actions=N_ACTIONS, gamma=0.97, lr=1e-3,
                         buffer_capacity=20_000, batch_size=64,
                         target_sync_every=500, min_buffer_before_train=500)
    ram_to_kc = RamToKC(n_in=128 * N_FRAME_STACK, n_kc=n_kc, seed=args.seed)
    env = PongEnv()
    env.reset(seed=args.seed)

    print(f"Target: {target_steps:,} environment steps, seed={args.seed} [HYBRID: trained Q-head]")
    t_start = time.time()

    history = []
    early_run, late_run, best_run = None, None, None
    total_steps = 0
    ep = 0
    last_log_t = t_start

    while total_steps < target_steps:
        epsilon = max(0.05, 1.0 * (1 - total_steps / (target_steps * 0.5)))  # standard DQN-style anneal
        result = run_episode(env, agent, ram_to_kc, epsilon, record=True)
        total_steps += result["steps"]
        history.append({"ep": ep, "total_reward": result["total_reward"],
                         "scored": result["scored"], "steps": result["steps"],
                         "total_steps": total_steps})
        if ep == 0:
            early_run = result
        late_run = result
        if best_run is None or result["total_reward"] > best_run["total_reward"]:
            best_run = result
            best_run["ep"] = ep

        if time.time() - last_log_t > 5.0:
            last_log_t = time.time()
            window = history[-200:]
            score_rate = sum(h["scored"] for h in window) / len(window)
            avg_steps = sum(h["steps"] for h in window) / len(window)
            elapsed = time.time() - t_start
            pct = total_steps / target_steps * 100
            eta_s = elapsed / max(total_steps, 1) * (target_steps - total_steps)
            print(f"step {total_steps:>10,}/{target_steps:,} ({pct:5.1f}%)  ep {ep:>6}  "
                  f"score_rate(last200)={score_rate:5.1%}  avg_steps={avg_steps:6.1f}  "
                  f"eps={epsilon:.2f}  elapsed={elapsed/60:.1f}m  eta={eta_s/60:.1f}m")

        ep += 1

    env.close()
    elapsed = time.time() - t_start
    print(f"Done: {total_steps:,} steps over {ep} episodes in {elapsed/60:.1f} minutes "
          f"({total_steps/elapsed:.0f} steps/sec)")

    out = {
        "history": history, "early_episode": early_run, "late_episode": late_run,
        "best_episode": best_run, "n_kc": n_kc, "target_steps": target_steps,
        "total_steps": total_steps, "wall_clock_seconds": elapsed, "seed": args.seed,
        "architecture": "hybrid_qhead",
    }
    with open(args.out, "w") as f:
        json.dump(out, f)
    print(f"Wrote {args.out}")
    torch.save(agent.state_dict(), checkpoint_path)
    print(f"Wrote {checkpoint_path}")


if __name__ == "__main__":
    main()
