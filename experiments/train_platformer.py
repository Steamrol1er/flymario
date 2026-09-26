"""Train the real dopamine-learning pipeline (learning/dopamine.py +
learning/actor.py, on connectome.loader's toy odor->KC->MBON circuit) to
play env/simple_platformer.py, and record full trajectories (game state +
brain internals, step by step) for a replay visualization.

This is the "M2 milestone" pipeline from docs/PLAN.md, but pointed at the
original platformer instead of gym-super-mario-bros (which can't run here --
see env/mario_env.py's ImportError). Everything else is unchanged: the same
FeatureExtractor, the same SIMPLE_MOVEMENT action set, the same
MushroomBodyLearner three-factor rule and TD-error actor-critic.

Uses the TOY connectome (connectome.loader.load_toy_connectome), not the
real 138,639-neuron FlyWire connectome: identifying real KC/MBON/DAN/DN
neurons in the real connectome needs FlyWire Codex cell-type annotations
this project doesn't have yet (see docs/STATUS.md). The learning RULE is
identical either way; only which neurons carry the KC code differs.

Run: python -m experiments.train_platformer
Writes: analysis/platformer_run.json
"""
from __future__ import annotations

import json

import numpy as np
import torch

from connectome.loader import load_toy_connectome
from env.simple_platformer import SimplePlatformer, Level, TILE_PX
from interface.sensory import FeatureExtractor, FeatureExtractorConfig
from interface.motor import SIMPLE_MOVEMENT
from learning.actor import ActorCritic, ActorCriticParams
from learning.dopamine import DopamineParams

N_EPISODES = 2500
MAX_STEPS = 500
REWARD_DX_WEIGHT = 0.15
REWARD_TIME_PENALTY = 0.005
REWARD_DEATH_PENALTY = 1.0
REWARD_FLAG_BONUS = 1.0


def shaped_reward(info: dict) -> float:
    r = REWARD_DX_WEIGHT * info["dx"] / TILE_PX - REWARD_TIME_PENALTY
    if info.get("dead"):
        r -= REWARD_DEATH_PENALTY
    if info.get("flag_get"):
        r += REWARD_FLAG_BONUS
    return float(np.clip(r, -1.0, 1.0))


class FeatsToKC:
    """Same fixed random projection placeholder used in experiments/train.py
    -- a stand-in for the real connectome's projection-neuron->KC wiring,
    documented there as a known simplification (docs/STATUS.md item 2)."""
    def __init__(self, n_in: int, n_kc: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.proj = torch.tensor(rng.standard_normal((n_in, n_kc)).astype(np.float32) * 0.3)

    def __call__(self, feats: np.ndarray, rates: np.ndarray) -> torch.Tensor:
        raw = torch.tensor(rates, dtype=torch.float32) @ self.proj
        return torch.relu(raw)


def run_episode(env, ac, feat_ex, feats_to_kc, epsilon, record=False):
    obs, info = env.reset()
    feats = feat_ex.extract(info)
    kc = feats_to_kc(feats, feat_ex.to_rates(feats))

    trace = []
    total_reward = 0.0
    for step in range(MAX_STEPS):
        action, probs = ac.act(kc, epsilon=epsilon)
        value = float(ac.critic.value(kc))
        _, info, done = env.step(action)
        feats = feat_ex.extract(info)
        kc_next = feats_to_kc(feats, feat_ex.to_rates(feats))
        reward = shaped_reward(info)
        delta = ac.learn(reward, kc_next, done=done)
        total_reward += reward

        if record:
            trace.append({
                "t": step,
                "x": round(info["x_pos"] / TILE_PX, 3),
                "y": round(info["y_pos"] / TILE_PX, 3),
                "enemy_x": round(info["enemies"][0][0] / TILE_PX, 3),
                "action": SIMPLE_MOVEMENT[action],
                "kc_active": int((kc > 0).sum()),
                "kc_mean": round(float(kc.mean()), 4),
                "value": round(value, 4),
                "dopamine_delta": round(float(delta), 4),
                "reward": round(reward, 4),
            })

        kc = kc_next
        if done:
            break

    return {
        "steps": step + 1,
        "total_reward": total_reward,
        "final_tile": info["tile_x"],
        "won": info["flag_get"],
        "dead": info["dead"],
        "trace": trace,
    }


def main():
    conn, groups = load_toy_connectome()
    n_kc = len(groups["kc"])
    compartments = {"approach": +1, "avoid": -1}
    ac = ActorCritic(
        n_kc=n_kc,
        n_actions=len(SIMPLE_MOVEMENT),
        compartments=compartments,
        dopamine_params=DopamineParams(eta=0.02, trace_decay=0.9, w_min=0.0, w_max=3.0),
        actor_params=ActorCriticParams(gamma=0.94, eta_actor=0.15, trace_decay=0.5, da_gain=3.0),
    )
    feat_ex = FeatureExtractor(FeatureExtractorConfig(sense_range_px=6 * TILE_PX))
    feats_to_kc = FeatsToKC(n_in=2 * 8, n_kc=n_kc)  # 8 features, split +/- => 16 inputs

    env = SimplePlatformer(max_steps=MAX_STEPS)

    history = []
    early_run = None
    late_run = None
    best_run = None

    for ep in range(N_EPISODES):
        epsilon = max(0.05, 0.35 * (1 - ep / (N_EPISODES * 0.7)))
        result = run_episode(env, ac, feat_ex, feats_to_kc, epsilon, record=True)
        history.append({
            "ep": ep, "final_tile": result["final_tile"],
            "total_reward": round(result["total_reward"], 3), "won": result["won"],
        })
        if ep == 0:
            early_run = result
        if ep == N_EPISODES - 1:
            late_run = result
        if best_run is None or result["final_tile"] > best_run["final_tile"]:
            best_run = result
            best_run["ep"] = ep
        if ep % 50 == 0:
            window = history[-50:]
            avg_tile = sum(h["final_tile"] for h in window) / len(window)
            win_rate = sum(h["won"] for h in window) / len(window)
            print(f"ep {ep:4d}  tile {result['final_tile']:2d}/{env.level.flag_tile}  "
                  f"avg(last50) {avg_tile:5.1f}  win_rate(last50) {win_rate:.0%}  eps={epsilon:.2f}")

    # The actor picks actions by sampling its softmax, not argmax, so even at
    # epsilon=0 rollouts are still stochastic. Run several and report the
    # best -- and the win rate across all of them, so we don't overstate a
    # lucky single rollout as "the policy".
    n_eval = 10
    eval_runs = [run_episode(env, ac, feat_ex, feats_to_kc, epsilon=0.0, record=True) for _ in range(n_eval)]
    eval_runs.sort(key=lambda r: r["final_tile"], reverse=True)
    final_greedy = eval_runs[0]
    greedy_win_rate = sum(r["won"] for r in eval_runs) / n_eval
    greedy_avg_tile = sum(r["final_tile"] for r in eval_runs) / n_eval
    print(f"greedy eval ({n_eval} rollouts): best tile {final_greedy['final_tile']}/{env.level.flag_tile}, "
          f"avg tile {greedy_avg_tile:.1f}, win rate {greedy_win_rate:.0%}")

    level = env.level
    out = {
        "level": {
            "length": level.length, "gaps": level.gaps, "obstacles": level.obstacles,
            "flag_tile": level.flag_tile, "enemy_start": level.enemy_start,
            "enemy_range": level.enemy_range,
        },
        "history": history,
        "early_episode": early_run,
        "late_episode": late_run,
        "best_episode": best_run,
        "final_greedy_episode": final_greedy,
        "greedy_win_rate": greedy_win_rate,
        "greedy_avg_tile": greedy_avg_tile,
        "n_kc": n_kc,
    }
    with open("analysis/platformer_run.json", "w") as f:
        json.dump(out, f)
    print("Wrote analysis/platformer_run.json")


if __name__ == "__main__":
    main()
