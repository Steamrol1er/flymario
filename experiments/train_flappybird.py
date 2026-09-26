"""Train the real dopamine-learning pipeline (learning/dopamine.py +
learning/actor.py, on connectome.loader's toy odor->KC->MBON circuit) to
play FlappyBird (env/flappy_env.py), and record full trajectories for a
replay visualization.

Same pipeline as experiments/train_platformer.py, on a simpler task: 2
actions (flap or don't) instead of 7, and the environment's own 12-value
observation feeds the KC projection directly (no hand-crafted feature
extractor needed -- see env/flappy_env.py's docstring).

Run: python -m experiments.train_flappybird
Writes: analysis/flappybird_run.json
"""
from __future__ import annotations

import json

import numpy as np
import torch

from connectome.loader import load_toy_connectome
from env.flappy_env import FlappyEnv, N_ACTIONS, ACTION_NAMES
from learning.actor import ActorCritic, ActorCriticParams
from learning.dopamine import DopamineParams

N_EPISODES = 2000
MAX_STEPS = 2000  # ~35s of gameplay at 60fps if it survives that long


def shaped_reward(raw_reward: float, info: dict) -> float:
    """The environment's own reward (+0.1/frame alive regardless of action,
    +1/pipe, -1/death) turned out to give almost no usable learning signal:
    a flat per-step bonus for EITHER action doesn't teach WHEN to flap, and
    the sparse +1/-1 events are too rare, early in training, for a 2-action
    linear readout to find the right timing by trial and error alone (pure
    50/50 random flapping passes zero pipes in 100 tries -- this needs real
    positional control, not luck). Unlike env/simple_platformer.py's single
    discrete "jump at the gap" decision, this is a continuous tracking
    problem, so it needs continuous, not just event, feedback -- much like
    Mario's reward includes continuous forward progress (dx), not just
    terminal win/death. Reward here = negative distance from the gap
    center (bigger the further off-target you are, every frame) plus the
    real pipe-pass/death events.
    """
    if info.get("dead"):
        return -1.0
    top, bot = info["next_pipe"][1], info["next_pipe"][2]
    gap_center = (top + bot) / 2
    dist = abs(info["player_y"] - gap_center)
    # Small coefficient on the distance term keeps typical per-step rewards
    # close to 0 (roughly -0.1..0) instead of close to -1: with the actor's
    # linear update (eta_actor * delta * eligibility) applied every single
    # step, an always-near-(-1) reward produced deltas large enough to
    # saturate actor_w to its clamp bounds within the first ~50 episodes,
    # collapsing all state-dependent differentiation between actions (see
    # docs/STATUS.md for the diagnostic that caught this).
    r = -0.2 * dist - 0.005
    if raw_reward >= 1.0:  # passed a pipe
        r += 1.0
    return float(np.clip(r, -1.0, 1.0))


class ObsToKC:
    """Fixed random projection from the 12-value observation to KC space --
    same role as experiments/train_platformer.py's FeatsToKC, standing in
    for the real connectome's (not-yet-identified) projection-neuron->KC
    wiring.

    Includes divisive normalization on the output: without it, the total
    activation summed over ~100 simultaneously-active KCs (each an
    unbounded ReLU output) swings the mushroom body's linear MBON readout
    (learning/dopamine.py's `value()`, a plain weighted sum over KCs) into
    the tens or hundreds. Any tiny weight asymmetry between the
    approach/avoid compartments then produces a huge, unstable TD error,
    which saturates the actor's clamped weights within the first handful of
    episodes and destroys all state-dependent differentiation between
    actions (confirmed by direct inspection: actor_w collapsed to its
    [-1, 0] clamp bounds inside the very first 50-episode batch before this
    fix -- see docs/STATUS.md). Real KC->MBON circuits don't have this
    problem because Kenyon cells are sparse and roughly binary; dividing by
    the total activation here keeps the "activity budget" bounded
    regardless of how many KCs a given random projection happens to
    activate, the same role divisive normalization plays in real neural
    circuits.
    """
    def __init__(self, n_in: int, n_kc: int, seed: int = 0, target_total: float = 4.0):
        rng = np.random.default_rng(seed)
        self.proj = torch.tensor(rng.standard_normal((n_in, n_kc)).astype(np.float32) * 0.5)
        self.target_total = target_total

    def __call__(self, obs: np.ndarray) -> torch.Tensor:
        raw = torch.tensor(obs, dtype=torch.float32) @ self.proj
        kc = torch.relu(raw)
        return kc / (kc.sum() + 1e-6) * self.target_total


def run_episode(env, ac, obs_to_kc, epsilon, record=False):
    obs, info = env.reset()
    kc = obs_to_kc(obs)

    trace = []
    total_reward = 0.0
    score = 0
    for step in range(MAX_STEPS):
        action, probs = ac.act(kc, epsilon=epsilon)
        value = float(ac.critic.value(kc))
        obs, info, raw_reward, done = env.step(action)
        reward = shaped_reward(raw_reward, info)
        kc_next = obs_to_kc(obs)
        delta = ac.learn(reward, kc_next, done=done)
        total_reward += reward
        score = info.get("score", score)

        if record:
            trace.append({
                "t": step,
                "player_y": round(info["player_y"], 4),
                "player_rot": round(info["player_rot"], 4),
                "pipe": [round(v, 4) for v in info["next_pipe"]],
                "action": ACTION_NAMES[action],
                "kc_active": int((kc > 0).sum()),
                "value": round(value, 4),
                "dopamine_delta": round(float(delta), 4),
                "reward": round(reward, 4),
            })

        kc = kc_next
        if done:
            break

    return {
        "steps": step + 1,
        "total_reward": round(total_reward, 3),
        "score": score,
        "dead": bool(info.get("dead", True)),
        "trace": trace,
    }


def run_heuristic_episode(env, max_steps=MAX_STEPS):
    """A simple hand-coded reference policy (not the fly/RL agent): flap
    when below the gap center and not already moving upward fast. Used
    purely to show the environment is genuinely solvable (avg score 0.9,
    max 3 over 20 trials -- see docs/STATUS.md) alongside the agent's own
    result, since the trained agent did not solve even one pipe crossing
    in 2000 episodes."""
    obs, info = env.reset()
    trace = []
    for step in range(max_steps):
        top, bot = info["next_pipe"][1], info["next_pipe"][2]
        gap_center = (top + bot) / 2
        action = 1 if (info["player_y"] > gap_center and info["player_vel_y"] > -0.3) else 0
        obs, info, reward, done = env.step(action)
        trace.append({
            "t": step, "player_y": round(info["player_y"], 4), "player_rot": round(info["player_rot"], 4),
            "pipe": [round(v, 4) for v in info["next_pipe"]], "action": ACTION_NAMES[action],
            "kc_active": 0, "value": 0.0, "dopamine_delta": 0.0, "reward": round(reward, 4),
        })
        if done:
            break
    return {"steps": step + 1, "score": info["score"], "dead": bool(info.get("dead", True)), "trace": trace}


def main():
    conn, groups = load_toy_connectome()
    n_kc = len(groups["kc"])
    compartments = {"approach": +1, "avoid": -1}
    ac = ActorCritic(
        n_kc=n_kc,
        n_actions=N_ACTIONS,
        compartments=compartments,
        dopamine_params=DopamineParams(eta=0.02, trace_decay=0.9, w_min=0.0, w_max=3.0),
        actor_params=ActorCriticParams(gamma=0.95, eta_actor=0.05, trace_decay=0.6, da_gain=1.5),
    )
    obs_to_kc = ObsToKC(n_in=12, n_kc=n_kc)
    env = FlappyEnv()

    history = []
    early_run, late_run, best_run = None, None, None

    for ep in range(N_EPISODES):
        epsilon = max(0.1, 0.3 * (1 - ep / (N_EPISODES * 0.6)))
        result = run_episode(env, ac, obs_to_kc, epsilon, record=True)
        history.append({"ep": ep, "score": result["score"], "steps": result["steps"],
                         "total_reward": result["total_reward"]})
        if ep == 0:
            early_run = result
        if ep == N_EPISODES - 1:
            late_run = result
        if best_run is None or result["score"] > best_run["score"] or \
           (result["score"] == best_run["score"] and result["steps"] > best_run["steps"]):
            best_run = result
            best_run["ep"] = ep
        if ep % 50 == 0:
            window = history[-50:]
            avg_score = sum(h["score"] for h in window) / len(window)
            avg_steps = sum(h["steps"] for h in window) / len(window)
            print(f"ep {ep:4d}  score {result['score']:2d}  avg(last50) score {avg_score:4.2f}  "
                  f"steps {avg_steps:6.1f}  eps={epsilon:.2f}")

    n_eval = 10
    eval_runs = [run_episode(env, ac, obs_to_kc, epsilon=0.0, record=True) for _ in range(n_eval)]
    eval_runs.sort(key=lambda r: (r["score"], r["steps"]), reverse=True)
    final_greedy = eval_runs[0]
    greedy_avg_score = sum(r["score"] for r in eval_runs) / n_eval
    print(f"greedy eval ({n_eval} rollouts): best score {final_greedy['score']}, "
          f"avg score {greedy_avg_score:.2f}")

    # hand-coded reference policy, best of a few tries, to show the
    # environment is genuinely solvable (see run_heuristic_episode's
    # docstring) alongside the agent's own honest result
    heuristic_runs = [run_heuristic_episode(env) for _ in range(15)]
    heuristic_runs.sort(key=lambda r: (r["score"], r["steps"]), reverse=True)
    heuristic_best = heuristic_runs[0]
    print(f"heuristic reference (best of 15): score {heuristic_best['score']}, steps {heuristic_best['steps']}")

    env.close()

    out = {
        "history": history,
        "early_episode": early_run,
        "late_episode": late_run,
        "best_episode": best_run,
        "final_greedy_episode": final_greedy,
        "heuristic_episode": heuristic_best,
        "greedy_avg_score": greedy_avg_score,
        "n_kc": n_kc,
    }
    with open("analysis/flappybird_run.json", "w") as f:
        json.dump(out, f)
    print("Wrote analysis/flappybird_run.json")


if __name__ == "__main__":
    main()
