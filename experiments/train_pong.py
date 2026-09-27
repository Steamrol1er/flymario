"""Train the real dopamine-learning pipeline (learning/dopamine.py +
learning/actor.py, on connectome.loader's toy odor->KC->MBON circuit) to
play the real Atari Pong (env/pong_env.py), using Pong's own native
interface throughout: no hand-crafted feature extractor, no remapped
action set, no reshaped reward -- the KC projection reads the raw 128-byte
RAM directly, action indices are the game's own 6 actions, and the reward
is the game's own +1/-1/0. See docs/PONG.md for the full account: how the
ROM was obtained, what was researched beforehand, and what was found along
the way (including a real bug and its fix, applied here from the start
rather than discovered by trial and error a second time).

Run: python -m experiments.train_pong [--target-steps N] [--quick]
Writes: analysis/pong_run.json, analysis/pong_checkpoint.pt

## How many episodes is "standard"?

The original Atari DQN paper (Mnih et al. 2015, "Human-level control
through deep reinforcement learning", Nature) trains each game for
**50 million frames** -- their own words: "around 38 days of game
experience in total" (at 60fps with a 4-frame action repeat, 50M agent
decisions = 200M emulator frames = ~38.6 days of continuous real-time
play). That's measured in frames/steps, not episodes, because episode
length varies enormously with skill (an untrained agent's Pong point
lasts a couple hundred steps; a skilled one can rally for much longer)
-- so this script's default target is a STEP budget, and "episodes"
(one Pong point exchange each) fall out of that naturally.

`--target-steps 10_000_000` (this file's default) is about 20% of DQN's
50M-step standard -- chosen because it actually completes in this
environment (roughly 45-70 minutes at the ~0.2-0.3ms/step this pipeline
runs at on CPU) rather than the several days DQN's full budget would take
on the same hardware. It is NOT a claim that this matches DQN's result at
20% of the data -- DQN also uses a convolutional Q-network trained by
gradient descent over a replay buffer, a fundamentally more sample-
efficient setup than this project's linear KC readout and online
eligibility-trace update. Pass `--quick` for a ~400-episode sanity-check
run instead (roughly what earlier runs in docs/PONG.md used).
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from connectome.loader import load_toy_connectome
from env.pong_env import PongEnv, N_ACTIONS, ACTION_MEANINGS
from learning.actor import ActorCritic, ActorCriticParams
from learning.dopamine import DopamineParams

DEFAULT_TARGET_STEPS = 10_000_000   # ~20% of DQN's 50M-frame standard, see module docstring
QUICK_TARGET_STEPS = 100_000        # ~400 episodes at this task's typical episode length
MAX_STEPS = 1200                    # a single Pong point rarely takes longer than this


N_FRAME_STACK = 2  # see FrameStack docstring for why this exists


class FrameStack:
    """Concatenates the last N_FRAME_STACK raw RAM frames before they reach
    RamToKC.

    A single RAM frame gives the network positions only -- player_y,
    ball_x, ball_y, enemy_y -- with no way to tell which direction the
    ball is moving. That's a genuine information deficiency, not a tuning
    problem: standard Atari RL (the original DQN included) always stacks
    multiple consecutive frames for exactly this reason, since velocity
    matters as much as position for a real-time tracking task. This does
    the same thing at the RAM level instead of the pixel level, and
    changes nothing about the "native interface" principle in this file's
    module docstring -- it's still literally the game's own raw RAM,
    unmodified, just several consecutive readings of it instead of one,
    the same way DQN's frame stack is still literally the game's own
    pixels. It is NOT the same thing as hand-picking which of the 128
    bytes matter (see docs/STATUS.md for that trade-off, deliberately not
    taken here) -- the network still sees all 128 bytes, at each of the
    stacked timesteps, and has to find the useful ones itself.
    """
    def __init__(self, n_frames: int = N_FRAME_STACK):
        self.n_frames = n_frames
        self.frames: list[np.ndarray] = []

    def reset(self, first_ram: np.ndarray) -> np.ndarray:
        self.frames = [first_ram.copy() for _ in range(self.n_frames)]
        return np.concatenate(self.frames)

    def push(self, ram: np.ndarray) -> np.ndarray:
        self.frames.append(ram.copy())
        self.frames.pop(0)
        return np.concatenate(self.frames)


class RamToKC:
    """Fixed random projection from the (possibly frame-stacked) raw RAM,
    normalized to [0,1], to KC space, WITH divisive normalization on the
    output -- the fix from the FlappyBird run (docs/STATUS.md): without
    it, unbounded KC activation magnitude destabilizes the mushroom
    body's linear value estimate and saturates the actor's weights within
    the first few dozen episodes, before it can learn anything
    state-dependent.
    """
    def __init__(self, n_in: int, n_kc: int, seed: int = 0, target_total: float = 4.0):
        rng = np.random.default_rng(seed)
        self.proj = torch.tensor(rng.standard_normal((n_in, n_kc)).astype(np.float32) * 0.5)
        self.target_total = target_total

    def __call__(self, ram: np.ndarray) -> torch.Tensor:
        x = torch.tensor(ram, dtype=torch.float32) / 255.0
        raw = x @ self.proj
        kc = torch.relu(raw)
        return kc / (kc.sum() + 1e-6) * self.target_total


def run_episode(env, ac, ram_to_kc, epsilon, record=False):
    obs, info = env.reset()
    stack = FrameStack()
    kc = ram_to_kc(stack.reset(obs))

    trace = []
    total_reward = 0.0
    for step in range(MAX_STEPS):
        action, probs = ac.act(kc, epsilon=epsilon)
        value = float(ac.critic.value(kc))
        obs, info, reward, done = env.step(action)
        kc_next = ram_to_kc(stack.push(obs))
        delta = ac.learn(reward, kc_next, done=done)
        total_reward += reward

        if record:
            trace.append({
                "t": step,
                "player_y": info["player_y"], "ball_x": info["ball_x"],
                "ball_y": info["ball_y"], "enemy_y": info["enemy_y"],
                "action": ACTION_MEANINGS[action],
                "kc_active": int((kc > 0).sum()),
                "value": round(value, 4),
                "dopamine_delta": round(float(delta), 4),
                "reward": reward,
            })

        kc = kc_next
        if done or reward != 0.0:
            # Pong signals a scored point with a nonzero reward but doesn't
            # end the underlying game there -- we treat each point as one
            # training "episode" (bootstrapping the eligibility traces
            # fresh each point) since that's the natural unit of credit
            # assignment here, closer to a single game episode than playing
            # out a full 21-point match every time.
            break

    return {
        "steps": step + 1,
        "total_reward": total_reward,
        "scored": total_reward > 0,
        "trace": trace,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target-steps", type=int, default=DEFAULT_TARGET_STEPS,
                     help=f"total environment steps to train for (default {DEFAULT_TARGET_STEPS:,}, "
                          f"~20%% of DQN's 50M-frame Atari standard -- see module docstring)")
    ap.add_argument("--quick", action="store_true",
                     help=f"shortcut for --target-steps {QUICK_TARGET_STEPS:,} (a fast sanity check)")
    ap.add_argument("--checkpoint-every", type=int, default=1_000_000,
                     help="also save an intermediate checkpoint every this many steps")
    args = ap.parse_args()
    target_steps = QUICK_TARGET_STEPS if args.quick else args.target_steps

    conn, groups = load_toy_connectome()
    n_kc = len(groups["kc"])
    compartments = {"approach": +1, "avoid": -1}
    ac = ActorCritic(
        n_kc=n_kc,
        n_actions=N_ACTIONS,
        compartments=compartments,
        dopamine_params=DopamineParams(eta=0.02, trace_decay=0.9, w_min=0.0, w_max=3.0),
        actor_params=ActorCriticParams(
            gamma=0.97, eta_actor=0.05, trace_decay=0.7, da_gain=1.5,
            w_min=-3.0, w_max=3.0,
            normalize_advantage=True,  # the fix from docs/STATUS.md's FlappyBird finding
            # policy_gradient=True was tried and reverted -- see docs/PONG.md's
            # "proper policy gradient: tried, reverted" section. It looked
            # healthier in a short diagnostic but froze almost immediately and
            # never recovered over a full matched 2M-step run (1/7,806 scored,
            # 0.013%, vs. 2.56% without it) -- worse than either prior version,
            # not better. Left off.
        ),
    )
    ram_to_kc = RamToKC(n_in=128 * N_FRAME_STACK, n_kc=n_kc)
    env = PongEnv()

    print(f"Target: {target_steps:,} environment steps "
          f"({target_steps/DEFAULT_TARGET_STEPS*20:.1f}% of DQN's 50M-frame Atari standard)")
    t_start = time.time()

    history = []
    early_run, late_run, best_run = None, None, None
    total_steps = 0
    ep = 0
    next_checkpoint = args.checkpoint_every
    last_log_t = t_start

    while total_steps < target_steps:
        # anneal exploration over the STEP budget, not episode count, since
        # episode length itself changes a lot as (if) the policy improves
        epsilon = max(0.05, 0.3 * (1 - total_steps / (target_steps * 0.7)))
        result = run_episode(env, ac, ram_to_kc, epsilon, record=True)
        total_steps += result["steps"]
        history.append({"ep": ep, "total_reward": result["total_reward"],
                         "scored": result["scored"], "steps": result["steps"],
                         "total_steps": total_steps})
        if ep == 0:
            early_run = result
        late_run = result  # always keep the most recent; overwritten each loop
        if best_run is None or result["total_reward"] > best_run["total_reward"]:
            best_run = result
            best_run["ep"] = ep

        if time.time() - last_log_t > 5.0:  # wall-clock cadence, not episode cadence
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

        if total_steps >= next_checkpoint:
            torch.save(ac.state_dict(), "analysis/pong_checkpoint.pt")
            next_checkpoint += args.checkpoint_every

        ep += 1

    env.close()
    elapsed = time.time() - t_start
    print(f"Done: {total_steps:,} steps over {ep} episodes in {elapsed/60:.1f} minutes "
          f"({total_steps/elapsed:.0f} steps/sec)")

    out = {
        "history": history,
        "early_episode": early_run,
        "late_episode": late_run,
        "best_episode": best_run,
        "n_kc": n_kc,
        "target_steps": target_steps,
        "total_steps": total_steps,
        "wall_clock_seconds": elapsed,
    }
    with open("analysis/pong_run.json", "w") as f:
        json.dump(out, f)
    print("Wrote analysis/pong_run.json")

    torch.save(ac.state_dict(), "analysis/pong_checkpoint.pt")
    print("Wrote analysis/pong_checkpoint.pt (live/server.py warm-starts from this if present)")


if __name__ == "__main__":
    main()
