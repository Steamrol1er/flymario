"""Fly Marksman reproduction: our own extracted LC10a->AOTU->DNa02 circuit
(connectome/marksman_circuit.py, 3,963 real FlyWire v783 neurons) driving
ViZDoom's stock `defend_the_center` scenario through an UNTRAINED,
real-connectome LIF network (sim/marksman_brain.py). There is no learning
anywhere in this script: the wiring itself is the entire "policy". This is
what makes it a genuinely different kind of result from the rest of this
project (dopamine and hybrid Pong both LEARN); it directly tests the claim
that real fly wiring, run forward with no training at all, out-performs the
same wiring scrambled, which is the actual published Fly Marksman result
(webergithub/fruitfly-lab: 10.2 kills/episode real vs 0.7 scrambled).

Policies (--policy):
  fly            real wiring, real calibrated thresholds
  scrambled      degree-preserving scramble, reusing the REAL circuit's
                 thresholds (the direct, published comparison)
  scrambled_cal  same scramble, using its OWN calibration (fairer, isolates
                 whether scrambled wiring can be salvaged by re-tuning)
  lesion         real wiring with AOTU019_L/R + AOTU025_L/R silenced
  lesion025      real wiring with only AOTU025_L/R silenced
  random         uniform random button presses (floor baseline)
  oracle         steer/fire directly from ground-truth enemy azimuth,
                 bypassing the brain entirely (ceiling baseline)

Run:
  python -m experiments.run_marksman --calibrate
  python -m experiments.run_marksman --policy fly --episodes 10
  python -m experiments.run_marksman --policy scrambled --episodes 10
Writes: analysis/marksman_calibration_{fly,scrambled}.json,
        analysis/marksman_{policy}.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from connectome.marksman_circuit import load_marksman_circuit, scrambled_copy
from env.vizdoom_marksman_env import MarksmanEnv
from sim.marksman_brain import FlyBrain, steer_signal, PANEL

OUT_DIR = Path("analysis")
CAL_AZ = [-90, -75, -60, -45, -30, -20, -10, -5, 0, 5, 10, 20, 30, 45, 60, 75, 90]
POLICIES = ["fly", "scrambled", "scrambled_cal", "lesion", "lesion025", "random", "oracle"]


def load_cal(name: str) -> dict:
    return json.loads((OUT_DIR / f"marksman_calibration_{name}.json").read_text())["cal"]


def make_policy(name: str, circuit, seed: int):
    """Returns (act(targets) -> ([left,right,fire], rates_or_None), brain_or_None)."""
    if name in ("random", "oracle"):
        rng = np.random.default_rng(seed)

        def act(targets):
            if name == "random":
                return [int(x) for x in rng.integers(0, 2, 3)], None
            if not targets:
                return [0, 1, 0], None
            az = min(targets, key=lambda t: abs(t[0]))[0]
            return [int(az < -6.0), int(az > 6.0), int(abs(az) < 6.0)], None
        return act, None

    c = scrambled_copy(circuit, seed=seed + 100) if name.startswith("scrambled") else circuit
    lesion_groups = {"lesion": ["AOTU019_L", "AOTU019_R", "AOTU025_L", "AOTU025_R"],
                      "lesion025": ["AOTU025_L", "AOTU025_R"]}.get(name, [])
    silenced = np.concatenate([circuit.groups[g] for g in lesion_groups]) if lesion_groups else None

    brain = FlyBrain(c, seed=seed, silenced=silenced)
    cal = load_cal("scrambled" if name == "scrambled_cal" else "fly")

    def act(targets):
        r = brain.step(targets)
        s = steer_signal(r)
        left, right = int(s < -cal["turn_th"]), int(s > cal["turn_th"])
        fire = int(min(r["AOTU019_L"], r["AOTU019_R"]) > cal["fire_th"])
        if not targets and sum(r.values()) < cal["quiet_th"]:
            left, right = 0, 1  # no visual input: search right (same engineered behavior as oracle)
        return [left, right, fire], dict(r)

    return act, brain


def calibrate(which: str, seed: int = 0) -> dict:
    """Sweep azimuth -90..+90 with no learning, measure the circuit's own
    steering-tuning curve, and set fixed readout thresholds from it. Run
    once per wiring (fly / scrambled); the resulting thresholds are then
    frozen and reused for every subsequent episode of that policy."""
    circuit = load_marksman_circuit()
    c = scrambled_copy(circuit, seed=seed + 100) if which == "scrambled" else circuit
    rows = []
    t0 = time.time()
    for az in CAL_AZ:
        brain = FlyBrain(c, seed=seed)
        for _ in range(10):
            brain.step([(float(az), 1.0)])  # let smoothed rates settle
        acc = {k: 0.0 for k in PANEL}
        for _ in range(20):
            r = brain.step([(float(az), 1.0)])
            for k in PANEL:
                acc[k] += r[k] / 20
        rows.append({"az": az, **{k: round(v, 2) for k, v in acc.items()},
                      "steer": round(steer_signal(acc), 2)})
        print(f"[{which}] az={az:+4d} deg  DNa02 L/R={acc['DNa02_L']:6.1f}/{acc['DNa02_R']:6.1f}  "
              f"AOTU019 L/R={acc['AOTU019_L']:6.1f}/{acc['AOTU019_R']:6.1f}  "
              f"steer={rows[-1]['steer']:+7.1f}", flush=True)

    by = {r["az"]: r for r in rows}
    s0 = abs(by[0]["steer"])
    s_off = min(abs(by[-10]["steer"]), abs(by[10]["steer"]))
    turn_th = (s0 + s_off) / 2 if s_off > s0 else max(1.1 * s0, 1.0)
    fire_th = max(0.5 * min(by[0]["AOTU019_L"], by[0]["AOTU019_R"]), 1.0)
    cal = {"turn_th": round(turn_th, 2), "fire_th": round(fire_th, 2), "quiet_th": 5.0,
           "note": "turn_th = midpoint of |steer| at 0 deg and +/-10 deg; "
                   "fire_th = half the smaller of the two AOTU019 rates at 0 deg"}
    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / f"marksman_calibration_{which}.json").write_text(
        json.dumps({"rows": rows, "cal": cal, "seconds": round(time.time() - t0, 1)}, indent=1))
    print(f"[{which}] readout thresholds: {cal}  ({time.time() - t0:.1f}s)")
    return cal


def play(policy: str, episodes: int, seed: int, visible: bool = False) -> dict:
    circuit = load_marksman_circuit()
    env = MarksmanEnv(seed=seed, visible=visible)
    eps = []
    OUT_DIR.mkdir(exist_ok=True)
    for ep in range(episodes):
        act, _ = make_policy(policy, circuit, seed + ep)
        env.new_episode()
        ammo0 = env.ammo()
        t0 = time.time()
        while not env.is_finished():
            targets = env.enemies()
            action, _rates = act(targets)
            env.step(*action)
        kills = env.kills()
        shots = int(ammo0 - env.ammo())
        tics = env.tics()
        eps.append({"episode": ep, "kills": kills, "shots": shots, "tics": tics,
                    "accuracy": round(kills / shots, 3) if shots else 0.0,
                    "wall_s": round(time.time() - t0, 1)})
        print(f"[{policy}] ep{ep}: kills={kills} shots={shots} "
              f"survived={tics / 35:.1f}s wall={eps[-1]['wall_s']}s", flush=True)
    env.close()

    k = np.array([e["kills"] for e in eps], dtype=float)
    summary = {"kills": round(float(k.mean()), 2),
               "kills_sd": round(float(k.std(ddof=1)) if len(k) > 1 else 0.0, 2),
               "shots": round(float(np.mean([e["shots"] for e in eps])), 2),
               "survival_s": round(float(np.mean([e["tics"] for e in eps])) / 35, 2),
               "accuracy": round(float(np.mean([e["accuracy"] for e in eps])), 3)}
    out = {"policy": policy, "episodes": eps, "summary": summary, "seed": seed}
    (OUT_DIR / f"marksman_{policy}.json").write_text(json.dumps(out, indent=1))
    print(f"[{policy}] mean kills {summary['kills']} +/- {summary['kills_sd']}, "
          f"survival {summary['survival_s']}s, accuracy {summary['accuracy']}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", default="fly", choices=POLICIES)
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--visible", action="store_true")
    ap.add_argument("--calibrate", action="store_true")
    args = ap.parse_args()
    OUT_DIR.mkdir(exist_ok=True)
    if args.calibrate:
        calibrate("fly", args.seed)
        calibrate("scrambled", args.seed)
    else:
        play(args.policy, args.episodes, args.seed, args.visible)


if __name__ == "__main__":
    main()
