# Fly Marksman reproduction

An independent reproduction of `webergithub/fruitfly-lab`'s "Fly Marksman":
a real FlyWire visual-pursuit circuit (LC10a -> AOTU019/AOTU025 -> DNa02)
playing ViZDoom's stock `defend_the_center` scenario with **zero training**
-- the wiring itself is the entire policy. This is a different kind of
result from the rest of this project (dopamine Pong and hybrid Pong both
*learn*); it tests whether real connectome structure alone, run forward,
does something a scrambled version of the same structure can't.

## What's reproduced independently vs. borrowed

- **Borrowed**: which 3,963 of FlyWire's 138,639 neurons belong to this
  pathway, and their cell-type/side labels (`data/marksman_reference/
  fruitfly_lab_meta.json`, fetched from the original repo). That's their
  annotation work, matching FlyWire Codex cell types to root IDs -- this
  project has no Codex account to redo it from scratch.
- **Independently extracted**: the actual synaptic weights among exactly
  those 3,963 neurons, pulled from our own already-downloaded
  `Connectivity_783.parquet` / `Completeness_783.csv` (Shiu et al. 2024
  tables), not their `edges.bin`. All 3,963 of their root IDs were verified
  present in our own data (100% overlap, both FlyWire v783) before
  extraction. See `connectome/marksman_circuit.py`.
- **Verified**: `python -m connectome.marksman_circuit` reproduces their
  published circuit size exactly: 3,963 neurons, 354,879 edges (220,442
  excitatory, 134,437 inhibitory), matching group sizes (LC10a_L: 115,
  LC10a_R: 119, AOTU019/AOTU025/DNa02: 1 each per side).
- **Independently implemented, not copied**: the LIF controller
  (`sim/marksman_brain.py`), the degree-preserving scramble
  (`connectome/marksman_circuit.py::scrambled_copy`), and the play/calibrate
  loop (`experiments/run_marksman.py`) are all fresh implementations of the
  same published method (LC10a eccentricity-tuned Poisson drive -> real
  synaptic LIF propagation -> fixed-threshold readout on AOTU019/DNa02),
  built on this project's own simulator (`sim/lif.py`) rather than their
  code. One real difference: our simulator takes an external *current* per
  0.1ms step rather than a stimulation *rate* directly, so a target firing
  rate here is converted to independent Poisson spike events, each
  injecting a fixed membrane kick (`EXT_KICK_MV`, a tuned-not-fit free
  parameter). Because thresholds are calibrated from our own circuit's
  response curve, this only needs to give a sane, monotonic tuning curve --
  not to numerically match their internal rate scale.

## Calibration: does the real wiring even produce a sensible tuning curve?

Sweeping target azimuth -90 deg to +90 deg with no learning
(`python -m experiments.run_marksman --calibrate`):

**Real wiring**: steering signal swings smoothly and correctly-signed from
-119.9 (target hard left) through the sign flip near 0 deg to +86.0 (target
hard right), with clean separation by hemifield.

**Scrambled wiring**: steering signal is weak (magnitude 5-20 vs 60-130)
and, over most of the range, *wrong-signed* relative to the real circuit's
convention -- not just noisier, but its structure doesn't reliably encode
target side at all.

This alone is a meaningful "wiring matters" result before any gameplay is
run: the same neuron count, same edge count, same sign balance, same LIF
parameters -- only the specific pattern of who-connects-to-whom differs.

Fixed readout thresholds from this sweep (frozen for all subsequent
episodes): real wiring `turn_th=72.74, fire_th=2.5`; scrambled wiring (its
own calibration) `turn_th=10.11, fire_th=8.0`.

## Gameplay result (20 episodes per policy, seed 0)

| Policy | Kills/episode (mean +/- SD) | Survival | Accuracy |
|---|---|---|---|
| Oracle (ceiling, no brain) | 13.5 +/- 1.9 | 21.3s | 0.52 |
| **Fly (real wiring)** | **1.55 +/- 1.32** | 8.9s | 0.63 |
| Random button presses (floor) | 1.10 +/- 0.64 | 8.6s | 0.06 |
| Scrambled wiring, real thresholds | 0.75 +/- 0.55 | 8.2s | 0.59 |
| Scrambled wiring, own calibration | 0.45 +/- 0.51 | 8.6s | 0.29 |

Statistical tests, fly vs each control (Welch t, one-sided Mann-Whitney):

| Comparison | Welch t | p (two-sided) | Mann-Whitney p (fly greater) |
|---|---|---|---|
| fly vs scrambled | 2.51 | 0.019 | 0.015 |
| fly vs scrambled_cal | 3.48 | 0.002 | 0.0006 |
| fly vs random | 1.37 | 0.18 | 0.21 |

**What this supports and what it doesn't**

- **Supported:** real wiring scores more kills than degree-preserving
  scrambles of the same circuit (p about 0.02 and 0.002). With the
  calibration curves (real wiring gives a clean, correctly signed steering
  signal; scrambled does not), this is evidence that the specific wiring
  matters, which is the direction of the published result.
- **Not supported:** that the circuit beats chance. Its 1.55 kills is
  statistically indistinguishable from random button presses (1.10, p=0.18).
  The scrambled circuits score *below* random, mostly because they rarely
  fire, so the real finding is "scrambled is worse than random", not
  "fly is much better than random".
- **Where the fly does clearly better:** shooting accuracy, 0.63 vs 0.06 for
  random. It fires when aimed. It just turns too slowly to acquire enough
  targets, so total kills stay low. The oracle shows the headroom: about 13.5.
- **Scale gap:** the published result is 10.2 kills for real wiring. Ours is
  about 1.5. The likely cause is our `EXT_KICK_MV` external-drive constant,
  tuned only enough to get a sane calibration curve, not fit for performance.
  This is untested and is the first thing to try.
- Earlier 5-episode pilot (fly 1.4, scrambled 0.6) was consistent with these
  numbers but too small to conclude anything.

## What the "oracle" is

`oracle` is a baseline that bypasses the fly brain entirely. Each tic it
reads the true angle of the nearest visible enemy from ViZDoom's object
labels (the same input the fly circuit receives) and applies a hard-coded
rule: turn left if the enemy is more than 6 degrees left of center, turn
right if more than 6 degrees right, attack if within 6 degrees, and turn
right to search when nothing is visible. It aims perfectly and instantly, so
it measures how many kills are possible with ideal steering. `random`
presses buttons uniformly and is the floor.

## What the fly does and does not do here

There is no learning anywhere in this experiment. Wiring and parameters are
fixed, and only the readout thresholds are set once by a calibration sweep.
Several parts are engineered rather than biological: enemy bearing comes from
ViZDoom object labels instead of simulated eyes, and the thresholds, fire
rule and search-right default are hand-made. Only 3,963 of the connectome's
138,639 neurons are simulated, because the pursuit pathway is all this
behavior needs.

Wall-clock times were inflated by CPU contention (for example the first fly
calibration sweep took 8123s vs 98s for identical work). That affects run
time only, not the reported scores.

## Reproduce it

```
python -m experiments.run_marksman --calibrate --seed 0
python -m experiments.run_marksman --policy fly --episodes 10 --seed 0
python -m experiments.run_marksman --policy scrambled --episodes 10 --seed 0
```

Other policies available for further comparison (not yet run here):
`scrambled_cal` (scramble re-calibrated on its own curve, a fairer control),
`lesion` / `lesion025` (silencing AOTU019/AOTU025), `random` and `oracle`
(floor/ceiling baselines that bypass the brain entirely).
