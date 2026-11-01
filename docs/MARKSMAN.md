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

## Results

There are two stages: an initial untuned run, then a drive-strength sweep
followed by a held-out test. The held-out test is the number to cite.

### Held-out result (tuned drive, fresh seeds)

External drive set to 25 mV per injected spike (chosen by the sweep below),
calibrated on seed 1000, then 20 episodes per policy on seed 1000. These
seeds were not used for tuning.

| Policy | Kills/episode (mean +/- SD) | Median | Survival | Accuracy |
|---|---|---|---|---|
| Oracle (ceiling, no brain) | 12.4 +/- 2.7 | 14 | 20.8s | 0.48 |
| **Fly (real wiring)** | **8.0 +/- 5.2** | 6 | 15.7s | 0.44 |
| Lesion AOTU025 only | 7.85 +/- 5.3 | 6.5 | 15.4s | 0.48 |
| Random button presses | 1.05 +/- 0.69 | 1 | 8.7s | 0.06 |
| Scrambled, recalibrated (`scrambled_cal`) | 1.0 +/- 0.46 | 1 | 8.9s | 0.39 |
| Scrambled wiring | 0.0 +/- 0.0 | 0 | 8.3s | 0.00 |
| Lesion AOTU019 + AOTU025 | 0.0 +/- 0.0 | 0 | 8.3s | 0.00 |

| Comparison | Welch t | p (two-sided) | Mann-Whitney p (fly greater) |
|---|---|---|---|
| fly vs scrambled | 6.88 | 1.5e-6 | 3.9e-9 |
| fly vs random | 5.92 | 9.2e-6 | 5.9e-7 |
| fly vs scrambled_cal | 6.00 | 8.5e-6 | 2.0e-7 |
| fly vs lesion (AOTU019 + AOTU025) | 6.88 | 1.5e-6 | 3.9e-9 |
| fly vs lesion AOTU025 only | 0.09 | 0.93 | 0.45 |

Controls (same seed, drive and episode count):

- **Recalibrated scramble.** `scrambled_cal` gets readout thresholds from its
  own calibration sweep instead of the real circuit's. It rises only to random
  level (1.0 vs 1.05, p = 0.79), so the scrambled 0.0 is not an artifact of
  borrowed thresholds. Caveat: thresholds come from one scramble (calibration
  seed 1000), while each episode plays a different scramble.
- **Lesion of both AOTU types** removes the behavior completely: zero shots
  fired in all 20 episodes. The fire rule reads AOTU019, so this is partly
  by construction; it confirms the readout gets nothing from elsewhere.
- **Lesion of AOTU025 alone** has no measurable effect (7.85 vs 8.0). In this
  model AOTU025 is dispensable; the behavior runs through AOTU019 and DNa02.

The real circuit now clearly beats both the scrambled circuits and random
play. All 20 scrambled episodes scored zero kills. The fly reaches about 65%
of the oracle ceiling on average, with a wide spread (1 to 17 kills).

### Drive-strength sweep (tuning seed 500, 10 episodes each)

The first run used a drive constant (`EXT_KICK_MV`, membrane kick per injected
Poisson spike) of 6 mV that I had set only to get a sane calibration curve.
Each setting below was re-calibrated on its own sweep, then played against the
real circuit and against scrambled rewirings (a different rewiring each episode).

| Drive per spike | Fly kills | Scrambled kills |
|---|---|---|
| 3 mV | 1.8 +/- 0.8 | 0.1 +/- 0.3 |
| 10 mV | 1.7 +/- 2.0 | 0.3 +/- 0.5 |
| 15 mV | 2.9 +/- 1.9 | 0.2 +/- 0.6 |
| 25 mV | 7.1 +/- 4.4 | 0.0 +/- 0.0 |

Caveats:

- **Selection bias.** 25 mV was picked because it scored best on the tuning
  seeds, so its 7.1 is optimistic. The held-out seeds (above) are the fair
  estimate. They came out similar (8.0), so the choice does not look like
  an overfit to the tuning seeds.
- **Top of the range tried.** 25 mV was the largest value swept. Performance
  was still rising, so a higher drive could do better. Not tested.
- **The weaker drive misled the first result.** With the original 6 mV
  setting (20 episodes, seed 0) the fly scored 1.55 +/- 1.32, not
  significantly better than random (1.10 +/- 0.64, p = 0.18), although it
  already beat scrambled wiring (0.75, p = 0.02; recalibrated scramble 0.45,
  p = 0.002). The conclusion "the circuit does no better than chance" was a
  property of an untuned parameter, not of the wiring.

### What this supports

- Real wiring produces far more kills than degree-preserving scrambles of
  the same circuit (0.0 vs 8.0 held-out), and beats random presses, with no
  learning. This reproduces the direction of the published result (10.2 real
  vs 0.7 scrambled). Our 8.0 is somewhat lower than 10.2 and is from a
  different, independently written implementation.
- What it does not show: that any of this is "intelligence". The circuit is a
  fixed steering reflex fed by engineered inputs (below).
- Scrambling was done with 20 different rewirings per condition (one per
  episode), not a single scramble.

## What the "oracle" is

`oracle` is a baseline that bypasses the fly brain entirely. Each tic it
reads the true angle of the nearest visible enemy from ViZDoom's object
labels (the same input the fly circuit receives) and applies a hard-coded
rule: turn left if the enemy is more than 6 degrees left of center, turn
right if more than 6 degrees right, attack if within 6 degrees, and turn
right to search when nothing is visible. It aims perfectly and instantly, so
it measures how many kills are possible with ideal steering (about 12-13). `random`
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

The held-out table uses `--seed 1000 --kick 25 --tag _best --episodes 20`
(calibrate first with `--calibrate --cal-only fly`, and `--cal-only scrambled`
for `scrambled_cal`). Other policies: `scrambled_cal`, `lesion`, `lesion025`,
`random`, `oracle`.
