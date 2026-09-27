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

## Gameplay result

5 episodes each, `defend_the_center`, seed 0:

| Policy | Mean kills/episode | Survival | Accuracy |
|---|---|---|---|
| **Fly (real wiring)** | **1.4 +/- 0.89** | 8.2s | 0.82 |
| **Scrambled (degree-preserving)** | **0.6 +/- 0.55** | 7.9s | 0.60 |

Real wiring outperforms scrambled wiring by ~2.3x -- the same direction as
the published result (10.2 vs 0.7 kills/episode, ~14.6x), but at noticeably
smaller absolute scale.

**Being honest about what this does and doesn't establish:**

- At n=5 episodes per condition, the standard deviations are large enough
  relative to the mean difference that this is *suggestive, not
  statistically confirmed* (rough two-sample t ~= 1.7, short of the
  conventional p<0.05 threshold at this sample size). This project already
  learned the hard way on Pong that single-digit-episode comparisons can be
  noise; the honest read here is "the direction and the calibration-curve
  evidence both point the same way, but the gameplay sample is too small to
  call it confirmed" pending more episodes.
- The absolute kill rate (1.4 vs. the published 10.2) is well below the
  original. Likely reasons: `EXT_KICK_MV` (how strongly an injected Poisson
  spike drives LC10a) is a from-scratch guess tuned only enough to produce
  a sane calibration curve, not extensively fit the way the original
  authors' rate-based stimulation was; the accuracy (0.82) shows aim itself
  is fine when the circuit fires -- the gap looks like it's in how
  aggressively/quickly the circuit acquires and tracks targets, not in
  marksmanship once locked on.
- Both wall-clock runs were slowed by the same stray CPU-contention pattern
  documented elsewhere in this project (`docs/STATUS.md`) -- e.g. the fly
  calibration sweep took 8123s vs. scrambled's 98s for identical work. This
  affects wall time only, not the reported kills/survival/accuracy numbers
  themselves.

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
