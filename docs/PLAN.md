# Plan: Using the fly connectome to play Super Mario with dopamine-based learning

(See chat history / project README for the full narrative version. This file
is the working reference kept in-repo so it travels with the code.)

## 0. What's realistic
- FlyWire gives wiring (~139k neurons, ~50M synapses, predicted neurotransmitter
  per connection) — not neuron electrical parameters, synapse strengths,
  electrical synapses, or neuromodulator diffusion. Every model fills these in.
- Dopamine learning in the real fly happens mainly in the mushroom body
  (KC -> MBON synapses, gated by dopamine neurons). It learns "is this
  stimulus good or bad", not action sequences. Mario needs sequences, so an
  actor layer sits on top — clearly marked as non-biological engineering.
- A directly-mapped fly brain will likely play worse than a small RL network.
  The payoff is biological insight, not leaderboard score.

## 1. Environment
WSL2 + CUDA GPU (>=12GB) recommended. Python 3.10/3.11. torch, numpy, pandas,
pyarrow, brian2 (validation only), gym-super-mario-bros, nes-py,
opencv-python, wandb/tensorboard.

## 2. Connectome
FlyWire Codex v783 (neurons.csv, connections.csv, classification.csv), or
better: Shiu et al. 2024 (`philshiu/Drosophila_brain_model`) pre-built
Connectivity_783.parquet + Completeness_783.csv. Start from Shiu et al.
Optional: MANC/BANC, larval connectome (Winding et al. 2023, ~3k neurons) as
a small test bed.

## 3. Simulator
LIF neurons, current-based synapses. W[post,pre] = sign(nt) * syn_count *
w_scale (ACh +, GABA/Glu -). Sparse torch CSR on GPU, 1ms steps. Validate
against Shiu et al.'s sugar -> MN9 result before anything else. Build a
subcircuit extractor (visual PNs, mushroom body, central complex, descending
neurons, + 2-hop neighbors) to keep training fast (~10-25k neurons).

## 4. Mario environment
gym-super-mario-bros, SuperMarioBros-1-1-v0, SIMPLE_MOVEMENT actions, 4-frame
skip (~67ms/decision -> simulate ~50-100ms of brain time per decision).
Baselines: random, always-right-jump, PPO CNN upper bound.

## 5. Sensory interface
Stage A (do first): hand-crafted feature vector (enemy distance/height, gap
ahead, obstacle ahead, airborne) -> firing rates -> projection neurons ->
mushroom body Kenyon cells (uses the connectome's own sparse random KC code).
Stage B (harder, later): downsampled frame -> hexagonal ommatidia grid ->
photoreceptors -> optic lobe (lamina/medulla/lobula), check T4/T5 motion
responses before relying on them.

## 6. Motor interface
Descending neurons: P9/DNp09 (forward), MDN (backward), Giant Fiber DNp01
(jump), fast-walking DNs (run/B), DNa01/DNa02 (turning/tie-break). Decode by
comparing spike counts vs baseline over the decision window; softmax + noise
for exploration.

## 7. Dopamine learning
7a. Mushroom body core: PAM (reward) / PPL1 (punishment) dopamine neuron
groups, three-factor rule at KC->MBON synapses (dopamine depresses
recently-active KC synapses), eligibility trace ~1-2s.
7b. Actor-critic engineering layer (explicitly non-biological): critic =
approach/avoid MBON balance -> V(s); dopamine = reward prediction error
delta = r + gamma*V(s') - V(s); actor = learnable KC/MBON -> DN weights
updated with the same three-factor rule and eligibility traces. Rest of
connectome fixed (reservoir).
7c. Reward shaping: r = 0.1*dx - 0.01/step - 15 on death + 50 on flag,
clipped to [-1, 1].

## 8. Milestones
M1 toy two-odor conditioning learns in ~10 trials.
M2 Mario, Stage A features, subcircuit: beats random.
M3 clears first Goomba + pipes, >=50% runs reach x~600.
M4 beats "always right+jump" baseline.
M5 Stage B screen input or full-brain check, keeps most of M3 performance.
Stretch: clears World 1-1.

## 9. Analysis
Log reward/distance/DA activity/weight changes. Lesion PAM neurons and check
learning stops. Compare against random-wiring control with matched neuron/
synapse counts.

## 10. Risks
Runaway/dead activity -> tune w_scale + global inhibition, start from Shiu
params. DN barely responsive to MB changes -> use actor readout (7b). Too
slow -> subcircuit, shorter decision window, batched envs. Credit assignment
-> longer eligibility traces + reward-prediction-error signal. Screen input
too noisy -> stick to Stage A for main results.

## References
Dorkenwald et al. 2024 & Schlegel et al. 2024 (Nature, FlyWire connectome);
Shiu et al. 2024 (Nature, whole-brain spiking model); Aso et al. 2014 (eLife,
MB compartments/DANs); Eschbach et al. 2020 (Nat. Neurosci., DA-based MB
learning model); Hige et al. 2015 (Neuron, DA depresses KC->MBON synapses).
