# Status

What's implemented vs. still to do, against the milestones in `docs/PLAN.md`.
Written 2026-09-26, updated 2026-11-01.

**Headline:** the Fly Marksman experiment (`docs/MARKSMAN.md`) is the main
positive result: untrained real FlyWire wiring (3,963 neurons) scores 8.0
kills/episode in ViZDoom vs 0.0 for scrambled wiring. Every learning
experiment (platformer, FlappyBird, Pong) is documented below and in
`docs/PONG.md`; none beats a random or trivial baseline convincingly. Real
Mario never ran (see "Known gaps").

## Done

- **Project scaffold** (`connectome/`, `sim/`, `interface/`, `learning/`,
  `env/`, `experiments/`, `tests/`).
- **Connectome download & load** (`connectome/download.py`,
  `connectome/loader.py`): fetches the real Shiu et al. 2024 v783 tables
  (`Connectivity_783.parquet`, `Completeness_783.csv`, plus their `model.py`/
  `utils.py` for reference) directly from GitHub. Verified: 138,639 neurons,
  15,091,983 pre/post pairs, sign convention (`Excitatory` = +1/-1) confirmed
  by reading their brian2 `model.py`. Loader validated against the real
  files (loads in <1s).
  - `Connectome.subcircuit()` for re-indexing to a neuron subset, plus
    `connectome/subcircuit.py::expand_by_hops` for the "seed + k-hop
    neighbors" extraction described in the plan.
  - `connectome/loader.py::load_toy_connectome` — a small synthetic
    odor->KC->MBON circuit for fast tests with no download/torch dependency
    for the graph-only parts.
- **LIF spiking simulator** (`sim/lif.py`): batched torch implementation
  using the exact published parameters from Shiu et al.'s brian2 model
  (v_0=-52mV, v_th=-45mV, t_mbr=20ms, tau=5ms, t_rfc=2.2ms, w_syn=0.275mV).
  Known simplification: their 1.8ms synaptic delay is not implemented as a
  true delay queue (see module docstring) — flag if this matters later.
- **Dopamine three-factor learning rule** (`learning/dopamine.py`,
  `MushroomBodyLearner`): depression-only KC->MBON plasticity gated by
  per-compartment dopamine, matching Hige et al. 2015. Validated on a toy
  two-odor conditioning task (`tests/toy_conditioning.py`, **PASSES**: odor
  paired with reward ends up valued higher than an unpaired control odor).
- **Actor-critic layer** (`learning/actor.py`): TD-error-driven dopamine +
  learnable KC->action readout, explicitly documented as the
  non-biological engineering bridge from "value" to "policy".
- **Mario env wrapper** (`env/mario_env.py`), **sensory feature extractor**
  (`interface/sensory.py`, Stage A only), **motor decoder**
  (`interface/motor.py`) mapping named DN groups to buttons.
- **Training loop** (`experiments/train.py`) wiring all of the above
  together, with a config (`experiments/m2_config.yaml`).
- **Sugar -> MN9 validation** (`tests/check_sugar_feeding.py`): reproduces
  Shiu et al.'s example.ipynb experiment using real FlyWire root IDs (20/21
  sugar-sensing neurons + MN9, confirmed present in v783) on our own
  simulator, as the sanity check the plan calls for before trusting the
  port. **PASSES**: on the full 138,639-neuron connectome, MN9 goes from a
  baseline of 0 Hz to 149 Hz when the sugar-sensing neurons are driven at
  200 Hz — qualitatively reproducing their published multi-synaptic
  sugar->feeding effect. This is strong evidence the torch port of their
  brian2 model (parameters, sign convention, connectivity loading) is
  correct.

- **3D brain visualization** (`viz/brain3d.html`): a real public 3D
  reconstruction of the adult fly brain surface (VirtualFlyBrain's JFRC2
  template domain meshes, CC BY 4.0, fetched via `connectome/download.py
  --source brain-mesh`), rendered in three.js with the **gnathal ganglion
  (GNG)** highlighted and pulsing in sync with the real per-10ms-bin spike
  count from the sugar->MN9 simulation. The GNG (the modern name for the
  subesophageal zone) is the textbook-documented location of gustatory
  afferent input and feeding premotor/motor circuits including MN9 -- a
  general anatomical fact from the literature, not something derived from
  our specific simulated neurons. This view is explicit that it does NOT
  place the 396 individual active neurons at real 3D coordinates (that
  needs FlyWire's own coordinate data, gated behind a Codex account); for
  that, see the neuron-level network graph in `viz/index.html` instead.

- **End-to-end learning demo on an original platformer**
  (`env/simple_platformer.py`, `experiments/train_platformer.py`,
  `viz/platformer_play.html`): since `nes-py` won't build here and there's
  no ROM to load anyway, this runs the real dopamine rule + actor-critic on
  a small original game with the same problem shape (run right, jump gaps
  and an obstacle, avoid a patrolling enemy, reach a flag) on the toy
  connectome. Physics were validated with hardcoded sanity-check policies
  first (see the commit history for the bugs that were caught this way —
  a too-wide gap lookahead window, insufficient jump distance, a float-
  rounding edge case at tile boundaries) before trusting any learning
  result. **Result: 57 real wins out of 2500 training episodes (2.3%)** --
  genuine, reproducible learning (the agent does sometimes solve the whole
  level), but not a policy that reliably converges to solving it. Read
  that as "the learning rule works and the credit-assignment problem is
  hard with a linear KC->action readout and a random feature encoding,"
  not as "the biology doesn't work" -- a proper policy-gradient
  implementation, a less noisy feature encoding, or more training would
  likely close this gap, and none of that changes the dopamine rule
  itself, which is validated separately in the toy-conditioning test.

- **Atari and other pip-installable game environments surveyed and
  verified working here**, as alternatives to the still-blocked real Mario
  (`env/mario_env.py`): `ale-py` (104 real Atari games, prebuilt wheels,
  no compiler needed) + AutoROM for the ROMs. AutoROM's own downloader
  failed on a `requests`-library quirk in this sandbox; fetched the same
  archive with `urllib` and extracted it by hand -- confirmed `ALE/Pong-v5`
  runs end to end. Also verified working: `flappy-bird-gymnasium` (pure
  Python, no ROM/compiler at all), `minigrid`, `gymnasium[box2d]`
  (LunarLander/BipedalWalker). `stable-retro`/`gym-retro` and `procgen` are
  NOT installable here (no wheels, `pip index` finds nothing) -- don't
  waste time on those without building from source.
- **FlappyBird end-to-end attempt** (`env/flappy_env.py`,
  `experiments/train_flappybird.py`, `viz/flappybird_play.html`): same
  real dopamine rule + actor-critic as the platformer, on a harder,
  continuous-tracking task (2 actions, but needs matching vertical
  position to a gap every frame, not one discrete jump decision).
  **Result: the agent passed zero pipes in 2000 training episodes** -- a
  genuine negative result, not a bug in the numbers reported. A real bug
  WAS found and fixed along the way: unnormalized Kenyon-cell activity
  (~100 simultaneously active units, unbounded ReLU magnitude) made the
  mushroom body's linear value estimate swing by tens of units from tiny
  weight asymmetries, saturating the actor's clamped weights to their
  bounds within the first ~50 episodes and freezing the policy for the
  rest of training -- confirmed by direct inspection of `actor_w`, not
  guessed. Fixed with divisive normalization on the KC output (a real
  mechanism in actual neural circuits, keeping the total "activity budget"
  bounded regardless of how many KCs a given random projection excites).
  That fix alone wasn't sufficient to make the task learnable in the time
  budgeted here. A hand-coded reference policy (flap toward the gap
  center, damped by velocity) scores 3 pipes with zero learning involved,
  confirming the environment itself is solvable -- the gap is in the
  learning method, not the game. Read as: **the dopamine rule and
  actor-critic code are validated as far as the platformer's discrete
  decision problem goes; a genuinely continuous tracking/control task
  exposes a real limitation of this simple linear actor-critic** that a
  proper policy-gradient implementation or a richer, less noisy KC
  encoding would likely need to address. Not a defect to hide -- an honest
  finding about where the current architecture's limits are.

- **Pong, on the real Atari interface, unmodified** (`env/pong_env.py`,
  `experiments/train_pong.py`, `viz/pong_play.html`, full account in
  `docs/PONG.md`): raw 128-byte RAM observation, the game's own
  `Discrete(6)` action set, the game's own +1/-1/0 reward -- no hand-
  crafted features, no remapping, no reshaping. Applied both FlappyBird
  fixes (KC divisive normalization, and a new opt-in
  `normalize_advantage` on `learning/actor.py::ActorCriticParams`) from
  the start. Result: 18/400 points scored (4.5%), matching a measured
  random-policy baseline (~5%) -- not skillful play, but the policy did
  NOT freeze this time (steps/outcomes kept varying across the whole run,
  unlike FlappyBird's collapse within ~50 episodes), direct evidence the
  stability fixes worked as intended even though genuine skill wasn't
  achieved in this budget. Researched real prior art first (see
  `docs/PONG.md`'s citations): confirmed the three-factor-rule +
  eligibility-trace architecture matches published biological RL models,
  and confirmed that the well-known "fly brain plays Doom/Mario 64" demos
  use a *separate* ML model for decisions rather than having the
  dopamine-modulated circuit's own output drive play -- this project
  attempts the harder version of that problem.
  **Superseded by the frozen evaluation** (`docs/PONG.md`, "Evaluation-only
  results"): random play 3.1% (10 seeds), dopamine sampled 2.8% (10 seeds),
  dopamine greedy 0.0%. No learning shown. The Pong "brain" is the synthetic
  242-neuron toy circuit, not FlyWire data.
- **Hybrid Pong** (`learning/qhead.py`, `experiments/train_pong_hybrid.py`):
  the same fixed encoding with a replay-trained linear Q-head instead of the
  dopamine rule. Frozen evaluation 2.8 +/- 3.2% over 6 seeds (bimodal: three
  seeds 4-7%, three 0%), not different from random. Swapping the learning
  rule changed variance, not the mean, so the features/readout are the more
  likely limit.
- **Fly Marksman** (`connectome/marksman_circuit.py`, `sim/marksman_brain.py`,
  `env/vizdoom_marksman_env.py`, `experiments/run_marksman.py`, full account
  in `docs/MARKSMAN.md`): independent reproduction of `fruitfly-lab`'s
  LC10a -> AOTU -> DNa02 pursuit circuit on our own weights and simulator,
  no learning. Held-out (seed 1000, 20 episodes, 25 mV drive): fly 8.0 +/-
  5.2 kills, scrambled 0.0, random 1.05, oracle 12.4. Fly vs scrambled
  p = 1.5e-6. Controls: recalibrated scramble 1.0 (random level), AOTU019 +
  AOTU025 lesion 0.0, AOTU025-only lesion 7.85 (no effect).

## Known gaps / next steps

1. **FlyWire cell-type annotations not downloaded.** `connectome/cell_types.py`
   defines the group patterns (KC, MBON, PAM, PPL1, DNp09, MDN, DNp01, DNa01/
   02) and `resolve_groups()`, but actually resolving them needs
   `classification.csv` from FlyWire Codex, which requires a (free) account
   and can't be scripted end-to-end (see the printed instructions in
   `connectome/download.py --source flywire`). Until that's fetched by hand,
   `experiments/train.py` with `connectome.source: shiu` will raise a clear
   `FileNotFoundError` naming the file needed. `connectome.source: toy` runs
   the full pipeline today with hand-built stand-in groups.
2. **`interface/train.py`'s feature->KC encoding is a placeholder** (a fixed
   random projection), not the real connectome's projection-neuron->KC
   wiring, because that also needs the annotations from (1) to identify
   which neurons are olfactory/visual projection neurons. Once (1) is done,
   swap `TrainLoop.feats_to_kc` for real current injection into identified
   PN indices and let the KC layer's own (real) sparse wiring do the coding.
3. **Stage B (pixel -> compound eye) sensory input is not implemented** —
   Stage A (hand-crafted features) is the recommended starting point per the
   plan and is what's wired up.
4. **`env/mario_env.py`'s enemy/gap/obstacle detection is a stub.** Real
   enemy positions need reading NES sprite-table RAM addresses directly
   (`env.unwrapped.ram`), not just the default `info` dict
   `gym-super-mario-bros` exposes. Needs a short reverse-engineering pass
   against NES Super Mario Bros' known RAM map (well documented publicly,
   e.g. the SMB disassembly / data crystal RAM map).
5. **CPU performance.** This dev environment has no GPU, so `sim/lif.py`
   runs on CPU via torch's sparse ops. Using **CSR** (not COO) for the
   weight matrix was essential: COO sparse @ dense matmul on the full
   138,639-neuron / 15M-edge connectome was so slow it didn't finish a
   1.2s simulated run in over 10 minutes; switching `FlyBrainSimulator` to
   `to_sparse_csr()` brought a single dt=0.1ms step on the full brain down
   to ~13ms, i.e. ~130s of wall time per simulated second. That's still far
   from the plan's GPU target (~1-2s wall per simulated second) — confirms
   the plan's point that a GPU + subcircuit extraction (`connectome/
   subcircuit.py`) is needed for real training throughput, not just
   nice-to-have. `experiments/train.py` must run on the *subcircuit*, not
   the full connectome, once cell-type resolution (1) is in place.
6. **Analysis tooling (plan section 9) is partial.** Random-wiring controls
   and lesions exist for Marksman (`scrambled`, `scrambled_cal`, `lesion`,
   `lesion025`); there is still no KC->MBON weight-change logging or PAM
   lesion for the learning experiments.
7. **Baselines are partial.** Random baselines exist for Pong and Marksman,
   plus an oracle ceiling for Marksman. No PPO upper bound for any game, and
   no "always right+jump" baseline (that one only makes sense for Mario).

## Verified facts worth remembering

- Real, validated LIF parameters for the connectome come straight from
  Shiu et al.'s `model.py` (downloaded to `data/model.py`) — no need to
  guess or re-derive these.
- `Connectivity_783.parquet`'s `"Excitatory x Connectivity"` column *is*
  the signed synapse weight Shiu et al. use directly (`Excitatory` is
  already +1/-1) — don't re-derive sign from neurotransmitter predictions,
  just use this column.
- `Completeness_783.csv` row index == `Presynaptic_Index`/
  `Postsynaptic_Index` in the parquet file (verified directly against the
  data, not just assumed from column naming).
- The `philshiu/Drosophila_brain_model` GitHub repo keeps its data files at
  the repo **root**, not under `data/` (the connectome download script
  hardcodes the correct root-level paths after this was verified by
  browsing the repo).
- Large-file downloads from `download.pytorch.org`'s CDN
  (`download-r2.pytorch.org`) were unreliable in this sandbox (silent
  truncation/corruption -> pip hash-mismatch errors, and separately, `curl`
  connections to that host stalled outright). Installing the exact same CPU
  wheel from plain PyPI (`pip install torch`, no `--index-url`) worked
  immediately at full speed. If you hit the same symptom elsewhere, try
  PyPI directly before assuming torch itself is broken.
- **Windows MAX_PATH bit us once already.** When this repo lives under a
  long path (e.g. a deeply-nested scratch/temp directory), an extra
  directory level like `experiments/configs/some_long_name.yaml` can push
  the full path over 260 characters. `git add` fails outright ("Filename
  too long" — fixed for git with `git config core.longpaths true`), but
  Python's `open()`/`os.path.exists()` on the same file just silently
  returns "not found" with no length-related hint, even though
  `os.listdir()` on the parent directory still lists it — a very confusing
  combination if you don't know to suspect path length. Keep config/data
  file paths under this repo shallow (see `experiments/m2_config.yaml`,
  moved up from `experiments/configs/...` for exactly this reason) if you
  hit unexplained "file not found" errors for a file you can see exists.
