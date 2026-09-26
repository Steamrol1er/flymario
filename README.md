# FlyMario

Use the *Drosophila* whole-brain connectome (FlyWire) as a spiking neural
network, train it with a dopamine-based (three-factor) learning rule modeled
on the fly mushroom body, and use it to play Super Mario Bros.

This is a research/education project, not a competitive Mario bot. See
`docs/PLAN.md` for the full write-up of the approach, the biological
simplifications made, and the milestones. Short version:

- The **mushroom body** (Kenyon cells → MBONs, gated by dopamine neurons)
  is the one part of the fly brain with a well-established, experimentally
  grounded learning rule. We use it, faithfully, as a value/critic circuit.
- Turning "how good is this situation" into "which button to press" is
  **not** something the fly connectome does on its own — we add a small
  actor layer on top and say so explicitly (see `learning/actor.py`).
- Everything else (photoreceptors → optic lobe → descending neurons) is
  used as a fixed, non-plastic feature extractor / motor decoder, in line
  with what's actually known about those circuits.

## Project layout

```
flymario/
  data/            connectome files (downloaded, gitignored)
  connectome/      download + load connectivity into a sparse weight matrix
  sim/             leaky integrate-and-fire spiking simulator (torch)
  interface/       game screen/state -> sensory input, DN spikes -> buttons
  learning/        dopamine three-factor rule (mushroom body) + actor-critic
  env/             Super Mario Bros gym wrapper + reward shaping
  experiments/     configs and training entry points
  tests/           unit tests, including the toy-conditioning sanity check
```

## Setup

```bash
pip install -r requirements.txt
```

`torch` is CPU-only by default (see requirements.txt); swap in a CUDA build
if you have a GPU. `gym-super-mario-bros`/`nes-py` need a C/C++ toolchain to
build on some platforms (notably Windows) — if that install fails, the rest
of the pipeline (connectome, simulator, learning rule, toy-conditioning
tests) still works and can be validated independently. See
"Mario environment install notes" below.

## Getting the connectome

```bash
python -m connectome.download --dest data/
```

This fetches the pre-built connectivity table from Shiu et al. 2024's
whole-brain model (`philshiu/Drosophila_brain_model`), which is much smaller
and easier to work with than raw FlyWire exports and has already been
validated against real experiments. Raw FlyWire Codex downloads
(`neurons.csv`, `connections.csv`, `classification.csv`) are also supported
— see `connectome/download.py --source flywire`.

## Quickstart: reproduce the sugar -> feeding sanity check

Before touching Mario at all, confirm the simulator + connectome are wired
up correctly:

```bash
python -m tests.check_sugar_feeding
```

## Quickstart: toy conditioning (M1 milestone)

Validates the dopamine three-factor rule on a tiny two-odor circuit,
independent of the full connectome and of Mario:

```bash
python -m tests.toy_conditioning
```

## Visualizing brain activity

To see which real neurons actually fire and watch activity spread from the
sugar-sensing neurons to the feeding motor neuron MN9 (the same experiment
`tests/check_sugar_feeding.py` validates, run on a 2-hop subcircuit so it's
fast):

```bash
python -m analysis.record_sugar_activity   # simulates, writes analysis/sugar_activity.npz
python -m analysis.build_viz_data          # writes analysis/brain_activity.json
```

Then open `viz/index.html` (published as a Claude Artifact alongside
`analysis/brain_activity.json`) — it plays back the real 300ms simulated
response as an animated graph: node size/glow = spike activity, color =
sugar-sensing / relay / MN9, with a scrubber to step through time bin by
bin and click-to-inspect for each neuron's real FlyWire root ID.

There's also a 3D view, `viz/brain3d.html`: a real public 3D reconstruction
of the adult fly brain (VirtualFlyBrain's JFRC2 template domain meshes, CC
BY 4.0), with the **gnathal ganglion (GNG)** — the modern name for the
subesophageal zone, where gustatory/sugar afferents arrive and where
feeding motor neurons like MN9 sit — highlighted and pulsing in sync with
the real per-bin spike count from the same simulation. It needs two more
downloads first (done once, saved under `data/brain_mesh/`, see
`analysis/build_brain_mesh.py`'s docstring for the exact source URLs) and
then:

```bash
python -m analysis.build_brain_mesh          # writes analysis/brain_mesh.json
python -m analysis.build_activity_timeline   # writes analysis/activity_timeline.json
```

This view shows a real anatomical region, not real per-neuron 3D positions
— FlyWire's own neuron coordinates need a Codex account this project
doesn't have yet. See `docs/STATUS.md`.

## Watch the agent play

Since the real Mario environment can't run here (see below), there's a
self-contained demo that trains the real dopamine three-factor rule and
TD actor-critic on an original small platformer and replays the result:

```bash
python -m experiments.train_platformer   # writes analysis/platformer_run.json
```

Then open `viz/platformer_play.html` (published as a Claude Artifact
alongside `analysis/platformer_run.json`) — pick between the agent's first
episode, its best run (a real, recorded win), and a typical run late in
training, and watch it play back frame by frame with a live readout of
value, dopamine, and active Kenyon cells. Current result: 57 real wins out
of 2500 training episodes (2.3%) — genuine but partial learning, not yet a
reliably solved level. See `docs/STATUS.md` for what that partial result
does and doesn't mean.

## FlappyBird: an honest negative result

A second, harder game, on the same real learning pipeline:

```bash
python -m experiments.train_flappybird   # writes analysis/flappybird_run.json
```

Then open `viz/flappybird_play.html` (published as a Claude Artifact
alongside `analysis/flappybird_run.json`). Unlike the platformer, this
**did not work**: the agent passed zero pipes in 2000 episodes. A real bug
(unnormalized Kenyon-cell activity destabilizing the value function and
saturating the actor's weights within ~50 episodes) was found and fixed
along the way, but the task still didn't converge in the time budgeted —
a genuine finding about the limits of this simple linear actor-critic on a
continuous-tracking task, not a discrete one-shot decision like the
platformer's gap jump. A hand-coded reference policy is included in the
same visualization to show the game itself is solvable. See
`docs/STATUS.md` for the full account.

## Pong: real Atari, native interface, live

The real game, downloaded for real, played through its own unmodified
interface — no hand-crafted features, no remapped actions, no reshaped
reward — watched live, not replayed from a precomputed file. Full account
(research, bugs found and fixed, honest result) in **`docs/PONG.md`**.

```bash
python -m connectome.download --source brain-mesh --dest data/   # one-time
python live/build_assets.py                                       # one-time
python run_live_demo.py
```

Opens three browser tabs against one local server that's actually running
the real emulator and the real agent right now:

- **Game** — a smooth vector recreation of the classic Pong screen (RESET /
  MAIN MENU, dashed centre line, the works), drawn from the real RAM-derived
  paddle/ball positions with a 0.1x–4x speed control and pause (button or
  the `P` key) so you can actually watch the ball and the fly's reactions.
- **Fly** — a hand-built anatomical fly model (translucent "x-ray" style)
  with a leg pressing the real up/down control, plus its live value
  estimate, dopamine signal, exploration rate, and running score rate.
- **Brain** — a REAL 3D reconstruction of the fly brain (rotate/zoom), with
  the right optic lobe (medulla + lobula — the actual visual
  motion-detection circuitry) highlighted and pulsing with live activity.

`game.html`/`fly.html` are fully offline-capable (no CDN dependencies);
`brain.html` loads three.js from a CDN for its 3D view, so needs internet
on first load. Stop the demo with Ctrl+C in the terminal running it.

`difficulty=0` (kept explicit in `env/pong_env.py`, not just the default)
is empirically the hardest of Pong's 4 built-in settings — a random policy
scores 0/15 against it vs. up to 13% at the easiest setting — the closest
thing to "the perfect bot" this game exposes.

For a one-shot training run instead of the live demo:
`python -m experiments.train_pong` (writes `analysis/pong_run.json` and a
checkpoint). Result from one such run: 18/400 points scored (4.5%),
matching a measured random-policy baseline (~5%) — not skillful play. Real
run-to-run variance observed: some runs stay dynamic throughout, others
collapse to a frozen policy partway through, even with identical code and
hyperparameters — see `docs/PONG.md` for the full, honest account.

## Other game environments

Real Mario needs `nes-py`, which needs a C/C++ compiler this environment
doesn't have (see below). Verified working here without any compiler:

- **Atari, via `ale-py`** — 104 real games (Pong, Breakout, Freeway, ...).
  `pip install ale-py "autorom[accept-rom-license]"`, then
  `AutoROM --accept-license`. If that fails with a network error even
  though your network is fine, see the note in `requirements.txt` — it's a
  `requests`-library quirk in some sandboxes, not a real connectivity
  problem; fetching the same URL with `urllib` works.
- **`flappy-bird-gymnasium`** — pure Python, no ROM or compiler at all.
- **`minigrid`**, **`gymnasium[box2d]`** (LunarLander, BipedalWalker) — also
  verified installable, not yet wired into this project's pipeline.

`stable-retro`/`gym-retro` and `procgen` are NOT installable via pip here
(no prebuilt wheels found) — don't spend time on those without a from-
source build.

## Running Mario training

```bash
python -m experiments.train --config experiments/m2_config.yaml
```

## Mario environment install notes

`nes-py` compiles a C++ extension. On Windows you need the "Desktop
development with C++" workload from the Visual Studio Build Tools
installed, or run everything under WSL2 (recommended in the plan). If you
can't get it installed:

- `env/mario_env.py` will raise a clear `ImportError` naming the missing
  package instead of failing silently elsewhere.
- Everything upstream of the environment (connectome, `sim/`, `learning/`)
  has its own tests that don't depend on `gym-super-mario-bros` at all, so
  development can continue while that install is sorted out.

## Status

See `docs/STATUS.md` for what's implemented vs. still to do against the
milestones in `docs/PLAN.md`.
