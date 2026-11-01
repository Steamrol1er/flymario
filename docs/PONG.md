# Fly Brain Plays Pong

This is a standalone account of the Pong experiment: why Pong, how the
game was obtained, what was researched before building anything, what
interface is actually used, what problems came up, and what the real
result is. Code lives in `env/pong_env.py`, `experiments/train_pong.py`,
`viz/pong_play.html`.

> **Corrections and headline result (added after the final evaluations).**
> 1. **The Pong agents do not use real FlyWire data.** Both
>    `experiments/train_pong.py` and `experiments/train_pong_hybrid.py` build
>    their "brain" from `load_toy_connectome()`: a synthetic 242-neuron
>    circuit (200 Kenyon-cell-like units, 2 output units, 40 odor units),
>    chosen for speed. The real 138,639-neuron connectome is used only in the
>    sugar-feeding validation and the Fly Marksman experiment
>    (`docs/MARKSMAN.md`). Read the Pong "fly brain" as a random sparse
>    projection plus a learning rule, not as fly anatomy.
> 2. **Neither agent beats random play.** Evaluated with learning frozen
>    over 1,000 points per seed, uniform-random play scores **3.1 +/- 0.6%**
>    (10 seeds). The dopamine agent sampling from its own policy scores
>    2.8 +/- 1.0% (10 seeds) and the hybrid agent 2.8 +/- 3.2% (6 seeds):
>    neither differs significantly from random. The dopamine agent's greedy
>    policy scores exactly 0.0% on all 10 seeds. See "Evaluation-only
>    results" below.

## Why Pong

Pong is a real-time motion-tracking task: track a moving object, move a
paddle to intercept it. That's close to what real fly visual neurons
(lobula plate tangential cells — the same optomotor-following circuitry a
fly uses to stabilize its own flight) actually compute. It's also one of
the best-studied benchmarks in reinforcement learning, so there's a lot to
compare against.

## Downloading the actual game

Real Mario needs `nes-py`, which needs a C/C++ compiler this environment
doesn't have, and there's no legal ROM to load anyway. Atari 2600 games
don't have that problem:

```bash
pip install ale-py "autorom[accept-rom-license]"
AutoROM --accept-license
```

This downloads 104 real, playable Atari ROMs (including `pong.bin`) under
AutoROM's standard license agreement — not a copyright grey area, the
established path the RL research community actually uses.

**Problem hit:** `AutoROM --accept-license` failed with "Network connection
error" in this sandbox, even though the network was fine. Tracing
AutoROM's source showed it fetches
`https://gist.githubusercontent.com/jjshoots/61b22aefce4456920ba99f2c36906eda/raw/.../Roms.tar.gz.b64`
using the `requests` library. Fetching the exact same URL with Python's
built-in `urllib` worked immediately. **Fix:** downloaded the archive with
`urllib`, base64-decoded it, and extracted the tar.gz into AutoROM's
`roms/` directory by hand. Whatever the root cause (`requests`'s bundled
CA bundle vs. the system one is a common culprit for this exact symptom),
this is a real, reproducible workaround worth knowing about, not a Pong-
specific issue.

**Verification, not assumption:** confirmed `ALE/Pong-v5` actually runs
(`gym.make(...)`, `env.reset()`, `env.step(...)`, real reward returned)
before building anything on top of it.

## Research done before writing code

Three searches, summarized here with sources so the claims are checkable:

1. **"Spiking neural network reinforcement learning Atari Pong dopamine
   STDP"** — confirms reward-modulated STDP with eligibility traces
   ("when a postsynaptic spike occurred within a few ms of a presynaptic
   spike, the connection became eligible ... tagged with an exponentially
   decaying trace; when reward arrived, the tagged weight was increased or
   decreased") is an established, real technique, and that spiking
   networks converted from RL-trained ANNs can match ANN performance on
   Atari. A dedicated racket-ball game closely modeled on Pong has been
   used specifically to test STDP-RL rules.
   [arxiv.org/pdf/2008.13044](https://arxiv.org/pdf/2008.13044),
   [journals.plos.org/plosone (racket-ball game)](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0265808)

2. **"Insect brain connectome reinforcement learning play video game
   fly"** — surfaced very recent (Sept 2026) projects using Google/Janelia's
   MaleCNS v1.0 connectome (166,000+ neurons) to play Doom, Super Mario 64,
   Minecraft, and Beat Saber. Critically: **"the simulated fly isn't
   learning or making decisions; a distinct machine learning model is
   performing these tasks"** — those demos map game frames onto sensory
   neurons and pipe reward events into dopamine cells (PPL101) for
   illustration, but a *separate* conventional model actually plays. This
   project attempts something more ambitious and harder: having the
   dopamine-modulated circuit's own output *be* the policy, not just react
   to one. That directly explains why this is genuinely difficult, rather
   than something a "normal" implementation would get right easily.
   [Hackaday](https://hackaday.com/2026/09/14/fly-brain-connectome-used-to-trade-stocks-and-play-games/),
   [Tom's Hardware](https://www.tomshardware.com/software/programming/google-maps-entire-brain-and-central-nervous-system-of-adult-male-fruit-fly-software-engineers-immediately-make-it-run-doom-ai-powered-3d-model-of-over-166-000-neurons-can-also-play-super-mario-64)

3. **"Actor-critic policy gradient weight saturation fix eligibility trace
   REINFORCE baseline"** — confirmed that subtracting/normalizing a
   baseline from the return "leaves the expected value of the gradient
   unchanged while having a significant effect on its variance" — the
   standard, textbook fix for exactly the instability found on FlappyBird
   (see below).
   [Lil'Log: Policy Gradient Algorithms](https://lilianweng.github.io/posts/2018-04-08-policy-gradient/)

4. Directly checked the honeybee SNN reinforcement-learning model (PMC4699142) for
   architecture details: confirms a three-factor rule (pre-synaptic ×
   post-synaptic × a global dopaminergic/octopaminergic signal) with an
   eligibility trace bridging the delay between stimulus and reward — the
   same structure `learning/dopamine.py` and `learning/actor.py` already
   use.
   [PMC4699142](https://pmc.ncbi.nlm.nih.gov/articles/PMC4699142/)

**Conclusion drawn from the research**: the architecture (three-factor
dopamine rule + eligibility trace + a TD-error-driven actor) is the right
family of approach — it matches real published biological RL models. The
problem on FlappyBird wasn't the *architecture family*, it was two
specific, fixable bugs (below), applied here from the start.

## Interface: exactly the game's own, nothing custom

Unlike Mario (`interface/sensory.py`'s hand-built feature extractor) or the
platformer, Pong's interface is used completely unmodified:

| | Used here |
|---|---|
| Observation | The raw 128-byte Atari RAM, `gym.make("ALE/Pong-v5", obs_type="ram")`. Fed to the Kenyon-cell projection as-is (just scaled to [0,1]), no derived features. |
| Actions | The game's own `Discrete(6)`: `NOOP, FIRE, RIGHT, LEFT, RIGHTFIRE, LEFTFIRE`. No remapping, no reduced action set. |
| Reward | The game's own `+1` / `-1` / `0`. No reshaping (contrast with FlappyBird, where reshaping the reward was itself part of the problem). |

Three RAM addresses (player paddle y, ball x, ball y, enemy paddle y) are
read out **only** for the replay visualization, found empirically by
moving the paddle and watching which bytes changed — not fed to the
network, which sees all 128 bytes uniformly, and not taken from a
published RAM map that might not match this exact `ale-py` build. See
`env/pong_env.py`'s docstring for exactly how each address was confirmed
(or, for the enemy paddle, only inferred by elimination — flagged as
such).

## Bugs found and fixed (applied proactively this time)

Both diagnosed on FlappyBird first (`docs/STATUS.md` has the full account
with the actual weight-inspection numbers) and carried over here before
training even started, rather than rediscovering them the hard way twice:

1. **Unbounded Kenyon-cell activation magnitude.** ~100 simultaneously
   active KCs with unbounded ReLU output made the mushroom body's linear
   value estimate (`learning/dopamine.py`'s `value()`, a plain weighted
   sum) swing by tens of units from a tiny weight asymmetry. Fixed with
   divisive normalization on the KC output (`RamToKC` in
   `experiments/train_pong.py`) — a real mechanism in actual neural
   circuits, not just a numerical hack.
2. **No baseline/advantage normalization on the actor update.** A
   persistently one-signed TD error (which FlappyBird's shaped reward
   produced structurally) saturates the actor's clamped weights within the
   first few dozen episodes regardless of learning rate, freezing the
   policy for the rest of training. Added `normalize_advantage` to
   `learning/actor.py::ActorCriticParams` (opt-in, default off, so it
   doesn't change already-tuned behavior elsewhere) — a running mean/std
   normalization of the delta used for the actor's own update, leaving the
   dopamine delivered to the critic untouched. This is the standard
   "subtract a baseline" fix confirmed in the research above.

Pong's own reward is already naturally close to zero-mean (equally often
+1 or -1 depending on who scores, 0 the rest of the time) — a better-
conditioned problem than FlappyBird's structurally-always-negative shaped
reward, independent of these two fixes.

## Result

### Short run: 400 episodes (100K steps)

One "episode" = one point exchange, ending when either side scores or
after 1200 steps. Real numbers from an early `analysis/pong_run.json`:

- **18 points scored out of 400 (4.5%)**, roughly matching a **uniform-
  random policy's own baseline (~5%, measured directly)** on this
  environment (the built-in CPU opponent is not perfect, so even random
  paddle movement scores sometimes).
- First-half vs. second-half score rate: 3.5% → 5.5% — a mild upward trend,
  but with only 18 total scoring events across the whole run this is not a
  statistically confident claim of learning.
- The policy did not freeze: steps-per-point and score outcomes kept
  varying across the *entire* 400-episode run rather than collapsing to an
  identical, repeated failure pattern within the first ~50 episodes.

### Long run: 2,000,000 steps (`--target-steps 2000000`)

7,402 episodes, 629 seconds (10.5 minutes) wall clock, 3,179 steps/sec.
This is a genuinely different, and more nuanced, result than the short
run -- worth reporting exactly as it happened rather than only the
flattering part:

- **The policy froze for roughly the first 22% of training** (episodes
  0-1600, ~400K steps): `avg_steps` pinned at exactly 256.0 and
  `score_rate` at 0.0% for a long stretch — the same degenerate pattern
  first diagnosed on FlappyBird (docs/STATUS.md), recurring here at a
  larger scale despite both fixes being in place from the start. The
  advantage-normalization and KC-normalization fixes stop weights from
  saturating in a handful of episodes; they don't guarantee the policy
  never gets stuck in a bad basin for a while.
- **It then broke out** and started scoring: score rate fluctuated mostly
  in the 1-6% range in the back half of training (e.g. 6.0% at episode
  4194), with `avg_steps` climbing into the 265-290 range (versus the
  frozen phase's flat 256), i.e. genuinely different, more varied behavior
  from episode to episode.
- **Overall: 140 points scored out of 7,402 episodes (1.89%)** — including
  the frozen phase, this is *below* both the short run's 4.5% and the
  measured ~5% random baseline. The back half alone (episodes 1600-7402)
  scores at 2.41% — better than the frozen phase, still below random
  chance.

**Read this as**: training 20x longer did not produce a better result on
this metric, and in one concrete way (the extended frozen phase) produced
a worse one. That's a genuine, informative finding, not a disappointing
footnote to hide: more data does not fix a credit-assignment problem in
this simple linear, non-recurrent actor-critic the way it does for a
proper deep RL setup with a replay buffer and gradient-based updates.
That's consistent with the research finding earlier in this file — even
the well-known "fly brain plays Doom/Mario 64" demos don't have the fly's
own circuit produce the policy; they use a separate model for that and
route reward events through dopamine cells for illustration. Making the
dopamine-modulated circuit's output *be* the policy, well enough to
reliably win at a real-time tracking task, is closer to an open problem
than a solved one, and this project's simple actor-critic hits that limit
regardless of how many episodes it's given.

### Frame stacking: a real, if modest, improvement

A single RAM frame gives the network ball/paddle *positions* only, with no
way to tell which direction the ball is moving -- a genuine information
deficiency for a motion-tracking task, not a tuning problem. Standard
Atari RL (DQN included) always stacks multiple consecutive frames for
exactly this reason. `experiments/train_pong.py::FrameStack` and
`live/server.py`'s copy of it now concatenate the last 2 raw RAM frames
before they reach the KC projection -- still literally the game's own raw
RAM, unmodified, just more of it (the same principle as DQN's own frame
stack), not the same thing as hand-picking which of the 128 bytes matter.

Re-ran the identical 2,000,000-step comparison with this change:

| | Without frame stack | With frame stack |
|---|---|---|
| Overall score rate | 1.89% (140/7,402) | **2.56% (188/7,339)** |
| First ~1,500 episodes | 0% exactly (fully frozen) | 1.13% (slow, but never pinned at exactly 0%) |
| Back half | 2.41% | **2.93%** |

A real, measurable improvement, and the early freeze was less complete
(never stuck at literally 0% the way the unstacked run was) -- consistent
with the network now having a genuinely useful signal it lacked before,
not just noise. **Still below the ~5% random-policy baseline overall.**
Giving the network velocity information helped; it did not resolve the
underlying credit-assignment limitation described above.

### Proper policy gradient: tried, made things worse, reverted

The original actor update (`actor_w += eta * delta * eligibility`, with
`eligibility[:, action] += kc_spikes`) only ever credited the KC activity
present when an action was taken to that action's own column -- it never
adjusted the other actions' logits the way a real softmax policy gradient
does. The textbook-correct eligibility-trace actor-critic update (Sutton &
Barto, *Reinforcement Learning*, 2nd ed., section 13.5) uses the softmax's
actual score function, `grad_logits = one_hot(action) - probs`, so
reinforcing one action also proportionally discourages the others.
Implemented as `learning/actor.py::ActorCriticParams.policy_gradient`
(opt-in, default off).

A short diagnostic (500 episodes, batches of 50) looked genuinely
healthier: `actor_w` grew smoothly and stayed well inside its clamp bounds
(max magnitude ~1.4 against a ±3 limit) instead of the old rule's
near-instant saturation. **That result did not hold up at full scale.**
The matched 2,000,000-step run: 24.0 minutes (vs. 6.7 for the
frame-stacking-only run -- the outer-product update costs more per step),
and it froze almost immediately (by episode ~250) and *never recovered*:

| | Frame stacking only | + proper policy gradient |
|---|---|---|
| Overall score rate | **2.56% (188/7,339)** | 0.013% (1/7,806) |
| Frozen-checkpoint fraction | ~1 in 5 of progress checkpoints | 187 of 194 (96%) |
| Wall clock (2M steps) | 6.7 min | 24.0 min |

**Reverted** (`policy_gradient=True` removed from both
`experiments/train_pong.py` and `live/server.py`; the code stays in
`learning/actor.py`, opt-in and off, since it's a real, documented,
tested-and-rejected result worth keeping rather than a live option to
leave on). Read as: a more theoretically correct update rule is not
automatically a better one for this specific (small, linear, online,
eligibility-trace-based) setup -- the added coupling between actions'
logits, which is exactly what makes it "correct," appears to have made it
easier to lock into a self-reinforcing bad policy here rather than harder.
Whether that's specific to this architecture, this environment, or these
hyperparameters isn't established by one run; it's reported as what
actually happened, not diagnosed further.

### Multi-seed evaluation: the earlier single-run comparisons were noisier than they looked

Every result up to this point (the original run, frame stacking, the
policy-gradient attempt) was **one run each**. `experiments/train_pong.py`
now takes `--seed` (controls torch/numpy RNG, the environment's own RNG
stream, and the KC projection matrix, so a seed is fully reproducible) and
`--out` (a distinct results path per seed), specifically to stop drawing
conclusions from single, possibly-lucky-or-unlucky runs.

Also raised the exploration floor from ε=0.05 to ε=0.15 (decay stretched
so it's reached around 75% through training instead of ~58%) at the same
time, since every freeze observed so far happened *while* ε was still
fairly high, then got locked in as ε kept dropping and closed off the
route back out.

Three seeds, 700,000 steps each (a shorter budget than the 2,000,000-step
runs above, chosen to fit three runs in one session):

| Seed | Episodes | Scored | Rate |
|---|---|---|---|
| 0 | 2,670 | 25 | 0.94% |
| 1 | 2,679 | 13 | 0.49% |
| 2 | 2,514 | 82 | **3.26%** |

**A 6.7x range (0.49%-3.26%) from the same code, same step budget, only
the random seed different.** This is the real headline finding, and it
changes how much weight the earlier single-run comparisons deserve:

- The **frame-stacking result** (2.56% vs. 1.89%, one run each, a 0.67
  percentage-point gap) is well within the variance just measured
  directly. It should be read as "not established either way" now, not
  "confirmed" — the earlier write-up overstated it by treating a
  single-run difference as decisive. Frame stacking is still kept (it's
  well-justified on independent grounds -- a real information gap, not
  just a score that happened to be higher this once), but the specific
  score improvement claimed for it doesn't hold up to this scrutiny.
- The **policy-gradient result** (0.013%, 1/7,806) is a different story:
  it sits far below even the *worst* seed in this natural variance band
  (0.49%). A gap that large relative to the measured noise floor is not
  explained by seed variance alone, so that revert decision stands.
- (One seed -- 0 -- took 163 minutes instead of the expected ~8, almost
  certainly a stray CPU-contending process from an earlier background job
  rather than anything algorithmic; the wall-clock number for that run is
  an artifact, its score-rate result is not.)

**Read this as**: this project's own earlier reporting wasn't rigorous
enough on the smaller effect sizes, and this section says so plainly
rather than quietly using a more careful method going forward without
flagging that the past comparisons need to be discounted. Any future
claim about a modest-sized improvement needs multiple seeds before it's
trustworthy; large, qualitative differences (frozen vs. not, 0.013% vs.
2-3%) don't.

## Final evaluation: dopamine vs. hybrid vs. random (3 seeds each)

*Training-time score rates (include exploration). The random row here is an
early 3-seed estimate (3.7%); the 10-seed frozen evaluation below gives 3.1%
and is the figure to use.*

Score rate = fraction of episodes (points) won, the same metric throughout.
Dopamine rows: 700K steps, trace_decay 0.9. Hybrid rows: 2M steps,
`experiments/train_pong_hybrid.py`, linear Q-head on the same fixed encoding.
Random: uniform random actions, 1,000 episodes per seed.

| Policy | Seed 0 | Seed 1 | Seed 2 | Mean | Last 10% of episodes (mean) |
|---|---|---|---|---|---|
| Random play | 2.8% | 5.0% | 3.3% | **3.7%** | n/a |
| Dopamine, trace 0.7 | 0.94% | 0.49% | 3.26% | 1.56% | 1.0% |
| Dopamine, trace 0.9 | 3.02% | 1.96% | 2.21% | 2.40% | 2.6% |
| Hybrid Q-head | 1.79% | 3.91% | 1.38% | 2.36% | 1.3% |

Per-seed last-10% rates: dopamine 0.9 = 2.3 / 1.9 / 3.5%; hybrid =
0.4 / 2.9 / 0.4%.

**What this shows, honestly**

- **Neither agent learned to beat random play.** Random scores 3.7%; the
  best seed-mean for a trained agent is 2.4%. With three seeds per arm none
  of these differences is statistically meaningful, but nothing here
  supports "the agent learned Pong". The earlier "2-3%" figures were
  presented without this baseline, which was a gap in the earlier reporting.
- **trace_decay 0.9 vs. 0.7:** the three-seed range tightened (0.49-3.26%
  became 1.96-3.02%) and the mean rose, which matches the reasoning that a
  3-step eligibility window is too short for a ~270-step episode with only a
  terminal reward. With n=3 this is suggestive, not established.
- **Hybrid vs. dopamine:** equal on the whole-run mean (2.36% vs. 2.40%).
  The hybrid is *less stable*: two of three seeds scored 3-4% early and
  decayed to about 0.4% by the end (seed 0: 4.2% in the first tenth of
  training, 0.4% in the last; seed 2: 3.2% to 0.4%), while the dopamine
  agent held steady or improved. The early scores occur while epsilon is
  near 1 (mostly random actions), so they mostly reflect random play, and
  the decline looks like the Q-head degrading as exploration drops.
- **So the training method is probably not the bottleneck.** Swapping the
  dopamine rule for backprop with replay and a target network did not help.
  The more likely limits are the features (a fixed random projection of
  raw RAM bytes, where the decision depends on a relation such as ball
  height minus paddle height) and the single linear readout. Untested next
  steps: a deeper Q-head, an encoder that preserves that relation, and
  real mushroom-body structure from FlyWire in place of the toy circuit.
- **Caveats:** three seeds, one hyperparameter setting per arm, no tuning of
  the hybrid. Absence of a gain here does not show the approach cannot work.

## Evaluation-only results (learning frozen, no exploration)

The training-time score rates above mix exploration with learning, so the
final agents were also evaluated frozen (`experiments/eval_pong.py`): 1,000
points per seed, no weight updates. Seeds used: 10 dopamine (trace 0.9, 700K
steps), 6 hybrid (2M steps; seeds 6 and 7 were started and deliberately
stopped to free CPU, so they are excluded), 10 random.

| Agent | Seeds | Mean +/- SD across seeds | Per-seed rates (%) |
|---|---|---|---|
| Random play | 10 | **3.10 +/- 0.58%** | 3.6, 3.2, 2.8, 3.2, 3.2, 2.8, 3.4, 2.3, 2.3, 4.2 |
| Dopamine, sampled from policy | 10 | 2.81 +/- 1.03% | 3.6, 0.8, 3.1, 3.4, 3.5, 1.1, 3.4, 3.2, 3.5, 2.5 |
| Dopamine, greedy (argmax) | 10 | 0.00 +/- 0.00% | all 0.0 |
| Hybrid Q-head, greedy | 6 | 2.83 +/- 3.23% | 0.0, 7.0, 0.0, 5.8, 4.2, 0.0 |

Tests against random (Welch t / Mann-Whitney):

- Dopamine sampled: t = -0.77, p = 0.45 (MWU 0.97). Not different from random.
- Hybrid greedy: t = -0.20, p = 0.85 (MWU 1.0). Not different from random.
- Dopamine greedy: t = -16.9, p = 4e-8. Significantly *worse* than random.

**Reading this honestly**

- **No arm beats random play.** The earlier training-time figures (2-3%)
  looked like learning only because no random baseline had been measured.
- **The greedy dopamine policy is degenerate.** Taking the single best action
  every step scores exactly zero on every seed, which suggests it settles
  on one action (a stuck paddle never intercepts). Its sampled version
  recovers random-level scores because the sampling restores movement.
  That means the "score" of the sampled policy is mostly the randomness
  itself.
- **The hybrid is bimodal, not better.** Three seeds exceed random (4.2,
  5.8, 7.0%) and three collapse to 0%. Across seeds this is statistically
  indistinguishable from random, and it matches the training logs, where two
  seeds decayed from 3-4% to about 0.4% as exploration dropped.
- **Likely cause: features and readout, not the training rule.** Swapping
  the dopamine rule for replay-based Q-learning changed variance, not mean
  performance. The fixed random projection of raw RAM bytes does not expose
  the relation the task needs (ball height minus paddle height), and a
  single linear layer cannot build it. Untested next steps: an encoder that
  preserves that relation, a deeper head, and real mushroom-body structure
  from FlyWire instead of the synthetic circuit.
- **Caveats:** a point counts as scored when the agent wins it; every arm uses
  the same metric and environment. Results cover one hyperparameter setting
  per arm and no tuning of the hybrid, so they do not show that these
  approaches cannot work, only that they did not here.

## Live local demo (not a precomputed replay)

The first version of this (`viz/pong_play.html`) was a Claude Artifact
that hand-drew an approximation of the game from RAM values and replayed
one precomputed episode. Two real problems with that: it wasn't the real
game's actual graphics, and it wasn't live.

`run_live_demo.py` (project root) fixes both. Run it from a terminal
(including a VS Code integrated terminal):

```bash
python run_live_demo.py
```

It starts `live/server.py` — a small local HTTP server (stdlib only, no
new dependencies) that runs the real game and the real agent continuously
in a background thread, point after point, learning as it goes — and
opens three browser tabs, each polling that one live process:

- **`live/game.html`** — the real emulator screen, live, via
  `<img src="/stream.mjpg">`. This went through three versions worth
  recording: (1) the actual RGB frame, but fetched fresh over HTTP as a
  polled PNG on every simulation step — this was the version with the
  visible stutter, because each poll is an independent request/response
  and image decode, so motion looks like a slideshow; (2) a hand-drawn
  vector recreation of the classic Pong screen (dashed centre line, RESET/
  MAIN MENU buttons) built from RAM-derived positions to sidestep the
  encoding cost — smoother, but it stopped being the real game's actual
  graphics, and the RESET/MAIN MENU buttons didn't do anything (there's
  nothing to reset to or navigate from in a live simulation), which is
  worse than just not having them; (3) **the real emulator frame again,
  but as a proper MJPEG video stream** (`multipart/x-mixed-replace`, the
  same mechanism old IP cameras use) — one long-lived connection over
  which the server pushes a new JPEG the moment it's ready, and the
  browser's own `<img>` tag repaints continuously with no polling and no
  per-frame request overhead. This is the version kept: real graphics,
  genuinely smooth, no decorative controls that don't do anything. A speed
  control (0.1x–4x) and pause (button or the `P` key) still let you slow
  the whole simulation down to actually watch the ball and the fly's
  reactions. `env/pong_env.py` defaults to `obs_type="ram"` (no per-step
  rendering cost) and only switches on rendering (`render_rgb=True`) for
  the live demo specifically, where the video stream needs it.
- **`live/fly.html`** — a hand-built SVG fly model (translucent blue
  "x-ray" wireframe: segmented abdomen, thorax, compound eye, antenna, one
  wing, six jointed legs in a standing pose), one front leg animating to
  press the up/down control matching the real action taken, and a cluster
  of glowing colored blobs over the head whose brightness tracks live
  Kenyon-cell activity — labeled explicitly as a schematic stand-in for
  "the brain is active," not a real 3D neuron placement (that's what
  `live/brain.html` is for). The model and its build reasoning are
  self-contained in the file; the design was checked against real
  *Drosophila* body-plan anatomy first (head/thorax/abdomen, one wing pair
  + halteres, three leg pairs) rather than drawn freehand.
- **`live/brain.html`** — a REAL 3D reconstruction of the adult fly brain
  (`live/build_assets.py` processes the same public VirtualFlyBrain JFRC2
  template meshes used in the earlier Claude-Artifact brain visualization),
  rendered in three.js, rotatable and zoomable, with the **right optic
  lobe (medulla + lobula)** highlighted and pulsing with live Kenyon-cell
  activity — not the gnathal ganglion used for the earlier sugar-feeding
  demo. That choice isn't arbitrary: for a real-time visual
  motion-tracking task, the medulla and lobula are where the fly's actual
  T4/T5 motion-detection circuits live (see the citations earlier in this
  file). This page needs one CDN script (three.js from cdnjs) and so needs
  internet on first load; `game.html` and `fly.html` stay fully
  offline-capable.

`live/assets/sample_frame.png` (a real captured emulator frame, not a
mockup) and `live/assets/brain_mesh.json` (the preprocessed 3D mesh data)
are the "necessary images/assets" kept alongside the code rather than
fetched at runtime.

Stop the demo with Ctrl+C in the terminal that's running
`run_live_demo.py` — closing the browser tabs alone leaves the server (and
the training loop) running.

Note on reproducibility: this reuses the same architecture as
`experiments/train_pong.py` but starts the agent fresh each time you run
it, rather than loading a saved checkpoint. That's deliberate — re-running
`experiments/train_pong.py` itself has produced both "stays varied and
occasionally scores" and "collapses to a frozen policy after ~50-80
episodes" outcomes on different runs with the *same* code and
hyperparameters (real, observed variance — not a claim that this is
solved), so a live session lets you watch that variability directly rather
than depending on one saved run being the lucky one. `learning/actor.py`'s
`ActorCritic.state_dict()`/`load_state_dict()` exist if you want to
warm-start from a specific `experiments/train_pong.py` run instead.

## What would likely help next

- **A proper policy-gradient implementation** (actual log-probability
  gradient of the softmax, not the current ad-hoc `eligibility * delta`
  rule) — closer to what the actor-critic literature above actually
  specifies.
- **A richer, less noisy KC encoding.** A fixed random projection of raw
  RAM bytes is a poor substitute for the real connectome's actual
  projection-neuron → Kenyon-cell wiring (still blocked on FlyWire Codex
  annotations — see `docs/STATUS.md`).
- **Frame stacking / a short RAM history** instead of a single-frame
  snapshot, so the network can see ball *velocity*, not just position —
  standard practice in Atari RL (DQN stacks 4 frames for exactly this
  reason) and something this implementation currently doesn't do at all.
- **Longer training.** 400 episodes is small by Atari RL standards (DQN-era
  papers train for millions of frames); this budget was chosen to fit a
  single interactive session, not because it's expected to be enough.
