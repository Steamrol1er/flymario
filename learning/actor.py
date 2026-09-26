"""Actor-critic layer on top of the mushroom body.

** This layer is NOT a faithful model of fly biology. ** It's the
engineering bridge needed because the mushroom body's dopamine learning rule
(`learning/dopamine.py`) only ever learns "is this situation good or bad" —
that's what the real circuit does, and that's all it does. Mario needs
"which of ~7 buttons should I press right now", i.e. a sequential policy,
which nothing in the connectome is known to compute end-to-end.

What we borrow from biology and what we bolt on:

  BIOLOGICAL:
    - MBON approach/avoid balance as a value function V(s)
      (`MushroomBodyLearner.value`, unchanged, real learning rule)
    - Descending neurons (DNs) as the actual motor output channels
      (`interface/motor.py`)
    - PAM (reward) / PPL1 (punishment) as the two dopamine channels, and the
      real MBON->DAN feedback loops that exist in the connectome are the
      *inspiration* for using a reward-prediction-error signal at all
      (Eschbach et al. 2020 model this feedback explicitly)

  NOT BIOLOGICAL (clearly marked so a reader can tell what's fly and what's
  us):
    - Computing a TD/reward-prediction-error delta = r + gamma*V(s') - V(s)
      and using its sign/magnitude to decide how much dopamine to deliver.
      Real DANs are driven by their own upstream circuits (some of which do
      carry expectation-related signals, but not a clean scalar TD error).
    - A learnable KC/MBON -> descending-neuron "actor" weight matrix, updated
      with the same three-factor depression rule but on a different, added
      set of synapses that don't exist in the real connectome. We're reusing
      the *rule*, not the *wiring*, for this part.
    - Treating the rest of the brain (everything upstream of the MB and
      downstream of the DNs) as a fixed, non-plastic feature extractor /
      motor decoder for this project. That's a simplification of what's
      surely also somewhat plastic in the real animal, done purely to keep
      the number of trainable parameters small enough to learn from a few
      thousand Mario episodes.
"""
from __future__ import annotations

import dataclasses

import torch

from learning.dopamine import MushroomBodyLearner, DopamineParams


@dataclasses.dataclass
class ActorCriticParams:
    gamma: float = 0.9          # discount factor
    eta_actor: float = 0.01     # actor learning rate
    trace_decay: float = 0.9    # actor eligibility trace decay (~0.5-2s of decisions)
    da_gain: float = 1.0        # scales |delta| -> dopamine drive
    w_min: float = -1.0
    w_max: float = 1.0
    # Standard RL fix (running advantage normalization), opt-in and off by
    # default so it doesn't change behavior for scripts already tuned
    # without it (experiments/train_platformer.py). Added after diagnosing
    # a real failure mode on FlappyBird: a reward that's persistently
    # negative for structural reasons (not because of the action taken)
    # gives a delta with a near-constant sign, which saturates actor_w to
    # its clamp bounds within the first few dozen episodes regardless of
    # eta_actor, freezing the policy before it can differentiate between
    # actions (confirmed by direct inspection, see docs/STATUS.md). Dividing
    # the delta used for the ACTOR update (not the dopamine delivered to the
    # critic, which stays as-is) by a running estimate of its own std
    # removes that structural bias without touching the critic's learning.
    normalize_advantage: bool = False
    advantage_ema: float = 0.99  # decay for the running mean/var estimate


class ActorCritic:
    """Wraps a MushroomBodyLearner (critic) with a learnable KC -> action
    readout (actor), trained by TD(0)-style reward prediction error mapped
    onto PAM/PPL1-like dopamine channels.
    """

    def __init__(
        self,
        n_kc: int,
        n_actions: int,
        compartments: dict[str, int],
        dopamine_params: DopamineParams | None = None,
        actor_params: ActorCriticParams | None = None,
        device: str | torch.device = "cpu",
    ):
        self.device = torch.device(device)
        self.critic = MushroomBodyLearner(n_kc, compartments, dopamine_params, device)
        self.params = actor_params or ActorCriticParams()
        self.n_actions = n_actions

        # actor weights: KC -> action logit, one column per action
        self.actor_w = torch.zeros(n_kc, n_actions, device=self.device)
        self.actor_eligibility = torch.zeros(n_kc, n_actions, device=self.device)

        self._prev_value: float | None = None
        self._last_kc: torch.Tensor | None = None
        self._last_action: int | None = None
        self._delta_mean = 0.0
        self._delta_var = 1.0

    def act(self, kc_spikes: torch.Tensor, epsilon: float = 0.1) -> tuple[int, torch.Tensor]:
        """Choose an action from KC activity via softmax over actor logits,
        with epsilon-greedy exploration. Returns (action_index, probs)."""
        self.critic.step(kc_spikes)  # update critic eligibility traces too

        logits = kc_spikes @ self.actor_w  # [n_actions]
        probs = torch.softmax(logits, dim=0)
        if torch.rand(1).item() < epsilon:
            action = torch.randint(0, self.n_actions, (1,)).item()
        else:
            action = int(torch.multinomial(probs, 1).item())

        # actor eligibility: credit this (kc, action) pair, decay others
        self.actor_eligibility.mul_(self.params.trace_decay)
        self.actor_eligibility[:, action] += kc_spikes

        self._last_kc = kc_spikes
        self._last_action = action
        return action, probs

    def learn(self, reward: float, kc_spikes_next: torch.Tensor, done: bool = False) -> float:
        """Call after observing `reward` and the next state's KC activity.
        Computes TD error, delivers it as dopamine to the critic (PAM if
        positive, PPL1 if negative) and updates the actor. Returns delta for
        logging."""
        p = self.params
        v_next = 0.0 if done else float(self.critic.value(kc_spikes_next))
        v_curr = float(self.critic.value(self._last_kc)) if self._last_kc is not None else 0.0
        delta = reward + p.gamma * v_next - v_curr

        da = abs(delta) * p.da_gain
        if delta >= 0:
            dopamine = {name: (da if val > 0 else 0.0) for name, val in self.critic.compartments.items()}
        else:
            dopamine = {name: (da if val < 0 else 0.0) for name, val in self.critic.compartments.items()}
        self.critic.apply_dopamine(dopamine)

        # actor update: nudge chosen action's weights by delta * eligibility.
        # Optionally normalize delta by a running estimate of its own scale
        # first (see ActorCriticParams.normalize_advantage's docstring).
        actor_delta = delta
        if p.normalize_advantage:
            self._delta_mean = p.advantage_ema * self._delta_mean + (1 - p.advantage_ema) * delta
            self._delta_var = p.advantage_ema * self._delta_var + (1 - p.advantage_ema) * (delta - self._delta_mean) ** 2
            actor_delta = (delta - self._delta_mean) / (self._delta_var ** 0.5 + 1e-6)

        self.actor_w += p.eta_actor * actor_delta * self.actor_eligibility
        self.actor_w.clamp_(p.w_min, p.w_max)

        if done:
            self.critic.reset_eligibility()
            self.actor_eligibility.zero_()
            self._prev_value = None
        return delta

    def state_dict(self) -> dict:
        """Learned weights only (not eligibility traces or running stats,
        which are meant to reset between training runs) -- for saving a
        checkpoint to warm-start a live session from prior training
        (e.g. live/server.py loading experiments/train_pong.py's result)."""
        return {
            "actor_w": self.actor_w.clone(),
            "critic_weights": {k: v.clone() for k, v in self.critic.weights.items()},
        }

    def load_state_dict(self, state: dict) -> None:
        self.actor_w.copy_(state["actor_w"])
        for k, v in state["critic_weights"].items():
            self.critic.weights[k].copy_(v)
