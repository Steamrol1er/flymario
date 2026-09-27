"""A trained readout on top of the fixed, connectome-derived KC encoding --
the "hybrid" architecture from docs/PONG.md's research survey.

## How the brain is used here, precisely

The pipeline has two stages, and only one of them is "the fly":

  1. **The brain** (`RamToKC`/`FrameStack` in `experiments/train_pong.py`,
     `experiments/train_pong_hybrid.py`): raw game RAM -> a fixed, never-
     updated random projection onto `n_kc` units, ReLU'd and divisively
     normalized. This is the same computation the mushroom body's actual
     wiring performs on its sensory input -- each Kenyon cell samples a
     small, fixed set of upstream inputs, giving any given stimulus a
     sparse, roughly-orthogonal code. That's a real, structural property
     of the fly's mushroom body, independent of whether any plasticity
     happens downstream of it (Aso et al. 2014). The connectome's
     contribution stops here: it never changes during training.

  2. **QHead** (this file): a small trained network that maps that fixed
     KC code to action values. This is NOT biological -- it's trained by
     ordinary backpropagation against a replay buffer and a target
     network, the standard DQN recipe (Mnih et al. 2015), with nothing
     fly-inspired about the update rule itself.

Why build it this way instead of either a plain generic-input DQN or the
pure dopamine three-factor rule this project uses elsewhere:

  - A plain DQN over raw/hand-picked features would have NOTHING
    connectome-derived in it at all.
  - The pure dopamine rule (`learning/actor.py`, used in
    `experiments/train_pong.py`) tries to make BOTH stages biological --
    the encoding AND the learning rule -- and we now have direct evidence
    (docs/PONG.md's research section) that this specific combination is
    exactly the one that reliably fails in this niche: DOOMFLY's real
    166,700-neuron connectome with dopamine-gated KC->MBON plasticity and
    no trained readout reported "not demonstrated learned survival", while
    every project in the same survey that added a genuinely trained
    readout on a fixed connectome-derived encoding (Fly Dino, Fly
    Marksman, fly-swing) reported strong, unambiguous results.

This file keeps the connectome's real structural contribution (the fixed
sparse encoding) and swaps only the part the evidence says needs to be
conventional: the actual decision-making.
"""
from __future__ import annotations

import random
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F


class QHead(nn.Module):
    """A single linear layer, KC activity -> one Q-value per action.

    Kept linear (not a deeper MLP) so the comparison against the dopamine
    rule's linear actor isolates ONE variable -- the training method
    (backprop + replay vs. online three-factor eligibility) -- rather than
    also changing the function class. If this works better, a deeper head
    is a natural next step; if it doesn't, we'll know the training method
    wasn't the bottleneck.
    """
    def __init__(self, n_kc: int, n_actions: int):
        super().__init__()
        self.fc = nn.Linear(n_kc, n_actions)

    def forward(self, kc: torch.Tensor) -> torch.Tensor:
        return self.fc(kc)


class ReplayBuffer:
    """Fixed-capacity ring buffer of (kc, action, reward, kc_next, done)
    transitions, sampled uniformly at random. This is the piece that
    actually fixes the instability the pure online rule kept hitting:
    training on a random past batch instead of the single, highly
    correlated live transition breaks the correlation that made online
    updates so noisy (this is the textbook reason DQN needs a replay
    buffer at all, not an optional extra)."""

    def __init__(self, capacity: int = 20_000):
        self.buffer: deque = deque(maxlen=capacity)

    def push(self, kc, action, reward, kc_next, done):
        self.buffer.append((kc, action, reward, kc_next, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        kc, action, reward, kc_next, done = zip(*batch)
        return (
            torch.stack(kc),
            torch.tensor(action, dtype=torch.long),
            torch.tensor(reward, dtype=torch.float32),
            torch.stack(kc_next),
            torch.tensor(done, dtype=torch.float32),
        )

    def __len__(self):
        return len(self.buffer)


class HybridAgent:
    """Wraps QHead + a target network + a replay buffer into the same
    act()/observe() shape experiments/train_pong.py's ActorCritic uses, so
    the training loop around it stays easy to compare side by side."""

    def __init__(
        self,
        n_kc: int,
        n_actions: int,
        gamma: float = 0.97,
        lr: float = 1e-3,
        buffer_capacity: int = 20_000,
        batch_size: int = 64,
        target_sync_every: int = 500,
        min_buffer_before_train: int = 500,
        device: str | torch.device = "cpu",
    ):
        self.device = torch.device(device)
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_sync_every = target_sync_every
        self.min_buffer_before_train = min_buffer_before_train

        self.q = QHead(n_kc, n_actions).to(self.device)
        self.q_target = QHead(n_kc, n_actions).to(self.device)
        self.q_target.load_state_dict(self.q.state_dict())
        self.opt = torch.optim.Adam(self.q.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_capacity)

        self._train_steps = 0
        self._last_kc = None
        self._last_action = None

    def act(self, kc: torch.Tensor, epsilon: float = 0.1) -> tuple[int, None]:
        if random.random() < epsilon:
            action = random.randrange(self.n_actions)
        else:
            with torch.no_grad():
                q_values = self.q(kc.unsqueeze(0)).squeeze(0)
            action = int(torch.argmax(q_values).item())
        self._last_kc = kc
        self._last_action = action
        return action, None

    def learn(self, reward: float, kc_next: torch.Tensor, done: bool = False) -> float:
        """Push the transition, then (once there's enough data) take one
        batched gradient step. Returns the batch's mean |TD error| for
        logging, matching the dopamine ActorCritic.learn()'s `delta`
        return value so the two training scripts print comparable numbers."""
        self.buffer.push(self._last_kc, self._last_action, reward, kc_next, float(done))

        if len(self.buffer) < max(self.batch_size, self.min_buffer_before_train):
            return 0.0

        kc_b, action_b, reward_b, kc_next_b, done_b = self.buffer.sample(self.batch_size)
        kc_b, action_b, reward_b = kc_b.to(self.device), action_b.to(self.device), reward_b.to(self.device)
        kc_next_b, done_b = kc_next_b.to(self.device), done_b.to(self.device)

        with torch.no_grad():
            next_q = self.q_target(kc_next_b).max(dim=1).values
            target = reward_b + self.gamma * next_q * (1.0 - done_b)

        current_q = self.q(kc_b).gather(1, action_b.unsqueeze(1)).squeeze(1)
        loss = F.mse_loss(current_q, target)

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        self._train_steps += 1
        if self._train_steps % self.target_sync_every == 0:
            self.q_target.load_state_dict(self.q.state_dict())

        return float((current_q - target).abs().mean().item())

    def state_dict(self) -> dict:
        return {"q": self.q.state_dict()}

    def load_state_dict(self, state: dict) -> None:
        self.q.load_state_dict(state["q"])
        self.q_target.load_state_dict(state["q"])
