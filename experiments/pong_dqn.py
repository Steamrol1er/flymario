"""Pong DQN with a swappable encoder and head, for the features-vs-learning-rule
question left open by docs/PONG.md: no earlier agent beat random play (3.1%).

Encoders (all take the 2-frame stacked 256-byte RAM, scaled to [0,1]):
  kc    the earlier fixed random projection to 200 toy "Kenyon cells" (+ divisive norm)
  raw   the 256 RAM bytes themselves (no fly structure; checks the pipeline can learn at all)
  rel   HAND-ENGINEERED relation features (ball_y - paddle_y, ball x, ball velocity).
        This breaks the native-interface rule on purpose: an upper reference that
        tests whether features are the bottleneck, not a fly-brain claim.
  mb    real FlyWire mushroom-body wiring (see connectome/mushroom_body.py)
  mbshuf  same, with ALPN->KC targets shuffled (degree-preserving control)

Head: linear (hidden=0) or one hidden layer. Unlike experiments/train_pong_hybrid.py,
the end of a point is treated as terminal for the Q-target (no bootstrapping across
points). The replay buffer stores raw RAM and encodes at sample time.

Run: python -m experiments.pong_dqn --encoder rel --hidden 128 --seed 0 --steps 1000000 --tag rel_mlp
Writes analysis/pong_{tag}_s{seed}.json (curve + final greedy eval).
"""
import argparse
import json
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from connectome.loader import load_toy_connectome
from env.pong_env import PongEnv, N_ACTIONS
from experiments.train_pong import FrameStack, N_FRAME_STACK

MAX_STEPS = 1200
RAM_P_Y, RAM_B_Y, RAM_B_X, RAM_E_Y = 51, 54, 49, 50


def make_encoder(kind, seed):
    n_in = 128 * N_FRAME_STACK
    if kind == "kc":
        n_kc = len(load_toy_connectome()[1]["kc"])
        proj = torch.tensor(np.random.default_rng(seed).standard_normal((n_in, n_kc)).astype(np.float32) * 0.5)

        def enc(x):
            kc = torch.relu(x @ proj)
            return kc / (kc.sum(1, keepdim=True) + 1e-6) * 4.0
        return enc, n_kc
    if kind == "raw":
        return (lambda x: x), n_in
    if kind == "rel":
        def enc(x):
            r = x * 255.0
            old, new = r[:, :128], r[:, 128:]
            return torch.stack([(new[:, RAM_B_Y] - new[:, RAM_P_Y]) / 100.0, new[:, RAM_B_X] / 160.0,
                                (new[:, RAM_B_Y] - old[:, RAM_B_Y]) / 5.0,
                                (new[:, RAM_B_X] - old[:, RAM_B_X]) / 5.0, new[:, RAM_P_Y] / 160.0], 1)
        return enc, 5
    if kind in ("mb", "mbshuf"):
        from connectome.mushroom_body import make_mb_encoder
        return make_mb_encoder(seed, n_in, shuffle=(kind == "mbshuf"))
    raise ValueError(kind)


class Net(nn.Module):
    def __init__(self, n_in, hidden):
        super().__init__()
        self.f = nn.Linear(n_in, N_ACTIONS) if hidden == 0 else nn.Sequential(
            nn.Linear(n_in, hidden), nn.ReLU(), nn.Linear(hidden, N_ACTIONS))

    def forward(self, x):
        return self.f(x)


class Agent:
    def __init__(self, enc, n_in, hidden, lr=1e-3, gamma=0.97, cap=20_000, bs=64, sync=500, warm=500):
        self.enc, self.gamma, self.bs, self.sync, self.warm = enc, gamma, bs, sync, warm
        self.q, self.qt = Net(n_in, hidden), Net(n_in, hidden)
        self.qt.load_state_dict(self.q.state_dict())
        self.opt = torch.optim.Adam(self.q.parameters(), lr=lr)
        self.buf, self.cap, self.n_upd = [], cap, 0

    def act(self, ram, eps):
        if random.random() < eps:
            return random.randrange(N_ACTIONS)
        with torch.no_grad():
            return int(self.q(self.enc(torch.tensor(ram, dtype=torch.float32)[None] / 255.0)).argmax())

    def learn(self, ram, a, r, ram2, terminal):
        if len(self.buf) >= self.cap:
            self.buf.pop(random.randrange(len(self.buf)))
        self.buf.append((ram, a, r, ram2, float(terminal)))
        if len(self.buf) < max(self.bs, self.warm):
            return
        b = random.sample(self.buf, self.bs)
        x = torch.tensor(np.stack([t[0] for t in b]), dtype=torch.float32) / 255.0
        x2 = torch.tensor(np.stack([t[3] for t in b]), dtype=torch.float32) / 255.0
        a_t = torch.tensor([t[1] for t in b])
        r_t = torch.tensor([t[2] for t in b], dtype=torch.float32)
        d_t = torch.tensor([t[4] for t in b], dtype=torch.float32)
        with torch.no_grad():
            target = r_t + self.gamma * self.qt(self.enc(x2)).max(1).values * (1 - d_t)
        loss = F.smooth_l1_loss(self.q(self.enc(x)).gather(1, a_t[:, None]).squeeze(1), target)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 10.0)
        self.opt.step()
        self.n_upd += 1
        if self.n_upd % self.sync == 0:
            self.qt.load_state_dict(self.q.state_dict())


def play_point(env, agent, eps, learn):
    obs, _ = env.reset()
    stack = FrameStack()
    ram = stack.reset(obs)
    for t in range(MAX_STEPS):
        a = agent.act(ram, eps)
        obs, _, r, done = env.step(a)
        ram2 = stack.push(obs)
        if learn:
            agent.learn(ram, a, r, ram2, done or r != 0.0)
        ram = ram2
        if done or r != 0.0:
            return t + 1, r > 0
    return MAX_STEPS, False


def evaluate(agent, seed, points, eps):
    env = PongEnv()
    env.reset(seed=1000 + seed)
    random.seed(1000 + seed)
    return 100.0 * float(np.mean([play_point(env, agent, eps, False)[1] for _ in range(points)]))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--encoder", choices=["kc", "raw", "rel", "mb", "mbshuf"], required=True)
    ap.add_argument("--hidden", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=1_000_000)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--eval-every", type=int, default=250_000)
    a = ap.parse_args()
    torch.manual_seed(a.seed), np.random.seed(a.seed), random.seed(a.seed)
    enc, n_in = make_encoder(a.encoder, a.seed)
    agent = Agent(enc, n_in, a.hidden)
    env = PongEnv()
    env.reset(seed=a.seed)
    hist, evals, steps, nxt, t0 = [], [], 0, a.eval_every, time.time()
    while steps < a.steps:
        eps = max(0.05, 1.0 - steps / (0.5 * a.steps))
        n, won = play_point(env, agent, eps, True)
        steps += n
        hist.append(won)
        if steps >= nxt:
            nxt += a.eval_every
            evals.append({"steps": steps, "greedy": evaluate(agent, a.seed, 200, 0.0),
                          "train_last200": 100.0 * float(np.mean(hist[-200:]))})
            print(a.tag, a.seed, evals[-1], f"{(time.time() - t0) / 60:.1f}m", flush=True)
    final = {"greedy": evaluate(agent, a.seed, 1000, 0.0), "eps05": evaluate(agent, a.seed, 1000, 0.05)}
    print(a.tag, a.seed, "FINAL", final, flush=True)
    json.dump({"tag": a.tag, "seed": a.seed, "encoder": a.encoder, "hidden": a.hidden, "steps": steps,
               "evals": evals, "final": final}, open(f"analysis/pong_{a.tag}_s{a.seed}.json", "w"))


if __name__ == "__main__":
    main()
