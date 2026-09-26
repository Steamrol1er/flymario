"""Local live demo: real Atari Pong + the real dopamine-learning agent,
playing and learning continuously, watched from three browser tabs opened
by run_live_demo.py (project root) -- game, fly parameters, brain
activity. No Claude Artifact involved; this runs entirely on your own
machine, e.g. started from a VS Code terminal.

Architecture (stdlib only, no new dependencies):
  - a background thread runs the real game + agent loop forever, one
    Pong point per "episode", updating a thread-safe shared State
  - a stdlib http.server.ThreadingHTTPServer serves:
      GET  /                -> game.html
      GET  /game.html, /fly.html, /brain.html, /style.css, /assets/*
      GET  /state.json      -> current agent/game state (poll this;
                                includes real RAM-derived paddle/ball
                                positions -- game.html draws its own crisp
                                vector reconstruction from these instead of
                                fetching a re-encoded frame image, which is
                                what fixed this view's earlier stutter)
      POST /speed           -> {"speed": float} playback speed multiplier
      POST /pause           -> {"paused": bool}
  - the three HTML pages poll /state.json on a timer; no websockets needed
    at this update rate

See docs/PONG.md for the full account of the Pong experiment this reuses
(research, RAM addresses, the two stability fixes). This file adds:
continuous live play+learning instead of a precomputed replay, a speed
control, and the three-tab layout.
"""
from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from connectome.loader import load_toy_connectome
from env.pong_env import PongEnv, N_ACTIONS, ACTION_MEANINGS
from learning.actor import ActorCritic, ActorCriticParams
from learning.dopamine import DopamineParams

PORT = 8765
BASE_DELAY_S = 1.0 / 20.0   # "speed 1.0" ~= 20 simulated steps/sec (slower
                            # than real Atari's 60fps on purpose, so it's
                            # actually watchable without slowing it down)
MIN_SPEED, MAX_SPEED = 0.05, 4.0


class RamToKC:
    """Same divisive-normalization fix as experiments/train_pong.py's
    RamToKC (docs/STATUS.md / docs/PONG.md) -- kept as an independent copy
    here rather than imported, so this file works standalone if
    experiments/train_pong.py's internals change shape later."""
    def __init__(self, n_in: int, n_kc: int, seed: int = 0, target_total: float = 4.0):
        import torch
        rng = np.random.default_rng(seed)
        self.proj = torch.tensor(rng.standard_normal((n_in, n_kc)).astype(np.float32) * 0.5)
        self.target_total = target_total
        self._torch = torch

    def __call__(self, ram: np.ndarray):
        torch = self._torch
        x = torch.tensor(ram, dtype=torch.float32) / 255.0
        raw = x @ self.proj
        kc = torch.relu(raw)
        return kc / (kc.sum() + 1e-6) * self.target_total


class SharedState:
    """Everything the three tabs poll, behind one lock. Kept as plain
    JSON-able Python values (not tensors) so the HTTP handler never has to
    know about torch."""
    def __init__(self, n_kc: int):
        self.lock = threading.Lock()
        self.episode = 0
        self.step = 0
        self.server_time = 0.0     # time.time() at last update, for client-side interpolation
        self.action = "NOOP"
        self.value = 0.0
        self.dopamine_delta = 0.0
        self.kc_active = 0
        self.kc_activity = [0.0] * n_kc
        self.approach = 0.0
        self.avoid = 0.0
        self.epsilon = 0.0
        self.score_player = 0
        self.score_enemy = 0
        self.points_played = 0
        self.points_scored = 0
        self.speed = 1.0
        self.paused = False
        # real RAM-derived game positions, for the client's own vector
        # rendering (see live/game.html) -- rendering client-side from tiny
        # position data instead of fetching a re-encoded PNG every step is
        # what actually fixed this view's stutter; encoding+transferring an
        # image over HTTP on every single simulation step was the bottleneck.
        self.player_y = 0
        self.ball_x = 0
        self.ball_y = 0
        self.enemy_y = 0

    def snapshot_json(self) -> bytes:
        with self.lock:
            d = {k: v for k, v in self.__dict__.items() if k != "lock"}
        return json.dumps(d).encode("utf-8")


def simulation_loop(state: SharedState, stop_event: threading.Event):
    conn, groups = load_toy_connectome()
    n_kc = len(groups["kc"])
    ac = ActorCritic(
        n_kc=n_kc,
        n_actions=N_ACTIONS,
        compartments={"approach": +1, "avoid": -1},
        dopamine_params=DopamineParams(eta=0.02, trace_decay=0.9, w_min=0.0, w_max=3.0),
        actor_params=ActorCriticParams(
            gamma=0.97, eta_actor=0.05, trace_decay=0.7, da_gain=1.5,
            w_min=-3.0, w_max=3.0, normalize_advantage=True,
        ),
    )
    ram_to_kc = RamToKC(n_in=128, n_kc=n_kc)
    env = PongEnv()

    MAX_STEPS_PER_POINT = 1200
    ep = 0
    while not stop_event.is_set():
        ram, info = env.reset()
        kc = ram_to_kc(ram)
        step = 0
        for step in range(MAX_STEPS_PER_POINT):
            while state.paused and not stop_event.is_set():
                time.sleep(0.05)
            if stop_event.is_set():
                break

            epsilon = max(0.05, 0.3 * (1 - ep / 300))
            action, probs = ac.act(kc, epsilon=epsilon)
            value = float(ac.critic.value(kc))
            ram, info, reward, done = env.step(action)
            kc_next = ram_to_kc(ram)
            delta = ac.learn(reward, kc_next, done=done)

            mbon = ac.critic.mbon_output(kc)
            with state.lock:
                state.episode = ep
                state.step = step
                state.server_time = time.time()
                state.action = ACTION_MEANINGS[action]
                state.value = round(value, 3)
                state.dopamine_delta = round(float(delta), 3)
                state.kc_active = int((kc > 0).sum())
                state.kc_activity = [round(float(v), 3) for v in kc.tolist()]
                state.approach = round(float(mbon["approach"]), 3)
                state.avoid = round(float(mbon["avoid"]), 3)
                state.epsilon = round(epsilon, 3)
                state.player_y = info["player_y"]
                state.ball_x = info["ball_x"]
                state.ball_y = info["ball_y"]
                state.enemy_y = info["enemy_y"]
                if reward > 0:
                    state.score_player += 1
                    state.points_scored += 1
                elif reward < 0:
                    state.score_enemy += 1

            kc = kc_next
            speed = max(MIN_SPEED, min(state.speed, MAX_SPEED))
            time.sleep(BASE_DELAY_S / speed)
            if done or reward != 0.0:
                break

        with state.lock:
            state.points_played += 1
        ep += 1

    env.close()


class Handler(BaseHTTPRequestHandler):
    state: SharedState = None  # set by main()
    live_dir: Path = None

    def _send_bytes(self, data: bytes, content_type: str, cache: bool = False):
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        if not cache:
            self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _send_json(self, obj):
        self._send_bytes(json.dumps(obj).encode("utf-8"), "application/json")

    CONTENT_TYPES = {
        ".html": "text/html", ".css": "text/css", ".js": "application/javascript",
        ".json": "application/json", ".png": "image/png", ".jpg": "image/jpeg",
        ".svg": "image/svg+xml",
    }

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/":
            path = "/game.html"
        if path == "/state.json":
            self._send_bytes(self.state.snapshot_json(), "application/json")
            return
        # static files from live/ (and one level of subdirectory, e.g.
        # live/assets/brain_mesh.json), with a path-traversal check
        fpath = (self.live_dir / path.lstrip("/")).resolve()
        if fpath.exists() and fpath.is_file() and self.live_dir in fpath.parents:
            ctype = self.CONTENT_TYPES.get(fpath.suffix, "application/octet-stream")
            self._send_bytes(fpath.read_bytes(), ctype, cache=True)
            return
        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(body)
        except Exception:
            payload = {}

        if self.path == "/speed":
            with self.state.lock:
                self.state.speed = float(payload.get("speed", self.state.speed))
            self._send_json({"ok": True, "speed": self.state.speed})
            return
        if self.path == "/pause":
            with self.state.lock:
                self.state.paused = bool(payload.get("paused", self.state.paused))
            self._send_json({"ok": True, "paused": self.state.paused})
            return
        self.send_response(404)
        self.end_headers()

    def log_message(self, format, *args):
        pass  # keep the terminal quiet; the browser tabs are the UI


def main():
    conn, groups = load_toy_connectome()
    n_kc = len(groups["kc"])
    state = SharedState(n_kc)

    stop_event = threading.Event()
    sim_thread = threading.Thread(target=simulation_loop, args=(state, stop_event), daemon=True)
    sim_thread.start()

    Handler.state = state
    Handler.live_dir = Path(__file__).resolve().parent
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"FlyMario live Pong demo running at http://127.0.0.1:{PORT}/")
    print("Tabs: /game.html  /fly.html  /brain.html")
    print("Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        server.shutdown()


if __name__ == "__main__":
    main()
