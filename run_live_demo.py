#!/usr/bin/env python
"""Run this from a terminal (e.g. inside VS Code) to watch the fly play
real Atari Pong live: it starts a local server (live/server.py) running
the real game plus the real dopamine-learning agent, then opens three
browser tabs -- the game, the fly's parameters, and its brain activity --
all reading from that one live process. Everything runs on your own
machine; nothing is uploaded anywhere.

Usage:
    python run_live_demo.py

Then use the speed buttons on the Game tab to slow the simulation down (as
slow as 0.1x) to actually watch the fly and the ball frame by frame, or
speed it up (up to 4x) to let more points play out.

Stop with Ctrl+C in this terminal; closing the browser tabs alone does not
stop the server.
"""
from __future__ import annotations

import subprocess
import sys
import time
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PORT = 8765
BASE_URL = f"http://127.0.0.1:{PORT}"
TABS = ["/game.html", "/fly.html", "/brain.html"]


def wait_for_server(timeout_s: float = 15.0) -> bool:
    import urllib.request
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"{BASE_URL}/state.json", timeout=1)
            return True
        except Exception:
            time.sleep(0.3)
    return False


def main():
    print("Starting FlyMario live Pong server...")
    proc = subprocess.Popen([sys.executable, str(ROOT / "live" / "server.py")])
    try:
        if not wait_for_server():
            print("Server didn't come up in time -- check the terminal output above for errors.")
            return
        print("Server is up. Opening three tabs...")
        for tab in TABS:
            webbrowser.open_new_tab(BASE_URL + tab)
            time.sleep(0.3)
        print(f"\nRunning at {BASE_URL} -- Ctrl+C here to stop.\n")
        proc.wait()
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    main()
