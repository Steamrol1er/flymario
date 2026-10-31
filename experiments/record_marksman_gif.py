"""Record one real-wiring Fly Marksman episode as a GIF for the README.
Left: the ViZDoom view. Right: live firing rates of the 8 readout groups and
the chosen buttons. Run: python -m experiments.record_marksman_gif [--policy fly] [--seed 3]
"""
import argparse

import numpy as np
from PIL import Image, ImageDraw

from connectome.marksman_circuit import load_marksman_circuit
from env.vizdoom_marksman_env import MarksmanEnv
from experiments.run_marksman import make_policy
from sim.marksman_brain import PANEL, steer_signal

W, H = 320, 240


def draw(frame, rates, action, tic, kills, policy):
    img = Image.fromarray(frame).resize((W, H), Image.NEAREST)
    canvas = Image.new("RGB", (W + 200, H), (13, 10, 7))
    canvas.paste(img, (0, 0))
    d = ImageDraw.Draw(canvas)
    d.text((W + 8, 6), f"fly brain ({policy})", fill=(232, 184, 109))
    d.text((W + 8, 20), f"t={tic / 35:4.1f}s kills={kills}", fill=(232, 227, 220))
    for j, k in enumerate(PANEL):
        y = 40 + j * 22
        v = rates.get(k, 0.0) if rates else 0.0
        d.text((W + 8, y), f"{k} {v:5.0f}Hz", fill=(232, 227, 220))
        col = (201, 148, 74) if "DNa" in k else (127, 199, 194) if "AOTU" in k else (140, 130, 120)
        d.rectangle([W + 8, y + 11, W + 8 + int(min(v, 150) / 150 * 180), y + 16], fill=col)
    names = [n for n, on in zip(["LEFT", "RIGHT", "FIRE"], action) if on]
    d.text((W + 8, H - 18), "> " + (" + ".join(names) or "-"), fill=(255, 120, 90) if action[2] else (232, 227, 220))
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="fly")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--out", default="docs/marksman_demo.gif")
    a = ap.parse_args()
    circuit = load_marksman_circuit()
    env = MarksmanEnv(seed=a.seed)
    act, _ = make_policy(a.policy, circuit, a.seed)
    env.new_episode()
    frames, n = [], 0
    while not env.is_finished():
        st = env.game.get_state()
        action, rates = act(env.enemies())
        if n % 2 == 0 and st is not None:
            frames.append(draw(st.screen_buffer, rates, action, st.number, env.kills(), a.policy))
        env.step(*action)
        n += 1
    print("kills", env.kills(), "frames", len(frames))
    env.close()
    pal = [f.quantize(colors=64, method=Image.Quantize.FASTOCTREE) for f in frames]
    pal[0].save(a.out, save_all=True, append_images=pal[1:], duration=57, loop=0, optimize=True)


if __name__ == "__main__":
    main()
