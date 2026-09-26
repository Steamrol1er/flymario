"""Preprocess the real public VirtualFlyBrain adult brain domain meshes
(CC BY 4.0, JFRC2 template space -- same source as analysis/build_brain_mesh.py)
into a compact JSON for live/brain.html's 3D view.

Highlights the RIGHT OPTIC LOBE (medulla + lobula) rather than the gnathal
ganglion used for the earlier sugar-feeding demo: for a visual
motion-tracking task like Pong, the optic lobe -- specifically the medulla
and lobula, where the actual T4/T5 motion-detection circuits live -- is
the anatomically correct region to highlight, not an arbitrary choice.
See docs/PONG.md for the citations behind that.

Run: python live/build_assets.py
Writes: live/assets/brain_mesh.json
Needs: data/brain_mesh/{whole_brain,medulla_R,lobula_R}.obj -- get all of
them (plus gng.obj, not used here) with:
    python -m connectome.download --source brain-mesh --dest data/
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MESH_DIR = ROOT / "data" / "brain_mesh"
OUT_PATH = Path(__file__).resolve().parent / "assets" / "brain_mesh.json"


def parse_obj(path: Path):
    verts, faces = [], []
    with open(path) as f:
        for line in f:
            if line.startswith("v "):
                _, x, y, z = line.split()[:4]
                verts.append((float(x), float(y), float(z)))
            elif line.startswith("f "):
                idx = [int(tok.split("/")[0]) - 1 for tok in line.split()[1:4]]
                faces.append(idx)
    return verts, faces


def merge(*mesh_pairs):
    all_v, all_f = [], []
    offset = 0
    for verts, faces in mesh_pairs:
        all_v.extend(verts)
        all_f.extend([[i + offset, j + offset, k + offset] for i, j, k in faces])
        offset += len(verts)
    return all_v, all_f


def bbox(verts):
    xs = [v[0] for v in verts]; ys = [v[1] for v in verts]; zs = [v[2] for v in verts]
    return (min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs))


def main():
    whole_v, whole_f = parse_obj(MESH_DIR / "whole_brain.obj")
    me_v, me_f = parse_obj(MESH_DIR / "medulla_R.obj")
    lo_v, lo_f = parse_obj(MESH_DIR / "lobula_R.obj")
    visual_v, visual_f = merge((me_v, me_f), (lo_v, lo_f))

    lo, hi = bbox(whole_v)
    center = tuple((lo[i] + hi[i]) / 2 for i in range(3))
    span = max(hi[i] - lo[i] for i in range(3))
    scale = 2.0 / span

    def norm(verts):
        return [[round((x - center[0]) * scale, 4),
                  round((y - center[1]) * scale, 4),
                  round((z - center[2]) * scale, 4)] for x, y, z in verts]

    data = {
        "source": "VirtualFlyBrain/DrosAdultBRAINdomains (CC BY 4.0), JFRC2 template space",
        "highlight_region": "right optic lobe (medulla + lobula) -- visual motion processing",
        "whole_brain": {"vertices": norm(whole_v), "faces": whole_f},
        "visual_lobe": {"vertices": norm(visual_v), "faces": visual_f},
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(data, f)
    print(f"whole brain: {len(whole_v)} verts / {len(whole_f)} faces")
    print(f"visual lobe (medulla+lobula): {len(visual_v)} verts / {len(visual_f)} faces")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
