"""Preprocess the public VirtualFlyBrain adult brain domain meshes into a
compact JSON for the 3D visualization (viz/brain3d.html).

Source data (CC BY 4.0, VirtualFlyBrain/DrosAdultBRAINdomains, JFRC2 template
space). Fetch it first with:
    python -m connectome.download --source brain-mesh --dest data/
which downloads into data/brain_mesh/:
  - whole_brain.obj (domain index 0)  -- whole-brain neuropil surface
  - gng.obj (domain index 49)         -- adult gnathal ganglion / GNG, the
    modern name for the subesophageal zone -- where gustatory/sugar
    afferents arrive and where feeding motor neurons including MN9 sit;
    domain index looked up in that repo's refData/Original_Index.tsv

We parse the plain-text Wavefront OBJ ourselves (just `v x y z` and
`f i j k` lines; both files use pure triangles, no normals/UVs) rather than
pulling in an OBJLoader library, then center and scale both meshes into the
same coordinate frame so they overlay correctly.

Run: python -m analysis.build_brain_mesh
Writes: analysis/brain_mesh.json
"""
from __future__ import annotations

import json


def parse_obj(path: str):
    verts = []
    faces = []
    with open(path) as f:
        for line in f:
            if line.startswith("v "):
                _, x, y, z = line.split()[:4]
                verts.append((float(x), float(y), float(z)))
            elif line.startswith("f "):
                idx = [int(tok.split("/")[0]) - 1 for tok in line.split()[1:4]]
                faces.append(idx)
    return verts, faces


def bbox(verts):
    xs = [v[0] for v in verts]; ys = [v[1] for v in verts]; zs = [v[2] for v in verts]
    return (min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs))


def main():
    whole_v, whole_f = parse_obj("data/brain_mesh/whole_brain.obj")
    gng_v, gng_f = parse_obj("data/brain_mesh/gng.obj")

    # Center/scale using the WHOLE BRAIN's bounding box so both meshes stay
    # correctly aligned relative to each other (both are already in the same
    # JFRC2 template coordinate space, in micrometers).
    lo, hi = bbox(whole_v)
    center = tuple((lo[i] + hi[i]) / 2 for i in range(3))
    span = max(hi[i] - lo[i] for i in range(3))
    scale = 2.0 / span  # normalize longest axis to ~2 scene units

    def norm(verts):
        out = []
        for x, y, z in verts:
            out.append([
                round((x - center[0]) * scale, 4),
                round((y - center[1]) * scale, 4),
                round((z - center[2]) * scale, 4),
            ])
        return out

    data = {
        "source": "VirtualFlyBrain/DrosAdultBRAINdomains (CC BY 4.0), JFRC2 template space",
        "whole_brain": {"vertices": norm(whole_v), "faces": whole_f},
        "gng": {"vertices": norm(gng_v), "faces": gng_f},
    }
    with open("analysis/brain_mesh.json", "w") as f:
        json.dump(data, f)
    print(f"whole brain: {len(whole_v)} verts / {len(whole_f)} faces")
    print(f"GNG:         {len(gng_v)} verts / {len(gng_f)} faces")
    print("Wrote analysis/brain_mesh.json")


if __name__ == "__main__":
    main()
