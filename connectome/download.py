"""Download connectome data.

Two sources are supported:

* ``shiu`` (default, recommended): the pre-built connectivity table from
  Shiu et al. 2024, "A leaky integrate-and-fire computational model based on
  the connectome of the entire adult Drosophila brain" (PNAS). This is a
  cleaned, ready-to-simulate parquet file derived from FlyWire v783, and the
  paper already validates it against real behavioral/physiological
  experiments (e.g. sugar-sensing neurons driving feeding motor neuron MN9).
  Starting here saves weeks versus re-deriving weights from raw FlyWire
  exports, and gives us something to check our own simulator against.

* ``flywire``: raw FlyWire Codex CSV exports (neurons/classification,
  connections). Much larger, and needs a Codex account + API token, which
  can't be fully automated here. This function prints where to get it and
  what to download by hand.

Usage:
    python -m connectome.download --dest data/
    python -m connectome.download --dest data/ --source flywire
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
import urllib.error
from pathlib import Path

# Shiu et al. 2024 repo: philshiu/Drosophila_brain_model
# These are the files referenced in the paper/repo README as the connectivity
# tables derived from FlyWire v783.
SHIU_REPO_RAW = "https://raw.githubusercontent.com/philshiu/Drosophila_brain_model/main"
# NOTE: these files live at the repo ROOT, not under data/ — verified against
# the repo's file listing on GitHub (2026-09-26).
SHIU_FILES = {
    "Connectivity_783.parquet": f"{SHIU_REPO_RAW}/Connectivity_783.parquet",
    "Completeness_783.csv": f"{SHIU_REPO_RAW}/Completeness_783.csv",
    "model.py": f"{SHIU_REPO_RAW}/model.py",
    "utils.py": f"{SHIU_REPO_RAW}/utils.py",
}

# Public, CC BY 4.0 adult Drosophila brain domain meshes (JFRC2 template
# space) from VirtualFlyBrain/DrosAdultBRAINdomains -- used by
# analysis/build_brain_mesh.py for the 3D visualization (viz/brain3d.html).
# Domain index 49 = "GNG" (adult gnathal ganglion, the modern name for the
# subesophageal zone), looked up from that repo's
# refData/Original_Index.tsv. Domain index 0 is the whole-brain outline.
BRAIN_MESH_REPO_RAW = "https://raw.githubusercontent.com/VirtualFlyBrain/DrosAdultBRAINdomains/master"
BRAIN_MESH_FILES = {
    "whole_brain.obj": f"{BRAIN_MESH_REPO_RAW}/individualDomainFiles/AdultBrainDomain0000.obj",
    "gng.obj": f"{BRAIN_MESH_REPO_RAW}/individualDomainFiles/AdultBrainDomain0049.obj",
}

FLYWIRE_CODEX_URL = "https://codex.flywire.ai/api/download"
FLYWIRE_INSTRUCTIONS = """\
FlyWire Codex downloads require a (free) account and are served from a
dynamic, versioned download page rather than a stable direct-download URL,
so this can't be fully scripted. To get the raw connectome:

  1. Go to https://codex.flywire.ai/api/download and sign in.
  2. Pick the "v783" (or latest stable) public release.
  3. Download:
       - neurons.csv / classification.csv  (neuron IDs, cell types, side)
       - connections.csv                   (pre, post, region, syn count, NT)
       - cell_types.csv                    (annotations)
  4. Save them into: {dest}
  5. Re-run: python -m connectome.download --source flywire --verify --dest {dest}

Cite: Dorkenwald et al. 2024 (Nature) and Schlegel et al. 2024 (Nature) if
you use this data.
"""


def _download(url: str, dest_path: Path, timeout: int = 120) -> bool:
    try:
        print(f"  fetching {url}")
        req = urllib.request.Request(url, headers={"User-Agent": "flymario/0.1"})
        with urllib.request.urlopen(req, timeout=timeout) as resp, open(dest_path, "wb") as f:
            total = 0
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                total += len(chunk)
            print(f"  wrote {dest_path} ({total / 1e6:.1f} MB)")
        return True
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
        print(f"  FAILED: {e}", file=sys.stderr)
        return False


def download_shiu(dest: Path) -> bool:
    dest.mkdir(parents=True, exist_ok=True)
    ok = True
    for fname, url in SHIU_FILES.items():
        out = dest / fname
        if out.exists():
            print(f"  already have {out}, skipping")
            continue
        ok = _download(url, out) and ok
    return ok


def print_flywire_instructions(dest: Path) -> None:
    print(FLYWIRE_INSTRUCTIONS.format(dest=dest))


def download_brain_mesh(dest: Path) -> bool:
    dest.mkdir(parents=True, exist_ok=True)
    ok = True
    for fname, url in BRAIN_MESH_FILES.items():
        out = dest / fname
        if out.exists():
            print(f"  already have {out}, skipping")
            continue
        ok = _download(url, out) and ok
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dest", type=Path, default=Path("data"))
    ap.add_argument("--source", choices=["shiu", "flywire", "brain-mesh"], default="shiu")
    ap.add_argument("--verify", action="store_true", help="only check that expected files exist")
    args = ap.parse_args()

    if args.source == "brain-mesh":
        mesh_dest = args.dest / "brain_mesh"
        if args.verify:
            missing = [f for f in BRAIN_MESH_FILES if not (mesh_dest / f).exists()]
            if missing:
                print(f"Missing: {missing}")
                sys.exit(1)
            print("Brain mesh files present.")
            return
        print(f"Downloading VirtualFlyBrain adult brain domain meshes to {mesh_dest} ...")
        ok = download_brain_mesh(mesh_dest)
        if not ok:
            sys.exit(1)
        print("Done. Next: python -m analysis.build_brain_mesh")
    elif args.source == "shiu":
        if args.verify:
            missing = [f for f in SHIU_FILES if not (args.dest / f).exists()]
            if missing:
                print(f"Missing: {missing}")
                sys.exit(1)
            print("All Shiu et al. connectivity files present.")
            return
        print(f"Downloading Shiu et al. 2024 connectivity tables to {args.dest} ...")
        ok = download_shiu(args.dest)
        if not ok:
            print(
                "\nAutomated download failed (network restrictions, or the repo's "
                "file layout changed). Fallback: clone the repo by hand:\n"
                "  git clone https://github.com/philshiu/Drosophila_brain_model\n"
                "and copy its data/ files into "
                f"{args.dest}\n",
                file=sys.stderr,
            )
            sys.exit(1)
        print("Done. Next: python -m tests.check_sugar_feeding")
    else:
        print_flywire_instructions(args.dest)


if __name__ == "__main__":
    main()
