#!/usr/bin/env python3
"""Check ABC-Sim filter and normalization input identities before training."""

import argparse
import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG = "configs/petrel/pi05_abc_sim.yaml"
FILTER_CONFIG = "configs/petrel/frame_filters/abc_sim.yaml"
FILTER_ROOT = "data/petrel_cache/filter/abc_sim/train"
INDEX = "data/petrel_cache/path_index/abc_sim/train/path_index.jsonl"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def input_identity() -> dict[str, str]:
    config = yaml.safe_load((ROOT / CONFIG).read_text())
    paths = [CONFIG, "configs/petrel/control_profiles/abc_sim.yaml", "src/openpi/training/robot_models/yam/yam_fk.xml"]
    if config["frame_filter"]["enabled"]:
        sidecar = ROOT / FILTER_ROOT / "frame_filter.jsonl"
        manifest_path = ROOT / FILTER_ROOT / "frame_filter_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        if not manifest.get("published") or manifest.get("status") != "ready":
            raise ValueError("ABC-Sim frame filter is not ready; run scripts/build_abcsim_frame_filter.py")
        for path, expected in [
            (sidecar, manifest["sidecar_sha256"]),
            (ROOT / FILTER_CONFIG, manifest["filter_config_sha256"]),
            (ROOT / INDEX, manifest["path_index_sha256"]),
        ]:
            if sha256(path) != expected:
                raise ValueError(f"ABC-Sim filter identity mismatch: {path}; rebuild frame filters")
        paths.extend([FILTER_CONFIG, f"{FILTER_ROOT}/frame_filter.jsonl", INDEX])
    return {path: sha256(ROOT / path) for path in paths}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-identity", type=Path, help="Capture normalization inputs before computation")
    args = parser.parse_args()
    identity = input_identity()
    if args.write_identity:
        args.write_identity.write_text(json.dumps(identity, indent=2) + "\n")
        return
    stats_root = ROOT / "assets/pi05_abc_sim/abc_sim"
    if not (stats_root / "norm_stats.json").is_file():
        raise FileNotFoundError("Missing ABC-Sim normalization statistics")
    metadata = json.loads((stats_root / "norm_stats_metadata.json").read_text())
    recorded = metadata.get("input_sha256", metadata.get("projection_sha256", {}))
    mismatches = [path for path, digest in identity.items() if recorded.get(path) != digest]
    if mismatches:
        raise ValueError(
            "ABC-Sim normalization was computed with different inputs: "
            + ", ".join(mismatches)
            + ". Recompute statistics with scripts/compute_abcsim_norm_stats.sh before training."
        )
    print("[data-preflight] frame filter and normalization input hashes match", flush=True)


if __name__ == "__main__":
    main()
