#!/usr/bin/env python3
"""Generate camera-schema-specific ABC-130K-v3 dataset-group manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

DEFAULT_DATA_ROOT = Path("/datahub/public/real/ABC-130K-v3/train")
DEFAULT_3CAM_OUTPUT = Path("configs/petrel/dataset_groups/abc130k/abc130k.yaml")
DEFAULT_4CAM_OUTPUT = Path("configs/petrel/dataset_groups/abc130k/abc130k_4cam.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--three-cam-output", type=Path, default=DEFAULT_3CAM_OUTPUT)
    parser.add_argument("--four-cam-output", type=Path, default=DEFAULT_4CAM_OUTPUT)
    parser.add_argument(
        "--four-cam-top-view",
        choices=("left", "right"),
        default="left",
        help="Stereo head camera mapped to the model's base_0_rgb input (default: left).",
    )
    return parser.parse_args()


def validated_schema_paths(data_root: Path) -> dict[str, list[tuple[str, str]]]:
    datasets: dict[str, list[tuple[str, str]]] = {"3cam": [], "4cam": []}
    for task_dir in sorted(path for path in data_root.iterdir() if path.is_dir()):
        validation_path = task_dir / "validation.json"
        if not validation_path.is_file():
            raise FileNotFoundError(f"Missing task validation report: {validation_path}")
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        if validation.get("passed") is not True:
            raise ValueError(f"Task validation did not pass: {validation_path}")
        for schema_dir in sorted(
            path for path in task_dir.iterdir() if path.is_dir() and path.name.startswith("schema=")
        ):
            if not (schema_dir / "meta/info.json").is_file():
                raise FileNotFoundError(f"Missing LeRobot metadata: {schema_dir / 'meta/info.json'}")
            relative_path = schema_dir.relative_to(data_root).as_posix()
            schema_name = schema_dir.name.removeprefix("schema=")
            if schema_name.startswith("3cam_"):
                camera_group = "3cam"
            elif schema_name.startswith("4cam_"):
                camera_group = "4cam"
            else:
                raise ValueError(f"Unsupported ABC camera schema: {schema_dir}")
            datasets[camera_group].append((f"{task_dir.name}__{schema_name}", relative_path))
    if not any(datasets.values()):
        raise ValueError(f"No schema directories found under {data_root}")
    return datasets


def write_manifest(output: Path, *, name: str, profile: str, datasets: list[tuple[str, str]]) -> None:
    if not datasets:
        raise ValueError(f"Cannot write empty ABC dataset group {name!r}")
    payload = {
        "schema": "openpi_petrel_dataset_group_v1",
        "name": name,
        "category": "abc130k",
        "profile": profile,
        "format": "v3",
        # Dataset paths are relative to storage.local_root in pi05_abc130k.yaml.
        "root": ".",
        "datasets": [{"name": dataset_name, "path": path} for dataset_name, path in datasets],
    }
    output = output.expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(payload, sort_keys=False, width=160), encoding="utf-8")
    print(f"Wrote {len(datasets)} {name} schema datasets to {output} (profile={profile})")


def main() -> None:
    args = parse_args()
    data_root = args.data_root.expanduser().resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"ABC-130K-v3 train root does not exist: {data_root}")

    datasets = validated_schema_paths(data_root)
    write_manifest(
        args.three_cam_output,
        name="abc130k_3cam",
        profile="abc130k_3cam",
        datasets=datasets["3cam"],
    )
    write_manifest(
        args.four_cam_output,
        name="abc130k_4cam",
        profile=f"abc130k_4cam_top_{args.four_cam_top_view}",
        datasets=datasets["4cam"],
    )


if __name__ == "__main__":
    main()
