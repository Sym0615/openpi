      
#!/usr/bin/env python3
"""Select ABC bottle episodes and probe PJ-EAI OpenPI's native Petrel reader.

Run inside the PJ-EAI OpenPI environment. Source data is read-only. The saved
episode lists are preparation artifacts; they do not change training configs.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

TASK = "sim_put_the_plastic_bottles_in_the_bin"
CAMERAS = ("observation.images.top", "observation.images.left", "observation.images.right")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def select_split(root, split, expected):
    split_root = root / split
    info = json.loads((split_root / "meta/info.json").read_text(encoding="utf-8"))
    require(str(info["codebase_version"]).startswith("v3"), f"{split}: expected LeRobot v3")
    require(float(info["fps"]) == 30, f"{split}: expected 30 FPS")
    for key in ("observation.state", "action"):
        require(list(info["features"][key]["shape"]) == [14], f"{split}: unexpected {key} shape")
    require(all(key in info["features"] for key in CAMERAS), f"{split}: missing camera feature")

    index = pq.read_table(root / "sidecar" / split / "index.parquet", columns=["episode_index", "task_name"]).to_pylist()
    ids = [int(row["episode_index"]) for row in index]
    require(len(set(ids)) == len(ids), f"{split}: duplicate episode IDs in sidecar")
    require(len(ids) == int(info["total_episodes"]), f"{split}: sidecar/info episode counts differ")
    categories = Counter(str(row["task_name"]) for row in index)
    selected_ids = sorted(int(row["episode_index"]) for row in index if row["task_name"] == TASK)
    require(len(selected_ids) == expected, f"{split}: expected {expected} bottle episodes, found {len(selected_ids)}")

    paths = sorted((split_root / "meta/episodes").rglob("*.parquet"))
    require(bool(paths), f"{split}: no episode metadata Parquet files")
    selected = set(selected_ids)
    all_meta_ids = set()
    episodes = {}
    for path in paths:
        parquet = pq.ParquetFile(path)
        columns = [key for key in ("episode_index", "length", "task_name", "tasks") if key in parquet.schema_arrow.names]
        require("episode_index" in columns and "length" in columns, f"Missing episode_index/length: {path}")
        for row in parquet.read(columns=columns).to_pylist():
            episode_id = int(row["episode_index"])
            require(episode_id not in all_meta_ids, f"{split}: duplicate episode metadata ID {episode_id}")
            all_meta_ids.add(episode_id)
            if episode_id in selected:
                if "task_name" in row:
                    require(row["task_name"] == TASK, f"{split}: sidecar/metadata task mismatch for {episode_id}")
                require(int(row["length"]) > 0, f"{split}: empty selected episode {episode_id}")
                episodes[episode_id] = row
    require(all_meta_ids == set(ids), f"{split}: sidecar and episode metadata ID sets differ")
    require(set(episodes) == selected, f"{split}: selected episode metadata is incomplete")
    tasks = pq.read_table(split_root / "meta/tasks.parquet", columns=["task_index", "task"]).to_pylist()
    task_map = {int(row["task_index"]): str(row["task"]) for row in tasks}
    require(len(task_map) == len(tasks), f"{split}: duplicate task_index")

    selection = {
        "source_root": str(root), "split": split, "task_name": TASK,
        "source_total_episodes": len(ids), "selected_episodes": len(selected_ids),
        "selected_frames": sum(int(row["length"]) for row in episodes.values()),
        "episode_indices": selected_ids,
        "note": "Original episode IDs are preserved. This list is not automatically applied to training.",
    }
    print(f"[{split}] selected={len(selected_ids)} frames={selection['selected_frames']} categories={len(categories)}", flush=True)
    return selection, episodes, task_map


def probe_split(root, out, split, selection, episodes, task_map, horizon, sample_count):
    # Lazy import permits --select-only without importing the training stack.
    from PIL import Image, ImageDraw
    from openpi.training.petrel_lerobot_reader import DatasetSpec, PetrelLeRobotDataset

    spec = DatasetSpec(
        name=f"abc_bottles_{split}_probe", remote_root=str(root / split), local_root=str(root),
        format_version=3, state_target_dim=14, action_target_dim=14,
        modalities={"state": {"keys": ["observation.state"]}, "action": {"keys": ["action"]},
                    "video": {"keys": list(CAMERAS)}},
    )
    dataset = PetrelLeRobotDataset(
        spec, episodes=selection["episode_indices"], num_frames=1, action_size=horizon,
        global_sample_stride=1, use_path_index_cache=False, local_video_path_decode=True,
        direct_parquet_row_group_read=True,
    )
    require(dataset.selected_episodes == selection["episode_indices"], f"{split}: reader episode selection mismatch")
    eligible = [episode_id for episode_id in selection["episode_indices"] if int(episodes[episode_id]["length"]) >= horizon]
    require(bool(eligible), f"{split}: no episode is long enough for horizon={horizon}")
    positions = np.linspace(0, len(eligible) - 1, min(sample_count, len(eligible)), dtype=int)
    report = []
    for position in positions:
        episode_id = eligible[int(position)]
        length = int(episodes[episode_id]["length"])
        starts = sorted({0, (length - horizon) // 2, length - horizon})
        sheet = Image.new("RGB", (224 * 3, 50 + 194 * len(starts)), "white")
        draw = ImageDraw.Draw(sheet)
        draw.text((6, 4), f"{split} episode={episode_id} length={length} (top | left | right)", fill="black")
        for row_index, frame_index in enumerate(starts):
            item = dataset.get_step_item(episode_id, frame_index)
            state = item["observation.state"].detach().cpu().numpy()
            actions = item["action"].detach().cpu().numpy()
            require(state.shape == (1, 14), f"Unexpected state shape: {state.shape}")
            require(actions.shape == (horizon, 14), f"Unexpected action shape: {actions.shape}")
            require(np.isfinite(state).all() and np.isfinite(actions).all(), "NaN/Inf in state or actions")
            require(not bool(item["action_is_pad"].any()), "Unexpected padding in a complete action window")
            task_index = int(item["task_index"].item())
            prompt = str(item["task"])
            require(bool(prompt.strip()) and task_map.get(task_index) == prompt, "task_index/text lookup mismatch")
            timestamp = float(item["timestamp"].reshape(-1)[0].item())
            require(abs(timestamp - frame_index / 30.0) < 0.001, "Frame/timestamp mismatch; inspect source timestamp convention")
            image_shapes = {}
            for column_index, camera in enumerate(CAMERAS):
                clip = dataset.get_video_for_item(item, video_key=camera, sample_indices=(0,))
                pixels = clip.detach().cpu().numpy()
                require(pixels.shape == (1, 3, 168, 224), f"Unexpected {camera} shape: {pixels.shape}")
                require(np.isfinite(pixels).all() and pixels.min() >= 0 and pixels.max() <= 1, f"Invalid RGB values: {camera}")
                rgb = np.rint(pixels[0].transpose(1, 2, 0) * 255).astype(np.uint8)
                sheet.paste(Image.fromarray(rgb), (224 * column_index, 50 + 194 * row_index))
                image_shapes[camera] = list(pixels.shape)
            draw.text((6, 50 + 194 * row_index + 171), f"frame={frame_index} t={timestamp:.3f}s task_index={task_index}", fill="black")
            stem = f"{split}_ep{episode_id:06d}_frame{frame_index:06d}"
            np.savez_compressed(out / f"{stem}.npz", state=state, action=actions,
                                task_index=np.int64(task_index), task=np.array(prompt), timestamp=np.float64(timestamp))
            report.append({"episode_index": episode_id, "frame_index": frame_index, "timestamp": timestamp,
                           "task_index": task_index, "task": prompt, "state_shape": list(state.shape),
                           "action_shape": list(actions.shape), "image_shapes": image_shapes})
        sheet.save(out / f"{split}_ep{episode_id:06d}_preview.png")
        print(f"[{split}] episode={episode_id} frames={starts} state=(1,14) action=({horizon},14) RGB=OK text={prompt!r}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/dataset/public/sim/abc-sim"))
    parser.add_argument("--out", type=Path, default=Path("data/abc_bottles_petrel_probe"))
    parser.add_argument("--expected-train", type=int, default=5588)
    parser.add_argument("--expected-val", type=int, default=287)
    parser.add_argument("--horizon", type=int, default=30, help="Probe window only; does not set a training configuration")
    parser.add_argument("--samples-per-split", type=int, default=2)
    parser.add_argument("--select-only", action="store_true")
    args = parser.parse_args()
    root, out = args.root.expanduser().resolve(), args.out.expanduser().resolve()
    require(root.is_dir(), f"Source root does not exist: {root}")
    require(not out.is_relative_to(root), "Output must be outside the shared source dataset")
    require(args.horizon > 0 and args.samples_per_split > 0, "Horizon and samples-per-split must be positive")
    out.mkdir(parents=True, exist_ok=True)
    report = {"source_root": str(root), "task_name": TASK, "horizon_for_probe": args.horizon,
              "selection_only": args.select_only, "splits": {}}
    for split, expected in (("train", args.expected_train), ("val", args.expected_val)):
        selection, episodes, tasks = select_split(root, split, expected)
        save_json(out / f"{split}_episode_selection.json", selection)
        samples = [] if args.select_only else probe_split(root, out, split, selection, episodes, tasks, args.horizon, args.samples_per_split)
        report["splits"][split] = {"selected_episodes": selection["selected_episodes"],
                                   "selected_frames": selection["selected_frames"], "samples": samples}
    report["status"] = "SELECTION_OK" if args.select_only else "READER_SMOKE_OK"
    save_json(out / ("selection_report.json" if args.select_only else "reader_smoke_report.json"), report)
    print(f"{report['status']} output={out}", flush=True)
    print("Episode lists are not yet wired into training. No action shifting, normalization, or 32D mapping was applied.", flush=True)


if __name__ == "__main__":
    main()
