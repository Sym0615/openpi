#!/usr/bin/env bash
set -Eeuo pipefail

OPENPI_WORKDIR="${OPENPI_WORKDIR:-/sunyiming/openpi}"
cd "${OPENPI_WORKDIR}"
export JAX_PLATFORMS=cpu
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-${OPENPI_WORKDIR}/data/openpi_assets}"

# Default: scan all valid train windows. Set OPENPI_NORM_MAX_FRAMES to sample
# windows throughout the train split; this does not restrict training data.
norm_args=(
    --config-name pi05_abc_sim
    --petrel-num-workers "${OPENPI_NORM_WORKERS:-32}"
    --petrel-episode-batch-size "${OPENPI_NORM_EPISODE_BATCH_SIZE:-4096}"
    --petrel-reservoir-rows "${OPENPI_NORM_RESERVOIR_ROWS:-200000}"
)
if [[ -n "${OPENPI_NORM_MAX_FRAMES:-}" ]]; then
    norm_args+=(--max-frames "${OPENPI_NORM_MAX_FRAMES}")
fi
for argument in "$@"; do
    if [[ "${argument}" == "--help" || "${argument}" == "-h" ]]; then
        exec .venv/bin/python scripts/compute_norm_stats.py "${norm_args[@]}" "$@"
    fi
done
identity_snapshot="$(mktemp "${TMPDIR:-/tmp}/abc-sim-norm-identity.XXXXXX")"
trap 'rm -f "${identity_snapshot}"' EXIT
.venv/bin/python scripts/check_abcsim_training_data.py --write-identity "${identity_snapshot}"
.venv/bin/python scripts/compute_norm_stats.py "${norm_args[@]}" "$@"
# Record the estimation limit and projection identity alongside the statistics.
.venv/bin/python - "${identity_snapshot}" "$@" <<'PY'
import datetime
import hashlib
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, "scripts")
from check_abcsim_training_data import input_identity

identity = json.loads(Path(sys.argv[1]).read_text())
if input_identity() != identity:
    raise RuntimeError("Normalization inputs changed during computation; recompute before training")
options = {}
arguments = iter(sys.argv[2:])
for argument in arguments:
    if argument.startswith("--"):
        name, separator, value = argument.partition("=")
        options[name] = value if separator else next(arguments, "")
output = Path(options.get("--output-dir", "assets/pi05_abc_sim")) / "abc_sim"
limit = options.get("--max-frames", os.environ.get("OPENPI_NORM_MAX_FRAMES"))
paths = ["configs/petrel/pi05_abc_sim.yaml", "configs/petrel/control_profiles/abc_sim.yaml",
         "src/openpi/training/robot_models/yam/yam_fk.xml"]
metadata = {
    "created_at": datetime.datetime.now(datetime.UTC).isoformat(),
    "dataset_id": "abc_sim", "stats_group": "dataset/abc_sim",
    "max_frames": int(limit) if limit else None,
    "input_sha256": identity,
    "projection_sha256": {path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in paths},
}
(output / "norm_stats_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
PY
