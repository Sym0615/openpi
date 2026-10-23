# PI0.5 base on the full ABC-Sim train split

Run these commands from `/sunyiming/openpi`. The dataset stays in LeRobot v3
format at `/dataset/public/sim/abc-sim/train`. Validation data is not mixed into
training. The full-data config is `pi05_abc_sim`; the older bottles-only config
is still available separately.

## 1. Data and controls

`configs/petrel/pi05_abc_sim.yaml` sets the dataset mount, dimensions, cameras,
sampling, and target action semantics. `dataset_groups/abc_sim/train.yaml`
selects the entire train split. Frame filtering is required, with complete
episode coverage and `full_chunk` windows. It uses ABC130K's motionless trim,
sudden-change, and action/state trend filters on available joint/gripper
signals: first the `trim` group, then the `quality` group. FK consistency,
extreme-value filtering, and video quality filtering are omitted. All task
types remain eligible; this does not use the bottles-only task filter.

Build the full train sidecars before reading the dataset:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python scripts/build_abcsim_frame_filter.py --workers 12
```

The builder reuses `/sunyiming/data_filter_pipeline/dfp`, configurable with
`--dfp-source` or `OPENPI_DFP_SOURCE`. The resolved rules are stored in
`configs/petrel/frame_filters/abc_sim.yaml`, including the reference ABC130K
run. ABC-Sim has no logged EE, velocity, or effort signals, so these checks
use its eight named state/action joint/gripper signals. Training still
derives EE using FK. The manifest records implementation, config, path-index,
and sidecar hashes. GPU nodes need the published sidecar and manifest under
`data/petrel_cache/filter/abc_sim/train`, not the DFP source repository.

`control_profiles/abc_sim.yaml` maps the named joint/gripper columns and derives
both EE poses from six joint angles using YAM FK. The reader produces 32D
tensors: joints/grippers at 0–13, EE XYZ/RPY at 14–25, padding at 26–31.
State poses are absolute in each arm's base frame. Joint and EE action labels
are relative to the observation at the start of the action chunk; grippers
remain absolute. No EE columns need to be written into Parquet.

Verify data and three cameras:

```bash
JAX_PLATFORMS=cpu .venv/bin/python -m openpi.training.petrel_lerobot_reader \
  --dataset-config configs/petrel/pi05_abc_sim.yaml --index 0 --with-video
```

The reader creates metadata and path-index caches under
`data/petrel_cache/{meta,path_index}/abc_sim/train` on first use.

## 2. Normalization statistics

```bash
bash scripts/compute_abcsim_norm_stats.sh
```

By default this scans all valid train windows, with 32 workers. It uses exactly
the same FK and delta projection as training. Output:
`assets/pi05_abc_sim/abc_sim/norm_stats.json`, with group `dataset/abc_sim`.

To estimate statistics from sampled windows instead, explicitly set a limit:

```bash
OPENPI_NORM_MAX_FRAMES=1000000 OPENPI_NORM_WORKERS=16 \
  bash scripts/compute_abcsim_norm_stats.sh
```

This limit affects statistics estimation only. It does not restrict the
training dataset. Regenerate statistics whenever the projection or action
semantics or frame filter changes. The wrapper records all input hashes and
checks they stayed fixed during computation. The training wrapper checks those
hashes before launching GPU ranks. Old bottles statistics do not cover the
derived EE values.

The earlier full statistics job is paused at the user's request. Do not resume
that process: its plan used the unfiltered windows. The existing 1M-window
statistics also predate frame filtering, so production training requires a new
statistics run on the filtered windows. No statistics computation is started by
building the filter. When computation is resumed, start a fresh job using the
command above instead of continuing the old process.

## 3. Base weights and training

`src/openpi/training/config.py` registers the model and data factory. The
PyTorch base weights are copied from
`/dataset/public/model/pi05_base_pytorch_bf16` to
`data/models/pi05_base_pytorch_bf16`, so tensor loading uses a local file:

```bash
bash scripts/prepare_abcsim_weights.sh
```

`OPENPI_WEIGHT_PATH` can override the cache directory.

Inspect the launch command without starting training:

```bash
OPENPI_LAUNCH_DRY_RUN=1 OPENPI_WANDB=0 \
  MASTER_ADDR=127.0.0.1 MASTER_PORT=29500 SCO_POD_NAME=master-0 \
  OPENPI_RUN_ROOT=/sunyiming/openpi/runs/abc_sim_launch_check \
  bash scripts/train_abcsim.sh
```

After statistics are generated, run a short training check on a GPU node:

```bash
WORLD_SIZE=1 OPENPI_NUM_NODES=1 OPENPI_GPUS_PER_NODE=1 \
  OPENPI_LOCAL_BATCH_SIZE=1 OPENPI_WORKER_BATCH_SIZE=1 \
  OPENPI_GRADIENT_ACCUMULATION_STEPS=1 \
  OPENPI_STAGE_PARQUETS=0 OPENPI_COMPILE_TRANSFORMER_LAYERS=0 \
  OPENPI_NUM_WORKERS=0 OPENPI_ASYNC_QUEUE_SIZE=0 OPENPI_DATALOADER_PREWARM=0 \
  OPENPI_RUN_ROOT=/sunyiming/openpi/runs/abc_sim \
  OPENPI_RUN_ID=abc_sim_smoke OPENPI_TRAIN_STEPS=100 \
  OPENPI_SAVE_INTERVAL=100 OPENPI_WANDB=0 \
  bash scripts/train_abcsim.sh
```

Start fine-tuning in a ClusterX job with **8 nodes, 8 GPUs per node**. Run the
same command on every node:

```bash
OPENPI_RUN_ID=abc_sim_64gpu_6epoch bash scripts/train_abcsim.sh
```

The job supplies `WORLD_SIZE=8` (node count before torchrun), `MASTER_ADDR`
(the master node's reachable address), `MASTER_PORT`, and master/worker role
names in `SCO_POD_NAME` or `HOSTNAME`. The wrapper defaults to eight nodes when
the node count is not injected; it does not allocate nodes. Use the same run
ID and shared dataset, weights, tokenizer, statistics, and output paths on all
nodes. `MASTER_ADDR` and `MASTER_PORT` are required for this multi-node run.

The launcher uses `train_distributed.sh` → `train_distributed.py` →
`train_pytorch.py`. The defaults follow the ABC-130K 64-GPU profile: per-GPU
batch 13, accumulation 4, DDP + ZeRO-1, BF16, gradient checkpointing, compiled
transformer layers, 16 workers per rank, streaming accumulation, asynchronous
queue 16, prewarming, and out-of-order data loading. The physical global batch
is 832; the effective batch per optimizer step is **3328**.

The filter generated on 2026-10-23 covers all 51,349 source trajectories:
51,329 remain eligible and 20 were rejected by trend alignment. It retains
39,786,011 of 42,363,500 frames, yielding **37,999,963** horizon-30 windows
(previously 40,874,402). The reader verified the exact count and checked 128
random windows remained within clean segments; three real FK/delta-projected
samples were finite. Results are in `runs/abc_sim_frame_filter_reader_audit.json`.

One equivalent pass is **11,418.26 optimizer steps** at effective batch 3328.
The current **75,000-step budget is 6.5684 equivalent epochs** and processes
249,600,000 sampled windows. Checkpoints are saved
every 5,000 steps. `pi05_abc_sim` uses a
1,000-step warmup and cosine decay through step 75,000, retaining the existing
peak and final learning rates. Sampling is random with replacement, so the
epoch count measures sampled data volume and does not guarantee unique-window
coverage. Recompute with `stats.valid_windows` in the filter manifest whenever
filter rules, node count, or effective batch changes. With the current filtered
dataset, five to six equivalent epochs would be roughly 57,091–68,510 steps;
the user's 75,000-step setting is preserved.

For a single-GPU check use the smoke command above, which overrides both node
count and GPU count and disables compilation/staging. The wrapper disables
ZeRO-1 by default for a single rank.

Checkpoints default to:
`/data/sunyiming/openpi/abc_sim/checkpoints/pi05_abc_sim/<run_id>_<timestamp>/<step>`.
Override `OPENPI_RUN_ROOT` for another writable storage location. Normal runs
enable W&B and use the existing proxy/login setup; `OPENPI_WANDB=0` allows an
offline run without that proxy preflight.

Local Parquet staging defaults to `OPENPI_STAGE_PARQUETS=1` and uses the
`abc_sim/train` path index and `/dev/shm/openpi-abc-sim-parquets` on each node.
Provision enough shared memory for the dataset's Parquet working set and
loader queues. The current Parquet working set is 30 files totaling about
12.69 GiB; provision at least 32 GiB of `/dev/shm` per node, with additional
space if other jobs share it. Metadata/index preparation and statistics estimation should
finish before staging is enabled. `OPENPI_STAGE_PARQUETS=0` uses the dataset
mount directly. Launch dry runs skip staging.

## 4. Checkpoint evaluation

Evaluate a full-data checkpoint with GPU model inference and CPU physics:

```bash
bash scripts/eval_abcsim.sh \
  --checkpoint /path/to/pi05_abc_sim/checkpoint/100 \
  --task put_plastic_bottles_in_bin --seed 42 --device cuda \
  --out runs/abc_openpi/abc_sim_step100_seed42
```

The wrapper uses `eval_abc_bottles_openpi.py` with `--config-name pi05_abc_sim`.
`--task` chooses an ABC task; `--prompt` can override its instruction. By default
the live task instruction is prefixed with `sim_`, matching the dataset text,
including instructions changed by task randomization/runtime.

The OpenPI worker loads the checkpoint's own statistics and uses the training
reader to build live FK EE states. Before applying a predicted chunk, the
bridge adds its initial observed joint angles back to joint delta outputs.
The simulator receives absolute joint targets and absolute gripper values.

On an ABC environment with a working CUDA driver, request native MJWarp physics
and rendering by adding:

```bash
--physics-backend mjwarp --gpu-id 0 --video-every-n-actions 15
```

`abc_openpi_mjwarp.py` adapts the native single-world batch API. ABC currently
rejects MJWarp tasks with a per-step task runtime (for example conveyor picking);
use CPU physics for those tasks.

Each invocation evaluates one task and seed and writes `summary.json`,
`chunk_metrics.jsonl`, `trajectory.npz`, and a three-camera video. Choose a new
`--out` directory for each task/seed/checkpoint. Old bottles-only checkpoints
can still be evaluated through `eval_abc_bottles_openpi.sh`.

## 5. Completed verification (2026-10-22)

- 67 reader, bridge, and MJWarp adapter tests passed; Python lint and shell
  syntax checks passed.
- Real full-data reader and three-camera decoding passed on a mug-rack example.
- Statistics from 1,000,000 unfiltered training windows were generated at
  `assets/pi05_abc_sim/abc_sim/norm_stats.json`. The companion
  `norm_stats_metadata.json` records the sampling limit and projection hashes.
  These are historical statistics and must be recomputed after enabling the
  frame filter.
- Native PI0.5 base BF16 weights are cached at
  `data/models/pi05_base_pytorch_bf16`.
- Training completed 100 optimizer steps on one A100 80GB, physical batch 1,
  accumulation 1, using `pi05_abc_sim` and the prepared statistics. Losses were
  finite, and checkpoint save completed. The training log is
  `runs/abc_sim_pipeline_smoke_train.log`.
- That checkpoint completed a seed-42 bottles rollout with CUDA inference,
  MJWarp physics/rendering, live FK, and delta-to-absolute joint conversion.
  `runs/abc_openpi/abc_sim_pipeline_step100_gpu_seed42/summary.json` reports
  `ROLLOUT_OK`; video and trajectory are in the same directory. This check used
  one chunk, one executed action, and two denoising steps. It verifies execution
  and provides no success-rate estimate.

The validation checkpoint is temporary:
`/tmp/abc_sim_pipeline_smoke/checkpoints/pi05_abc_sim/pipeline_smoke_1792684374/100`.
The 75,000-step production run has not been started. The 64-GPU distributed
launcher was checked with master/worker dry runs; actual distributed execution
still needs the 8-node GPU job.

Additional parameter review on 2026-10-23 verified the 75,000-step launcher and
Python config, and checked 128 projected data samples and normalization
projection hashes. The existing 1M-window normalization sample covers all 24
task types, drawn from 1,246 trajectories. It is an estimate, not a full-data
scan. Longer production runs may benefit from full statistics or more broadly
distributed sampling.

One A100 80GB also completed four compiled forward/backward microbatches of
physical batch 13 with real three-camera inputs, BF16, and gradient
checkpointing. Peak allocated memory was about 17.58 GiB; losses and gradients
were finite. This check reused one batch and did not perform an optimizer
update, DDP communication, ZeRO-1, or 16-worker loading. Results are in
`runs/abc_sim_batch13_audit.json`.

Each node runs 8 rank processes and 128 DataLoader workers. Check CPU and RAM
allocations and video read throughput on the actual GPU job. PyTorch saves full
model and optimizer states without automatically deleting older checkpoints:
15 checkpoints are about 293 GiB based on the completed smoke checkpoint;
provision at least 350 GiB for checkpoints and temporary writes. Model weights,
optimizer state, and step count are restored on resume, while RNG and the
DataLoader position are not saved. Training does not automatically run a
validation set or simulator evaluation; evaluate saved checkpoints separately.
