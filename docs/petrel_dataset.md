# Openpi native Petrel dataset

The main data definition is `configs/petrel/pi05_petrel.yaml`, using
`schema: openpi_petrel_v1`. Datasets with identical common parameters are
grouped under `configs/petrel/dataset_groups/` using
`openpi_petrel_dataset_group_v1`.

The main file contains the openpi contract, control profiles, sampling, storage,
and category weights. It loads dataset files through:

```yaml
dataset_files:
  directory: dataset_groups
  recursive: true
  pattern: '*.yaml'
```


A group YAML stores common parameters once:

```yaml
schema: openpi_petrel_dataset_group_v1
name: agibotworld
category: agibotworld
profile: agibot_a2d_end_pose
format: v2
root: manip:manip-dataset/public/real/agibotworld
fields:
  video:
    keys: [observation.images.head, observation.images.hand_left, observation.images.hand_right]
datasets:
  - name: task_327
    path: task_327
  - name: task_351
    path: task_351
```

Every item under `datasets` is restricted to exactly `name` and `path`. OXE has
several camera-field variants, so it is split into separate variant group files
rather than mixing incompatible common parameters.

## Configuration ownership

`config.data.create(config.assets_dirs, config.model)` combines two independent
configuration layers:

- `config.assets_dirs` and `config.model` come from openpi's Python
  `TrainConfig`. They determine the checkpoint-compatible model shape and where
  openpi loads/saves training assets such as normalization statistics.
- `pi05_petrel.yaml` describes remote datasets, fields, cameras, controls,
  sampling, and mixture weights. It must not contain `assets_dirs` or a Python
  model object.

During `PetrelDataConfig.create`, openpi validates that YAML `dataset_id` matches
`repo_id`, and that YAML action dimension and horizon match the selected model.
The dataloader validates the state dimension and normalization mode again before
opening remote data.

The current YAML contract is:

```text
dataset_id: petrel_mixed
images: 3 RGB uint8 slots, resized by openpi to 224x224
state: [32]
actions: [16, 32]
normalization: quantile
control profiles: 9
logical source groups: 50
dataset group YAML files: 58
remote datasets after expansion: 1785
```

## Normalization statistics

Run:

```bash
python scripts/compute_norm_stats.py \
  --config-name pi05_petrel \
  --max-frames 1000000
```

The output is:

```text
assets/pi05_petrel/petrel_mixed/norm_stats.json
```

Its JSON structure is:

```json
{
  "norm_stats": {
    "state": {
      "mean": [32 values],
      "std": [32 values],
      "q01": [32 values],
      "q99": [32 values]
    },
    "actions": {
      "mean": [32 values],
      "std": [32 values],
      "q01": [32 values],
      "q99": [32 values]
    }
  }
}
```

For state, each sampled training window contributes one 32D vector. For actions,
every action timestep contributes one vector, so one sample contributes 16 rows
to action statistics. `mean/std` are streaming statistics; `q01/q99` are
estimated with 5000-bin running histograms.

For multiple datasets, the mixture sampler first selects samples according to
`mixture.category_weights`. A category weight is distributed among its child
datasets according to valid training-window counts. All selected samples update
one global state statistic and one global action statistic; the standard openpi
format does not store per-dataset statistics.

For a single dataset, the algorithm and stats format are identical, but every
sample comes from that dataset. The main config references a group directory containing one YAML file with a
single `datasets: [{name, path}]` entry, together with one control profile and
one category weight.

### InternData-A1-only statistics

pi05_interna1 loads only configs/petrel/dataset_groups/intern_a1/. It still
produces one global statistic for the whole category: Franka, Lift2, Split
Aloha, Genie1, and AC-One samples are projected into the common OpenPI 32D
state/action space and update the same state and actions RunningStats.

Run:

    python scripts/compute_norm_stats.py --config-name pi05_interna1 --max-frames 1000000

Petrel normalization automatically uses the episode-vectorized fast path:
Parquet is read once per selected episode, windows are expanded in batches of
4096, and worker processes merge streaming mean/std plus reservoir q01/q99.
When max-frames is set, deterministic episode blocks are sampled until the
global cap is reached; without max-frames every valid window is processed.

The default Petrel fast-path options are 32 workers, 4096 windows per episode
batch, and a 200000-row quantile reservoir. They can be overridden with
`--petrel-num-workers`, `--petrel-episode-batch-size`, and
`--petrel-reservoir-rows`.

The output is assets/pi05_interna1/intern_a1/norm_stats.json. Train with the
same config so that OpenPI loads exactly this asset:

    python scripts/train.py pi05_interna1 --exp-name=interna1_pi05

## Training

```bash
python scripts/train.py pi05_petrel --exp-name=petrel_pi05
# or
python scripts/train_pytorch.py pi05_petrel --exp-name=petrel_pi05
```

The reader returns RGB `uint8` images. Openpi's existing model transforms resize
them to 224x224, tokenize the prompt, and apply quantile normalization from the
file above. Remote transport still requires `petrel_client` and the Petrel
credentials configured by `storage.petrel_conf`.
