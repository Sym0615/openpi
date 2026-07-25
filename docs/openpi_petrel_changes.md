# OpenPI Petrel 训练分支改动说明

本文档说明当前仓库相较于克隆时的官方 OpenPI 版本做了哪些修改，以及这些修改如何支持从 Petrel 远程读取多机器人 LeRobot 数据、计算归一化统计量并训练 PI0.5。

## 1. 对比基线和改动规模

- 官方基线 commit：`15a9616a00943ada6c20a0f158e3adb39df2ccac`
- 当前功能 commit：`4ec612a7d7fc616d9e54a1d0f43e6a209218c257`（`openpi petreloss training`）
- 中文训练说明 commit：`16185ef9c2189b24266c35130bc465714670d758`（`readme zh`）
- 总计：72 个文件发生变化，约 11,787 行新增、40 行删除。

改动的核心目标是：不再依赖 WAM 仓库中的 Python/YAML 文件，仅保留 Petrel 远程对象存储访问方式，在 OpenPI 仓库内部完成数据定义、数据读取、机器人控制空间转换、混合采样、图像解码、norm stats 计算以及 PI0.5 训练接入。

## 2. 总体数据流

```text
configs/petrel/*.yaml
        |
        v
解析主配置、extends 和 dataset_groups
        |
        v
DatasetSpec（远程路径、版本、字段、相机、控制 profile、权重）
        |
        v
PetrelLeRobotDataset（读取 metadata / parquet / mp4）
        |
        v
MixedPetrelLeRobotDataset（数据集、episode、窗口混合采样）
        |
        v
PetrelDataset（映射为 OpenPI state/actions/images/prompt）
        |
        v
OpenPI transforms（224x224、tokenizer、quantile norm、padding）
        |
        v
JAX 或 PyTorch PI0.5 训练入口
```

norm stats 走独立的无图像快速路径：

```text
YAML -> episode 计划 -> 每个 episode 一次读取 parquet
     -> 批量构造窗口 -> 32D 投影 -> 多进程统计合并
     -> assets/<config>/<dataset_id>/norm_stats.json
```

## 3. 新增的远程数据读取实现

### 3.1 `src/openpi/training/petrel_lerobot_reader.py`

这是底层远程 LeRobot reader，也是本次最大的新文件。它不导入 WAM 代码，主要能力如下。

#### Petrel/S3 访问

- 从 `~/petreloss.conf` 或 YAML 的 `storage.petrel_conf` 读取访问配置。
- 默认优先使用 `petrel_client.client.Client`。
- 当 `petrel_client` 不可用时，可独立使用 `boto3` 访问 S3-compatible endpoint。
- 设置 `OPENPI_PETREL_BACKEND=petrel` 时强制使用 `petrel_client`，初始化失败会直接报错，不再静默 fallback。
- Petrel client 按配置路径缓存，避免同一进程重复创建。

#### LeRobot v2/v3 数据支持

- 读取远程 metadata、episodes、tasks、Parquet 和 MP4。
- 兼容 LeRobot v2 与 v3 的文件组织和 episode 定位方式。
- 支持 metadata cache、path-index cache 和有限大小的 Parquet cache。
- 支持远程 MP4 字节解码，并按时间戳抽取所需帧。
- 支持损坏视频重试以及缺失相机处理。

#### 多机器人控制空间适配

- 通过各 dataset-group YAML 中的 `control_profiles` 定义该类机器人的 state/action 字段。
- 支持 joint、末端位姿、gripper 和 hand 等组件。
- 支持 quaternion、RPY、rotation-vector、rotation-6D 等姿态表示转换。
- 支持绝对目标动作转 delta 动作。
- 支持坐标系和末端执行器 frame transform。
- 用 dimension mask 标记缺失或 padding 的控制维度。

#### 混合采样

`MixedPetrelLeRobotDataset` 先选择数据集，再选择 episode，最后选择合法起始窗口：

- category 权重由主 YAML 提供。
- category 权重可按子数据集有效窗口数分配。
- episode 可按有效窗口数量平衡。
- 相同随机 seed 和 epoch 下采样可复现。
- `training_block_size` 让相邻 DataLoader index 对应同一局部 episode block，减少 Parquet/视频重复读取。
- 支持训练/评估范围和排除固定评估 episode/sample 的基础逻辑。

#### norm stats 专用接口

新增 `iter_episode_stat_batches()`：

- 一个 episode 的 Parquet 通常只加载一次。
- 不读取图像、不创建 prompt、不执行 tokenizer。
- 批量构造 state/action 窗口，避免训练式逐 sample 重复访问同一 episode。
- 可接收指定的 `start_indices`，用于全局 `max_frames` 下的确定性 block 抽样。

### 3.2 `src/openpi/training/petrel_dataset.py`

这是底层 reader 与 OpenPI DataLoader 之间的适配层。

新增内容包括：

- `PetrelDatasetConfig`：定义 YAML 路径、Petrel 配置覆盖、prompt 模板、图像开关、坏视频重试等运行参数。
- `PetrelDataset`：将 reader 输出转换为 OpenPI 原生样本 schema。
- `read_petrel_dataset_metadata()`：只读本地 YAML contract，不访问远端数据。
- `project_petrel_features()`：把不同机器人的 canonical 控制向量压缩/补零到 PI0.5 的固定 32D 空间。

OpenPI 输出约定为：

```text
state:       [32]
actions:     [16, 32]
prompt:      str
image:
  base_0_rgb:        [H, W, 3] uint8
  left_wrist_0_rgb:  [H, W, 3] uint8
  right_wrist_0_rgb: [H, W, 3] uint8
image_mask:  每个相机是否真实存在
```

32D compact projection 的规则是：

1. 优先保留所有有效 action 维度。
2. 剩余空间可加入只存在于 state 的维度。
3. 不足 32D 的尾部补零。
4. padding mask 对应的 state/action 数值强制置零。
5. 如果某数据集有效 action 维度超过 32，直接报错，避免静默截断训练目标。

图像侧新增批量读取优化：同一个 batch 中属于同一 dataset/episode 的样本会合并 MP4 请求并批量解码；失败时再退回逐样本重试。缺失相机使用零图填充，并通过 `image_mask` 告知模型。

## 4. OpenPI 原有配置系统的修改

修改文件：`src/openpi/training/config.py`。

### 4.1 `DataConfig` 扩展

新增：

```python
petrel: PetrelDatasetConfig | None = None
```

因此 Petrel 是 OpenPI 原生 `DataConfig` 的一个可选 backend，而不是训练脚本之外的旁路 loader。

### 4.2 新增 `PetrelDataConfig`

`PetrelDataConfig.create(config.assets_dirs, config.model)` 继续遵守官方 OpenPI 的 config factory 接口。它负责：

- 读取 YAML 中的 `dataset_id` 和 `openpi` contract。
- 校验 YAML `dataset_id` 与 Python `repo_id` 一致。
- 校验 YAML `action_dim/action_horizon` 与模型配置一致。
- 使用 OpenPI 原有 `ModelTransformFactory`，因此图像 resize、PaliGemma tokenizer、state/action padding 和 norm 都继续复用官方逻辑。
- 将 Petrel runtime config 放入最终 `DataConfig`。

YAML 不需要也不应该定义 `assets_dirs` 或 Python model。它们仍来自官方 `TrainConfig`。

### 4.3 新增训练配置

新增两个可由 CLI 直接选择的配置：

- `pi05_petrel`
  - 混合读取全部 Petrel 数据集。
  - PI0.5，`action_dim=32`，`action_horizon=16`。
  - `repo_id=petrel_mixed`。
- `pi05_interna1`
  - 最小闭环配置，只读取 InternData-A1。
  - PI0.5，`action_dim=32`，`action_horizon=16`。
  - `repo_id=intern_a1`。
  - Franka、Lift2、Split Aloha、Genie1、AC-One 共同使用一份全局 norm stats。

两个配置当前均设置 `num_workers=0`。远程数据 locality 主要由 block sampling、reader cache 和 batch decode 保证；这也避免每个 DDP rank 再创建多个进程后导致 Petrel client/缓存成倍复制。

## 5. OpenPI DataLoader 的修改

修改文件：`src/openpi/training/data_loader.py`。

### 5.1 Petrel dataset 路由

`create_torch_dataset()` 在 `data_config.petrel` 非空时构造 `PetrelDataset`，否则保留官方 fake/LeRobot/RLDS 路径。

### 5.2 batch 级读取

`TransformedDataset` 新增 `__getitems__()` 转发，使 PyTorch DataLoader 可以调用 Petrel dataset 的 batch fetch，实现同 episode 视频合并解码。

### 5.3 epoch 传播

新增 `set_epoch()` 转发，保证混合数据集和 DistributedSampler 在每个 epoch 使用一致但不同的确定性随机序列。

### 5.4 保留远程访问 locality

当 dataset 声明 `prefer_sequential_indices=True` 时，关闭外层随机 shuffle。随机性仍由内部 mixture sampler 提供，但 DataLoader index 保持顺序，从而使 block 内样本集中访问同一 episode。

### 5.5 PyTorch DDP 支持修正

- `framework="pytorch"` 时使用 `DistributedSampler`。
- global batch size 按 DDP world size 除成每 rank local batch size。
- PyTorch 路径不再调用 `jax.process_count()` 或初始化 JAX CUDA backend。
- JAX 路径仍保留官方 sharding 行为。

### 5.6 worker 生命周期

- 多 worker 使用 `spawn` context。
- dataset 可声明每个 epoch 重建 worker，避免远程 client/cache 状态跨 epoch 不一致。
- `TorchDataLoader.__iter__()` 同时更新 dataset epoch 和 sampler epoch。

## 6. Petrel YAML 配置体系

新增目录：`configs/petrel/`。

### 6.1 主配置 `pi05_petrel.yaml`

schema 为 `openpi_petrel_v1`，集中定义：

- OpenPI tensor contract：32D state/action、16-step horizon、3 个 RGB 相机槽、224x224、quantile norm。
- Petrel 配置路径和本地 cache 模板。
- state/action 时间窗口、stride、末尾 padding 策略和训练 block 大小。
- category mixture 权重与随机 seed。
- 原始机器人控制空间描述。
- dataset group 文件扫描规则。

当前全量配置展开后包含约 50 个逻辑 source group、58 个 dataset-group YAML 和 1,785 个远程数据集条目。

### 6.2 InternA1 配置 `pi05_interna1.yaml`

通过：

```yaml
extends: pi05_petrel.yaml
```

继承全部 OpenPI contract、storage 和 sampling 配置，仅覆盖：

- `dataset_id: intern_a1`
- category 权重只保留 `intern_a1: 1.0`
- dataset 文件扫描范围只指向 `dataset_groups/intern_a1/`

这样避免复制数百行公共配置。

### 6.3 dataset-group YAML

新增：

- `agibotworld/`
- `bridgev2/`
- `galaxea_no_mobile_torso/`
- `intern_a1/`
- `intern_m1/`
- `oxe/`

每个 group 文件 schema 为 `openpi_petrel_dataset_group_v1`。每个文件通过
`profile` 选择控制类型，并在本文件的 `control_profiles` 中定义且只定义
该 profile 的字段、切片、姿态语义和坐标变换。相同 root、format、profile
和字段映射的数据集共用一个 YAML；具体数据集列表中每项通常只有：

```yaml
- name: dataset_name
  path: remote_relative_path
```

OXE 因相机字段不同拆成多个 variant 文件，避免把不兼容字段混在一个 group 中。

## 7. norm stats 计算的修改

### 7.1 `scripts/compute_norm_stats.py`

官方脚本被扩展为自动识别 Petrel backend。

Petrel 路径的主要变化：

- 计算 norm 时用 `IdentityTransformFactory` 跳过 model transforms。
- 因此不会初始化 PaliGemma tokenizer，也不会为了 norm stats 去 Google Cloud 下载 `paligemma_tokenizer.model`。
- 自动调用 Petrel episode-vectorized 快速统计实现。
- 新增参数：
  - `--petrel-num-workers`，默认 32。
  - `--petrel-episode-batch-size`，默认 4096。
  - `--petrel-reservoir-rows`，默认 200000。
  - `--output-dir`，可覆盖保存根目录。
- 非 Petrel 的官方 LeRobot/RLDS 统计路径继续保留。

### 7.2 `src/openpi/training/petrel_norm_stats.py`

新增的快速统计器参考 episode 顺序读取思想，但输出严格使用 OpenPI norm stats 格式。

算法：

1. 枚举每个 dataset/episode 的合法训练窗口数。
2. 未设置 `max_frames` 时处理全部合法窗口。
3. 设置 `max_frames` 时，以确定性随机 episode block 选满全局窗口上限。
4. 将 episode 计划分配给多个 worker。
5. 每个 worker 按 episode 读取 Parquet，批量生成窗口并调用与训练相同的 32D projection。
6. 使用可合并的 streaming count/sum/sum-square 计算 mean/std。
7. 使用 reservoir sampling 估计 q01/q99。
8. 主进程合并 worker 统计量并保存为 OpenPI `NormStats`。

输出格式：

```json
{
  "norm_stats": {
    "state":   {"mean": [], "std": [], "q01": [], "q99": []},
    "actions": {"mean": [], "std": [], "q01": [], "q99": []}
  }
}
```

其中：

- state 每个训练窗口贡献一个 32D row。
- actions 每个训练窗口贡献 16 个 32D row。
- 多数据集配置最终仍生成一组全局 state stats 和一组全局 action stats，不按机器人分别保存。
- `pi05_interna1` 会把 InternA1 中所有机器人投影到共同 32D 空间后混合统计。

保存位置：

```text
assets/pi05_petrel/petrel_mixed/norm_stats.json
assets/pi05_interna1/intern_a1/norm_stats.json
```

## 8. 依赖修改

修改文件：`pyproject.toml` 和 `uv.lock`。

为远程 reader 新增或固定了以下运行依赖：

- `av>=16.0.0`：远程 MP4 字节流解码。
- `pyarrow>=20.0.0`：Parquet metadata 与数据读取。
- `pyyaml>=6.0.2`：仓库内 YAML 配置解析。
- `boto3>=1.35.0`：`petrel_client` 不可用时的 S3-compatible fallback。

`uv.lock` 已同步依赖解析结果，包括 boto3/botocore/s3transfer 等间接依赖。

`petrel_client` 本身仍由集群环境/SDK 提供，不从公共 PyPI 强制安装。集群上曾通过为容器补充 Ubuntu 22 的兼容 `libcurl.so.4` 解决其 native library 加载问题；该库目录是 workspace 本地运行资产，不属于 Git 仓库内容。

## 9. 测试新增

新增 `src/openpi/training/petrel_dataset_test.py`，覆盖：

- OpenPI 样本 schema 和 shape。
- norm 模式跳过视频解码。
- batch 视频解码。
- compact 32D projection 与 padding mask。
- Petrel S3 URL 解析。
- 主 YAML 和 dataset-group schema 翻译。
- 全量 YAML 数据集展开。
- `pi05_interna1` 只包含 InternA1。
- YAML contract 与 Python model/config 一致性检查。
- norm stats 不创建 tokenizer。
- 多数据集统计合并为标准全局格式。
- PyTorch DataLoader 不初始化 JAX backend。
- batch/epoch 方法转发。
- 快速统计 accumulator 和全局 block cap。

## 10. 文档和忽略规则

### 新增文档

- `docs/petrel_dataset.md`：Petrel 配置、数据 contract、norm stats 和训练命令说明。
- `README_ZH.md`：当前集群环境下 PI0.5 + Petrel 的中文训练示例和待办事项。
- 本文档：相对官方基线的完整改动清单。

### `.gitignore`

新增或明确忽略本地运行产物：

```text
assets/
checkpoints/
data/
wandb/
logs/
jobs/
*.tmp
```

这不会删除本地文件。特别是训练需要的 `norm_stats.json` 仍保留在本机并可被读取，但不会随 Git 提交。换机器或重新 clone 后必须重新计算或复制 norm stats。

`configs/`、`scripts/`、`src/openpi/training/` 和依赖锁文件没有被忽略，属于可复现训练所需的代码和配置。

## 11. 已验证的运行闭环

当前代码已完成以下实际验证：

- 使用 Petrel 读取 InternA1 metadata、Parquet 和 MP4。
- 使用 episode-vectorized 路径计算 1,000,000 个 InternA1 窗口的 norm stats。
- 生成 `assets/pi05_interna1/intern_a1/norm_stats.json`。
- 使用本地 PI0.5 PyTorch bf16 权重启动 8-GPU DDP。
- 每卡 local batch size 1、global batch size 8，完成 forward、backward、optimizer step 和 checkpoint 保存。
- W&B 在线记录 loss、learning rate、grad norm 和 time per step。

训练使用的 PaliGemma tokenizer 是模型文本输入所需的 SentencePiece 模型，不是 PI0.5 权重。它被放在 OpenPI 本地 download cache 中；norm stats 快速路径不会加载它。

## 12. 与官方代码保持不变的部分

- PI0/PI0.5 模型主体和 loss 定义未因 Petrel 接入而重写。
- PaliGemma/Gemma/SigLIP 模型组件仍使用原有实现。
- OpenPI transforms、quantile normalize/unnormalize 和 checkpoint asset 结构继续复用。
- 官方 JAX `scripts/train.py` 入口仍可使用。
- 官方 PyTorch `scripts/train_pytorch.py` 仍是手写 PyTorch + DDP 训练循环，不是 Hugging Face Trainer，也没有接入 DeepSpeed。
- 非 Petrel 的 LeRobot、RLDS/DROID 和 fake dataset 路径仍保留。

## 13. 当前限制和注意事项

1. **norm stats 是全局统计。** 一个 config/`repo_id` 只对应一组 stats；InternA1 内不同机器人不会分别归一化。
2. **固定 32D contract。** 当前 PI0.5 checkpoint 和 YAML 按 32D 配置；超过 32 个有效 action 维度的数据会报错。
3. **固定三相机槽。** 多余相机目前不会进入模型，缺少的槽位用零图和 mask 表示。
4. **norm stats 未纳入 Git。** 新环境必须准备对应 JSON，否则训练初始化会失败。
5. **远程 SDK 依赖环境。** 强制 `petrel_client` 时需保证 SDK、`petreloss.conf` 和兼容 native library 在容器内可见。
6. **PyTorch 当前使用 DDP。** 没有 DeepSpeed ZeRO/FSDP optimizer-state sharding；模型与 optimizer checkpoint 较大。
7. **训练作业时限需匹配步数。** 8 卡、global batch 8 时当前实测约 4--5 秒/step，20,000 steps 需要远大于 2 小时的 Slurm time limit。
8. **`num_workers=0` 是当前保守设置。** 数据读取性能主要依赖 block locality 和 batch decode；进一步增加 worker 需要重新评估每 rank 的网络连接数、缓存复制和内存占用。

## 14. 常用命令

计算 InternA1 norm stats：

```bash
python scripts/compute_norm_stats.py \
    --config-name pi05_interna1 \
    --max-frames 1000000
```

单进程 PyTorch 调试：

```bash
python scripts/train_pytorch.py pi05_interna1 \
    --exp_name interna1_debug \
    --pytorch_weight_path /path/to/pi05_base_pytorch_bf16
```

单机 8-GPU DDP：

```bash
torchrun --standalone --nnodes=1 --nproc-per-node=8 \
    scripts/train_pytorch.py pi05_interna1 \
    --exp_name interna1_train \
    --pytorch_weight_path /path/to/pi05_base_pytorch_bf16 \
    --batch_size 8
```

恢复训练时必须沿用原 `exp_name`，使用 `--resume`，且不能同时使用 `--overwrite`。
