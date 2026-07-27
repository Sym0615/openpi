# InternA1 单卡训练 Profiling 修改报告

## 目标

本次修改为 `pi05_interna1` 的单卡 PyTorch 训练增加可选 profiling，用来观察：

- 数据 access 和 pre-loading：路径读取、Parquet 读取与转换、视频读取和解码。
- preprocessing：数据 mapping、normalize、图像处理、tokenize 和 padding 等 transform 总耗时。
- DataLoader 等待时间和端到端样本吞吐。
- H2D、forward、backward、optimizer 的 GPU 时间。
- GPU 利用率、显存利用率、显存占用、功耗和温度。
- 可选的 PyTorch CPU/CUDA trace。

所有 profiling 功能默认关闭，不影响普通训练配置。

## 修改内容

### 1. 新增统一 profiling 模块

文件：`src/openpi/training/profiling.py`

- 新增线程安全的事件聚合器，只保留 count、total、average、max、bytes 和 MB/s，不保存每条样本事件。
- 新增 `GpuMonitor`，后台按固定周期调用 `nvidia-smi`。
- GPU 原始采样持续写入 `gpu_metrics.csv`。
- 支持按日志窗口读取并清空聚合统计，控制长时间训练的内存占用。
- 通过 `OPENPI_PROFILE` 和训练配置统一控制是否启用。

### 2. 接入底层数据读取事件

文件：`src/openpi/training/petrel_lerobot_reader.py`

原有 `_profile_record()` 除了支持 reader CLI 的逐事件输出，现在还会写入统一聚合器。训练期间可以统计现有事件，包括：

- `local.get_bytes`、`local_meta.read_bytes`
- `petrel.get_bytes`、`petrel.boto3_get_bytes`
- `parquet.open`、`parquet.read_table`、`parquet.to_tensors`
- `video.remote_bytes`、`video.tempfile_write`、`video.decode`
- tasks、episodes、path index 和 dataset 初始化事件

带字节数的事件会额外计算读取吞吐 MB/s。

### 3. 区分数据装载和 preprocessing

文件：`src/openpi/training/data_loader.py`

在 `TransformedDataset.__getitem__()` 和 `__getitems__()` 中增加两个边界：

- `data.access_and_preload`：底层 dataset 返回原始样本所需时间。
- `data.preprocessing`：OpenPI transform 链的总时间。

第二项覆盖 InternA1 状态/action mapping、normalize、图像 transform、tokenize 和 padding。底层 reader 的详细事件可继续将第一项拆成 Parquet、视频、缓存等部分。

### 4. 拆分训练 step

文件：`scripts/train_pytorch.py`

训练循环改为显式调用 `next(data_iter)`，增加：

- `data_wait_ms`
- `h2d_ms`
- `forward_ms`
- `backward_ms`
- `optimizer_ms`
- `step_ms`
- `samples_per_second`

CUDA 阶段使用 `torch.cuda.Event` 计时，并在每个 profiling step 结束统一同步，避免把 CUDA 异步提交时间误当成实际执行时间。普通训练关闭 profiling 时不会执行 CUDA 同步。

前 `profiling_warmup_steps` 个 step 只预热；到达边界后会清空数据、GPU 和训练阶段累计值，避免冷启动污染正式结果。

### 5. GPU 实时监控

文件：`scripts/train_pytorch.py`、`src/openpi/training/profiling.py`

主训练进程启动后台 GPU sampler，采集：

- GPU utilization
- memory utilization
- used/total memory
- power draw
- GPU temperature

采样 GPU 会根据 `CUDA_VISIBLE_DEVICES` 选择，避免 `CUDA_VISIBLE_DEVICES=3` 时错误监控物理 GPU 0。如果系统没有 `nvidia-smi`，训练继续执行并给出 warning。

### 6. 可选 PyTorch trace

文件：`scripts/train_pytorch.py`

设置 `OPENPI_TORCH_TRACE=1` 后，使用 `torch.profiler` 在预热结束后记录一个短 CPU/CUDA trace，包含 shape 和显存信息。Trace 默认关闭，因为它的开销高于普通阶段计时。

### 7. 单卡脚本参数化

文件：`scripts/train_single.py`、`scripts/train_single.sh`

单卡脚本新增环境变量：

| 环境变量 | 默认值 | 含义 |
|---|---:|---|
| `OPENPI_PROFILE` | `0` | 是否启用 profiling |
| `OPENPI_TRAIN_STEPS` | profile 时 `120` | 总训练步数 |
| `OPENPI_BATCH_SIZE` | `64` | batch size |
| `OPENPI_NUM_WORKERS` | `0` | DataLoader worker 数 |
| `OPENPI_PROFILE_WARMUP_STEPS` | `20` | 不进入统计的预热步数 |
| `OPENPI_PROFILE_LOG_INTERVAL` | `10` | profiling 汇总周期 |
| `OPENPI_GPU_MONITOR_INTERVAL` | `1.0` | GPU 采样周期，单位秒 |
| `OPENPI_TORCH_TRACE` | `0` | 是否生成 PyTorch trace |
| `OPENPI_TRACE_ACTIVE_STEPS` | `20` | trace active step 数 |

profiling run 的实验名使用 `interna1_pytorch_profile_<run_id>`，与普通 smoke run 隔离。

## 运行命令

先使用 `num_workers=0` 做精确阶段归因：

```bash
OPENPI_PROFILE=1 \
OPENPI_RUN_ID=profile_w0 \
OPENPI_NUM_WORKERS=0 \
OPENPI_TRAIN_STEPS=120 \
bash scripts/train_single.sh
```

生成 PyTorch trace：

```bash
OPENPI_PROFILE=1 \
OPENPI_TORCH_TRACE=1 \
OPENPI_RUN_ID=profile_trace \
OPENPI_NUM_WORKERS=0 \
bash scripts/train_single.sh
```

测试实际 DataLoader 并发吞吐时分别运行：

```bash
OPENPI_PROFILE=1 OPENPI_RUN_ID=profile_w2 OPENPI_NUM_WORKERS=2 bash scripts/train_single.sh
OPENPI_PROFILE=1 OPENPI_RUN_ID=profile_w4 OPENPI_NUM_WORKERS=4 bash scripts/train_single.sh
OPENPI_PROFILE=1 OPENPI_RUN_ID=profile_w8 OPENPI_NUM_WORKERS=8 bash scripts/train_single.sh
```

`num_workers>0` 时 worker 内的详细聚合器属于各自进程，因此主进程报告重点是 `data_wait_ms` 和端到端 samples/s。精确拆分 access/pre-loading/preprocessing 时应使用 `num_workers=0`。

## 输出文件

输出目录：

```text
<checkpoint_dir>/profiling/
```

包含：

- `step_metrics.jsonl`：训练阶段耗时、吞吐和窗口 GPU 汇总。
- `data_metrics.jsonl`：数据读取、解码、pre-loading 和 preprocessing 聚合事件。
- `gpu_metrics.csv`：每次 GPU 原始采样。
- `torch_trace/`：仅在启用 `OPENPI_TORCH_TRACE=1` 时生成。

控制台每个 profiling 窗口也会输出 samples/s、各训练阶段耗时和平均 GPU 利用率。

## 测试建议

建议固定 batch size 和数据顺序，对 `num_workers=0/2/4/8` 分别运行。`num_workers=0` 用来定位具体阶段，多个 worker 用来判断实际流水线能否覆盖 GPU 计算。不要把启用 PyTorch trace 的结果作为最终吞吐结果，因为 trace 会引入额外开销。

## 代码级详细修改说明

下面按照实际训练执行顺序，说明本次具体修改了哪些代码。

### 1. `src/openpi/training/profiling.py`：新增 profiling 基础设施

这是本次新增的文件，负责两类功能：

1. 聚合数据读取和预处理事件。
2. 在后台持续采样 GPU 状态。

#### 1.1 全局开关

模块加载时读取：

```python
_ENABLED = os.environ.get("OPENPI_PROFILE", "0") == "1"
```

训练开始时还可以调用：

```python
set_enabled(enabled=config.profiling_enabled)
```

`set_enabled()` 同时更新当前进程中的 `_ENABLED` 和环境变量 `OPENPI_PROFILE`。后者用于让采用
`spawn` 启动的 DataLoader worker 继承 profiling 开关。

profiling 默认关闭，因此普通训练不会累计事件，也不会启动 GPU monitor。

#### 1.2 `_Counter` 聚合结构

每类事件只保存：

```python
count
total_s
max_s
total_bytes
```

没有保存每个样本的完整事件和路径，避免训练时间增长时 profiling 内存也不断增长。

#### 1.3 `record()`

事件结束后调用：

```python
record(name, elapsed_s, num_bytes=...)
```

该函数在线程锁保护下累计：

```python
counter.count += 1
counter.total_s += elapsed_s
counter.max_s = max(counter.max_s, elapsed_s)
counter.total_bytes += num_bytes
```

关闭 profiling 时函数立即返回。

#### 1.4 `snapshot()`

`snapshot()` 将内部 counter 转换为：

```text
count
total_ms
average_ms
max_ms
total_bytes
throughput_mb_s
```

吞吐计算方式为：

```python
throughput_mb_s = total_bytes / 1e6 / total_s
```

训练使用：

```python
snapshot(reset=True)
```

每个统计窗口写入文件后清空聚合值，使下一条记录只表示下一个窗口。

#### 1.5 `GpuMonitor`

`GpuMonitor.start()` 创建 daemon thread。线程按照 `gpu_monitor_interval_s` 周期执行：

```bash
nvidia-smi \
  --query-gpu=index,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu \
  --format=csv,noheader,nounits \
  -i <gpu>
```

每次原始采样立即写入 `gpu_metrics.csv` 并 flush，因此训练过程中也可以实时查看文件。

`summary()` 提供当前窗口的：

```text
gpu_util_avg
gpu_util_max
memory_util_avg
memory_used_mib_max
power_w_avg
temperature_c_max
```

如果机器上没有 `nvidia-smi`，GPU monitor 会给出 warning，但不会中止训练。

### 2. `src/openpi/training/petrel_lerobot_reader.py`：接入底层 I/O 事件

原来的 reader 已经在文件读取、Parquet 和视频解码位置调用 `_profile_record()`，但这些数据主要供
reader 独立 CLI 的 `--profile` 使用。

本次在 `_profile_record()` 开头增加：

```python
_training_profiling.record(
    name,
    elapsed_s,
    num_bytes=info.get("bytes"),
)
```

因此训练模式可以直接复用现有底层事件，而不需要重新包裹所有读取函数。

训练期间可能出现的主要事件包括：

| 事件 | 说明 |
|---|---|
| `local.get_bytes` | 从 `/manip` 路径读取数据对象 |
| `local_meta.read_bytes` | 从 `data/petrel_cache` 读取 metadata |
| `petrel.get_bytes` | 通过 Petrel SDK 读取对象 |
| `petrel.boto3_get_bytes` | 通过 boto3 fallback 读取对象 |
| `parquet.open` | 打开 Parquet 数据 |
| `parquet.read_table` | 将 Parquet 读取为 Arrow table |
| `parquet.to_tensors` | Arrow/Parquet 数据转换为 tensor |
| `video.remote_bytes` | 读取 episode 视频字节 |
| `video.tempfile_write` | 将视频字节写入临时文件 |
| `video.import_av` | 初始化视频解码依赖 |
| `video.decode` | 解码指定时间戳的视频帧 |
| `dataset.path_index_hit` | 命中已有 path index |
| `dataset.read_info` | 读取数据集 info metadata |
| `dataset.read_tasks` | 读取任务 metadata |
| `tasks.read_parquet` | 读取 v3 tasks parquet |
| `episodes.read_parquet` | 读取 v3 episodes parquet |

携带 `bytes` 的事件会在 `data_metrics.jsonl` 中包含 MB/s。没有字节信息的事件只统计次数和耗时。

原 reader CLI 的 `_PROFILE_ENABLED` 和 `_PROFILE_EVENTS` 逻辑仍然保留，因此没有破坏独立 reader
profiling 的原有行为。

### 3. `src/openpi/training/data_loader.py`：增加两层数据计时边界

修改类：

```python
TransformedDataset
```

修改方法：

```python
__getitem__()
__getitems__()
```

#### 3.1 `data.access_and_preload`

单样本路径从：

```python
return self._transform(self._dataset[index])
```

拆成：

```python
started = time.perf_counter()
item = self._dataset[index]
_profiling.record(
    "data.access_and_preload",
    time.perf_counter() - started,
)
```

对于 InternA1，这个边界覆盖底层 dataset 返回一个可供 OpenPI transform 使用的样本所需的全部时间，
包括：

- dataset/episode 定位。
- metadata 和 path index 查询。
- 从 `/manip` 或 SDK 读取 Parquet、MP4。
- Parquet 解析和 tensor 转换。
- 视频帧读取和解码。
- `PetrelDataset._pack_item()`。
- canonical 控制量投影，例如 InternA1 的 80D 到 ALOHA 14D 投影。

所以 `data.access_and_preload` 是 access 与 pre-loading 的上层总耗时。具体的文件读取、Parquet、视频
解码耗时需要结合 reader 产生的细粒度事件分析。

#### 3.2 `data.preprocessing`

底层样本返回后，再单独统计：

```python
started = time.perf_counter()
result = self._transform(item)
_profiling.record(
    "data.preprocessing",
    time.perf_counter() - started,
)
```

这个边界覆盖外层 OpenPI transform 链，主要包括：

- repack transforms。
- policy input transforms。
- quantile/z-score normalize。
- 图像 resize 和格式转换。
- prompt tokenize。
- model transforms。
- `PadStatesAndActions(32)`。

对于 InternA1，80D 到 ALOHA 14D 的 projection 在 `PetrelDataset._pack_item()` 内，因此计入
`data.access_and_preload`；14D normalize 和最终 14D 到 32D padding 计入 `data.preprocessing`。

#### 3.3 batch 路径

PyTorch DataLoader 支持通过 `__getitems__()` 一次提交一组索引。本次也在这个路径增加了相同的两段
计时，避免 batch fetch 绕过 profiling：

```python
items = getitems(indices)
record("data.access_and_preload", ...)

result = [self._transform(item) for item in items]
record("data.preprocessing", ...)
```

### 4. `src/openpi/training/config.py`：增加 profiling 配置字段

在 `TrainConfig` 中新增：

```python
profiling_enabled: bool = False
profiling_warmup_steps: int = 20
profiling_log_interval: int = 10
profiling_torch_trace: bool = False
profiling_trace_active_steps: int = 20
gpu_monitor_interval_s: float = 1.0
```

字段含义：

| 字段 | 作用 |
|---|---|
| `profiling_enabled` | 是否启用整个 profiling 流程 |
| `profiling_warmup_steps` | 不计入正式结果的预热 step 数 |
| `profiling_log_interval` | 每多少 step 汇总和写入一次结果 |
| `profiling_torch_trace` | 是否额外启用高开销 PyTorch Profiler |
| `profiling_trace_active_steps` | PyTorch trace 的 active step 数量 |
| `gpu_monitor_interval_s` | `nvidia-smi` 采样周期 |

在 `TrainConfig.__post_init__()` 中增加合法性检查：

```python
profiling_warmup_steps >= 0
profiling_log_interval > 0
profiling_trace_active_steps > 0
gpu_monitor_interval_s > 0
```

这些字段全部有默认值，因此已有训练配置不需要修改。

### 5. `scripts/train_pytorch.py`：拆分训练 step 并输出结果

这是训练侧修改最多的文件。

#### 5.1 在构造 DataLoader 前打开 profiling

`train_loop()` 开始时增加：

```python
_profiling.set_enabled(enabled=config.profiling_enabled)
```

这一步在构造 DataLoader 和启动 worker 之前执行，确保 worker 可以继承 profiling 环境变量。

#### 5.2 创建输出目录

checkpoint 目录创建后增加：

```python
profile_dir = config.checkpoint_dir / "profiling"
profile_metrics_path = profile_dir / "step_metrics.jsonl"
data_metrics_path = profile_dir / "data_metrics.jsonl"
```

启用 profiling 后才创建目录和文件。

#### 5.3 选择被监控的 GPU

训练读取：

```python
CUDA_VISIBLE_DEVICES
```

并将第一个可见设备传给 `GpuMonitor`。例如：

```bash
CUDA_VISIBLE_DEVICES=3
```

会查询物理 GPU 3，而不是固定查询 GPU 0。这次脚本面向单卡 profiling，因此使用第一个可见设备。

#### 5.4 多 worker warning

当 `num_workers > 0` 时会输出 warning。原因是聚合器是进程内对象：

- worker 中可以计时，但主进程不能直接读取 worker 内存里的细粒度 counter。
- 主进程仍然可以正确测量 `data_wait_ms` 和端到端 samples/s。
- 精确拆分 access、pre-loading、preprocessing 时应使用 `num_workers=0`。
- `num_workers=2/4/8` 的测试主要用于比较实际流水线吞吐。

#### 5.5 新增 CUDA Event 辅助函数

新增：

```python
start_cuda_phase()
end_cuda_phase()
```

启用 profiling 且 device 是 CUDA 时，为每个阶段创建：

```python
torch.cuda.Event(enable_timing=True)
```

CUDA operation 是异步提交的。如果只使用 `time.perf_counter()`，测到的主要是 CPU 提交操作的时间，
不是 GPU 实际执行时间。CUDA Event 可以测量 GPU stream 上两个事件之间的实际耗时。

CPU 模式下使用 `time.perf_counter()` 作为 fallback。

#### 5.6 显式统计 DataLoader wait

训练循环从：

```python
for observation, actions in loader:
```

改为：

```python
data_iter = iter(loader)

data_started = time.perf_counter()
observation, actions = next(data_iter)
data_wait_ms = (time.perf_counter() - data_started) * 1000.0
```

`data_wait_ms` 表示训练主进程为了拿到下一个完整 batch 等待了多久。它是判断数据流水线是否让 GPU
空转的主要指标。

#### 5.7 H2D 计时

H2D 阶段包围：

```python
observation = jax.tree.map(
    lambda x: x.to(device, non_blocking=True),
    observation,
)
actions = actions.to(
    device=device,
    dtype=torch.float32,
    non_blocking=True,
)
```

输出字段：

```text
h2d_ms
```

#### 5.8 forward 计时

forward 阶段包围：

```python
losses = model(observation, actions)
loss = losses.mean()
```

输出字段：

```text
forward_ms
```

#### 5.9 backward 计时

backward 阶段包围：

```python
loss.backward()
```

输出字段：

```text
backward_ms
```

#### 5.10 optimizer 计时

optimizer 阶段包围：

```python
torch.nn.utils.clip_grad_norm_(...)
optim.step()
optim.zero_grad(set_to_none=True)
```

所以 `optimizer_ms` 包含梯度裁剪、参数更新和 optimizer 清梯度的时间，而不只是 `optim.step()`。

#### 5.11 CUDA 同步和阶段结果

profiling 模式下，每个 step 的四个 CUDA 阶段提交完成后执行：

```python
torch.cuda.synchronize(device)
```

然后读取：

```python
start_event.elapsed_time(end_event)
```

得到：

```text
h2d_ms
forward_ms
backward_ms
optimizer_ms
```

这个同步会带来 profiling 开销，因此 profiling 默认关闭，普通训练不会执行这次额外同步。

#### 5.12 step 总时间和 samples/s

`step_ms` 从请求下一个 batch 前开始，到当前 GPU 计算同步完成后结束，所以包含：

```text
data_wait + H2D + forward + backward + optimizer + 少量 CPU 调度开销
```

样本吞吐计算为：

```python
samples_per_second = effective_batch_size / (step_ms / 1000.0)
```

单卡训练时 `effective_batch_size` 等于配置中的 `batch_size`。

#### 5.13 warmup 清理

当完成 `profiling_warmup_steps` 后执行：

```python
_profiling.snapshot(reset=True)
gpu_monitor.summary(reset=True)
infos = []
start_time = time.time()
```

这样以下冷启动成本不会进入正式汇总：

- dataset 第一次初始化。
- 第一次读取 episode/Parquet/MP4。
- 视频 decoder 初始化。
- CUDA context 和 kernel 初始化。
- 第一次 forward/backward。
- optimizer state 第一次创建。

#### 5.14 周期日志和文件写入

profiling 模式使用：

```python
logging_interval = config.profiling_log_interval
```

每个窗口计算平均：

```text
data_wait_ms
h2d_ms
forward_ms
backward_ms
optimizer_ms
step_ms
samples_per_second
```

再合并 GPU monitor 的当前窗口汇总，写入：

```text
step_metrics.jsonl
```

数据侧聚合事件通过：

```python
_profiling.snapshot(reset=True)
```

写入：

```text
data_metrics.jsonl
```

#### 5.15 训练结束资源清理

训练结束后增加：

```python
torch_profiler.stop()
gpu_monitor.stop()
```

用于完成 trace 写入并结束 GPU 采样线程。

### 6. `scripts/train_pytorch.py`：可选 PyTorch Profiler

当同时满足：

```text
profiling_enabled = True
profiling_torch_trace = True
```

创建：

```python
torch.profiler.profile(...)
```

activities 包含 CPU；CUDA 可用时再加入 CUDA。schedule 为：

```python
torch.profiler.schedule(
    wait=profiling_warmup_steps,
    warmup=1,
    active=profiling_trace_active_steps,
    repeat=1,
)
```

同时启用：

```python
record_shapes=True
profile_memory=True
with_stack=False
```

`with_stack=False` 是为了控制 trace 开销和文件体积。

每个训练 step 结束调用：

```python
torch_profiler.step()
```

trace 输出到：

```text
<checkpoint_dir>/profiling/torch_trace/
```

该 trace 用于定位具体 PyTorch operator 和 CUDA kernel，不应使用 trace run 的 samples/s 作为最终吞吐结果。

### 7. `scripts/train_single.py`：将单卡 profiling 参数化

单卡入口从环境变量读取：

```python
profiling_enabled = os.environ.get("OPENPI_PROFILE", "0") == "1"
```

实验名根据模式隔离：

```python
exp_kind = "profile" if profiling_enabled else "smoke"
exp_name = f"interna1_pytorch_{exp_kind}_{run_id}"
```

因此 profiling 不会和普通 smoke run 使用同一输出目录。

新增 `env_int()`、`env_float()`，将以下环境变量写入 `TrainConfig`：

```text
OPENPI_TRAIN_STEPS
OPENPI_BATCH_SIZE
OPENPI_NUM_WORKERS
OPENPI_LOG_INTERVAL
OPENPI_PROFILE_WARMUP_STEPS
OPENPI_PROFILE_LOG_INTERVAL
OPENPI_TORCH_TRACE
OPENPI_TRACE_ACTIVE_STEPS
OPENPI_GPU_MONITOR_INTERVAL
```

profiling 模式下默认运行 120 step，而普通模式保留 10000 step。脚本启动时会打印 profiling、trace、
steps、batch size 和 worker 数，便于确认实际生效的配置。

模型权重路径仍然是：

```text
/datahub/public/model/pi05_base_pytorch_bf16
```

本次 profiling 修改没有改变权重加载逻辑。

### 8. `scripts/train_single.sh`：保留 shell 环境准备，调用独立 Python 入口

脚本负责：

- 设置 `OPENPI_WORKDIR`。
- 设置 `OPENPI_RUN_ROOT`。
- 默认设置 `CUDA_VISIBLE_DEVICES=0`。
- 优先使用 `OPENPI_PYTHON`。
- 其次使用仓库 `.venv/bin/python`。
- 最后 fallback 到系统 `python`。
- 执行 `scripts/train_single.py`。

脚本中新增 profiling 调用示例，但没有在 bash 中嵌入 Python 代码：

```bash
OPENPI_PROFILE=1 OPENPI_TORCH_TRACE=1 OPENPI_NUM_WORKERS=0 scripts/train_single.sh
```

### 9. `.gitignore`：保留 profiling 报告文件

仓库原本忽略整个 `docs/` 目录。本次将规则调整为忽略 `docs/*`，再单独允许：

```gitignore
!docs/interna1_single_gpu_profiling_report.md
```

这样 profiling 生成的数据仍然位于被忽略的 `runs/` 目录，不会误提交；这份人工修改报告可以被 Git 跟踪。

## 完整运行时调用链

启用 profiling 后，单卡训练的主要调用顺序为：

```text
scripts/train_single.sh
  -> scripts/train_single.py
     -> 读取 OPENPI_PROFILE 等环境变量
     -> 构造 pi05_interna1 TrainConfig
     -> scripts/train_pytorch.py::train_loop()
        -> profiling.set_enabled()
        -> 创建 checkpoint/profiling 目录
        -> 启动 GpuMonitor
        -> create_data_loader()
        -> next(data_iter)
           -> TransformedDataset.__getitems__()
              -> PetrelDataset.__getitems__()
                 -> local/Petrel access
                 -> Parquet 解析
                 -> MP4 读取与解码
                 -> InternA1 control projection
              -> data.access_and_preload
              -> Normalize/Resize/Tokenize/Pad
              -> data.preprocessing
        -> data_wait_ms
        -> H2D
        -> forward
        -> backward
        -> gradient clipping + optimizer
        -> CUDA synchronize + elapsed_time
        -> 写 step_metrics.jsonl
        -> 写 data_metrics.jsonl
        -> GpuMonitor 持续写 gpu_metrics.csv
        -> 可选写 torch_trace/
```

## 指标解释与定位方法

| 现象 | 优先查看 | 可能原因 |
|---|---|---|
| GPU utilization 低，`data_wait_ms` 高 | `data_metrics.jsonl` | 文件读取、视频解码、预处理或 worker 不足 |
| `local.get_bytes` MB/s 低 | reader I/O 事件 | `/manip` 挂载吞吐或大量随机小文件访问 |
| `video.decode` 时间高 | video reader 事件 | MP4 解码成为瓶颈 |
| `data.preprocessing` 高 | preprocessing 总事件 | resize、tokenize、normalize 或 Python transform |
| `h2d_ms` 高 | `step_metrics.jsonl` | CPU 到 GPU 传输或内存布局问题 |
| `forward_ms` 高 | step metrics/torch trace | 模型前向算子瓶颈 |
| `backward_ms` 高 | step metrics/torch trace | 反向算子或 gradient checkpointing 成本 |
| `optimizer_ms` 高 | step metrics | 梯度裁剪、AdamW 更新成本 |
| GPU utilization 周期性下降 | `gpu_metrics.csv` + `data_wait_ms` | DataLoader 不能持续供给 batch |

## 当前统计边界和限制

1. `data.access_and_preload` 是底层 dataset 总耗时。具体 access 和 pre-loading 需要结合 reader 的细粒度事件，而不是只看这一个字段。
2. InternA1 的 80D control projection 位于 `PetrelDataset._pack_item()`，当前计入 `data.access_and_preload`，没有独立的 `control_projection_ms`。
3. `num_workers>0` 时，worker 内聚合器属于各自进程，主进程的 `data_metrics.jsonl` 不能完整汇总所有 worker 的内部事件；此模式重点看 `data_wait_ms` 和 samples/s。
4. profiling 模式每 step 使用一次 CUDA synchronize，以获得准确阶段耗时，因此其绝对吞吐可能低于完全异步的正式训练。
5. PyTorch trace 还会进一步增加开销，trace run 适合定位算子，不适合作为最终吞吐基准。
6. GPU monitor 依赖可工作的 NVIDIA driver 和 `nvidia-smi`。缺少它们时数据训练 profiling 仍然可以运行，但不会获得 GPU 利用率。

## 验证结果

本次修改完成后执行了：

- Python `py_compile` 检查。
- Ruff 静态检查和格式检查。
- `bash -n scripts/train_single.sh`。
- `git diff --check`。
- Petrel/InternA1 相关测试：`21 passed`。
- profiling 聚合器 count、耗时、bytes 和 MB/s 的行为检查。
- `OPENPI_DRY_RUN=1 OPENPI_PROFILE=1` 单卡入口检查。

当前开发容器中的 `nvidia-smi` 无法连接 NVIDIA driver，因此没有在该容器内完成真实 GPU step 的性能采样。代码会在具有可用 NVIDIA GPU 和 driver 的训练节点上生成 GPU 指标。
