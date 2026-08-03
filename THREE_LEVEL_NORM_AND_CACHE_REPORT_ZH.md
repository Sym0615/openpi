# 三级 Norm 与 dataset_id 缓存修改报告

## 1. 修改目标

本次修改落实两项要求：

1. Norm 按“task/source → control profile/机器人本体 → dataset”三级选择。
2. `data/petrel_cache` 中的 metadata 与 path index 只按 `dataset_id` 建立一级数据集命名空间，不再把 control profile、robot id 或 norm group 放入缓存身份。

最终缓存路径为：

```text
data/petrel_cache/meta/{dataset_id}/{remote}
data/petrel_cache/path_index/{dataset_id}/{remote}
```

这里的 `{remote}` 是物理 leaf dataset 的相对路径，用于避免同一个 `dataset_id` 下不同 task 的 `meta/info.json` 和 `path_index.jsonl` 相互覆盖；缓存身份中没有 norm group、profile 或 robot 字段。

## 2. Norm 三级优先级

每个 `DatasetSpec` 最终携带：

- `stats_group`：当前样本优先使用的统计组。
- `stats_group_fallbacks`：按顺序尝试的后备统计组。

构造顺序如下：

```text
dataset_groups/<dataset>/<task>.yaml 的 stats_group 或 stats_group_template
    ↓ 没有或统计文件里缺失
control_profiles/<dataset>.yaml 的 stats_group 或 stats_group_template
    ↓ 没有或统计文件里缺失
pi05_<dataset>.yaml 的 norm_stats.dataset_group_template
```

如果 task 级没有配置，profile 级会被提升为 `stats_group`，dataset 级进入 fallback。如果 task 和 profile 都没有配置，则 dataset 级直接成为 `stats_group`。

核心实现位置：

- `src/openpi/training/petrel_lerobot_reader.py:3780-3787`：把 profile norm 配置写入内部 profile。
- `src/openpi/training/petrel_lerobot_reader.py:3843-3849`：把 task/source norm 配置和 `robot_id` 写入内部 source。
- `src/openpi/training/petrel_lerobot_reader.py:4220-4259`：解析主统计组，依次构造 profile、dataset fallback，并在上一级不存在时提升 fallback。
- `src/openpi/training/petrel_lerobot_reader.py:4451-4467`：定义 source → profile → dataset 的继承次序。

## 3. InternA1 的机器人级 Norm

InternA1 的不同机械臂本体不能默认假设具有相同的 state/action 分布。因此在：

```text
configs/petrel/control_profiles/intern_a1.yaml
```

增加：

```yaml
profile_defaults:
  stats_group_template: profile/{dataset_id}/{robot_id}
```

`robot_id` 优先读取 dataset-group 显式字段；未显式填写时，从物理 root 的第二段推导。例如：

```text
articulation_tasks/franka  -> franka
basic_tasks/lift2          -> lift2
basic_tasks/split_aloha    -> split_aloha
basic_tasks/acone          -> acone
basic_tasks/genie1         -> genie1
```

实现位置：

- `src/openpi/training/petrel_lerobot_reader.py:3806-3811`：解析或推导 `robot_id`。
- `src/openpi/training/petrel_lerobot_reader.py:4148-4149`：在 leaf spec 构造时固定 `dataset_id` 和 `robot_id`。
- `configs/petrel/control_profiles/intern_a1.yaml:2-4`：定义机器人级模板。

InternA1 当前生成五套机器人级统计：

```text
profile/intern_a1/acone
profile/intern_a1/franka
profile/intern_a1/genie1
profile/intern_a1/lift2
profile/intern_a1/split_aloha
```

每套机器人统计的 fallback 都是：

```text
dataset/intern_a1
```

如果未来某个 task YAML 增加：

```yaml
stats_group_template: task/{dataset_id}/{source}/{name}
```

该 task 的最终候选顺序会变成：

```text
task/intern_a1/<source>/<name>
profile/intern_a1/<robot_id>
dataset/intern_a1
```

## 4. ABC-130K 的 Norm

ABC-130K 的 3cam 与 4cam profile 只描述相机映射差异，state/action 控制空间相同，所以没有在 `control_profiles/abc130k.yaml` 中配置 profile 级 norm。

其数据集级统计由：

```yaml
norm_stats:
  dataset_group_template: dataset/{dataset_id}
```

产生，所有 ABC leaf dataset 当前都使用：

```text
dataset/abc130k
```

对应位置：

- `configs/petrel/pi05_abc130k.yaml:28-29`
- `src/openpi/training/petrel_dataset_test.py:777-788`

以后如果某个 ABC task 显式增加 task norm，它仍会优先于 dataset norm，但不会因为 3cam/4cam 自动拆分控制统计。

## 5. metadata 与 path_index 缓存

以下配置全部改为统一模板：

- `configs/petrel/pi05_interna1.yaml:21-25`
- `configs/petrel/pi05_abc130k.yaml:22-27`
- `configs/petrel/pi05_petrel.yaml:18-24`

模板为：

```yaml
metadata_cache: data/petrel_cache/meta/{dataset_id}/{remote}
path_index_cache: data/petrel_cache/path_index/{dataset_id}/{remote}
```

模板展开时传入 `dataset_id`，但不使用 `stats_group`、`robot_id` 或 profile 名称：

- `src/openpi/training/petrel_lerobot_reader.py:4148-4187`：展开 metadata 路径。
- `src/openpi/training/petrel_lerobot_reader.py:4189-4198`：展开 path-index 路径。
- `src/openpi/training/petrel_lerobot_reader.py:4431-4434`：缓存模板只从 source 显式覆盖或 pi05 配置继承；当前新 schema 的 dataset-group/profile 不提供缓存覆盖。

示例：

```text
data/petrel_cache/meta/intern_a1/articulation_tasks/franka/close_the_electriccooker
data/petrel_cache/path_index/intern_a1/articulation_tasks/franka/close_the_electriccooker

data/petrel_cache/meta/abc130k/<physical-leaf-path>
data/petrel_cache/path_index/abc130k/<physical-leaf-path>
```

注意：修改只影响之后解析和生成的缓存路径，不会自动删除历史缓存目录。

## 6. 测试与验证

定向测试：

```text
31 passed
```

覆盖内容包括：

- 缓存模板位于 pi05 dataset 配置，而不是 control profile。
- InternA1 五种机器人本体使用独立 profile norm。
- InternA1 dataset norm 是每个机器人统计的 fallback。
- task norm 存在时，候选顺序严格为 task → profile/robot → dataset。
- ABC-130K 只生成 dataset 级 norm。
- ABC metadata/path-index 路径都包含唯一的 `abc130k` 数据集命名空间。

真实 InternA1 小样本 norm smoke test 使用 64 个窗口完成，输出 12 个 key：五个机器人组加一个 dataset 组，各自包含 `state` 和 `actions`，所有统计维度均为 `(32,)`。

静态检查：

```text
ruff: All checks passed
```

仓库当前另有一处已有 task YAML 行尾空格：

```text
configs/petrel/dataset_groups/intern_a1/a1_articulation_franka.yaml:7
```

它不属于本次三级 norm/cache 实现，因此本次没有改写用户正在编辑的 dataset-group 文件。

## 7. 当前限制

`pi05_petrel.yaml` 的混合训练仍保持显式不可用，直到其余旧数据集全部迁移到 `mapped` control contract。此次只统一了它未来使用的 dataset-id 缓存模板，没有恢复未完成的混合训练入口。
