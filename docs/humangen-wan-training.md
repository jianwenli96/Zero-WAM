# 从 Wan 初始化的 HumanGen 外部 ICL 预训练

本机公共目录已包含五个外部来源解压后的机器人数据、动作、机器人 latent、人类视频 latent、配对 manifest，以及各机器人对应的动作转换配置。本流程只读取 `/mnt/sfs_turbo/public/datasets/HumanGen` 中的原始文件，在 `data/HumanGen` 下创建本地数据视图和缓存目录。

| 来源 | LeRobot 数据目录数（非语义任务数） | manifest 配对数 | 机器人训练片段数 |
| --- | ---: | ---: | ---: |
| AgiBot | 187 | 6,659 | 6,659 |
| RoboCOIN | 238 | 6,102 | 6,102 |
| RoboMIND | 392 | 6,072 | 6,082 |
| InternData-A1 | 1,161 | 11,230 | 11,230 |
| OXE / Bridge | 19 | 5,165 | 5,165 |
| 合计 | 1,997 | 35,228 | 35,238 |

RoboMIND 中多个机器人片段可以复用同一段人类示范。准备脚本保留这些对应关系，不会为了使数量相等而丢弃片段。数据目录数反映存储划分，不能用作论文中的语义任务数量。

## 数据准备与检查

在 Zero-WAM 仓库根目录执行：

```bash
.venv/bin/python script/prepare_humangen_training.py
bash script/train_humangen_wan_npu.sh --check-only
```

准备脚本检查每个 manifest 引用的人类 latent、选中轨迹的动作 parquet，以及配置中指定的相机 latent。它保留嵌套的数据目录结构、数据集合级的 `action_transform.yaml` 和归一化统计。若转换后的本地轨迹编号与源文件编号不同，会按元数据映射回原始轨迹及帧区间。检查结果写入 `data/HumanGen/external-preparation.json`。

CPU loader 检查默认从每组机器人动作配置选择一个真实数据目录，共 22 组，每个选中目录读取一个样本。检查内容包括：loader 是否保留该目录的全部样本、视频与动作帧是否对齐、张量是否为有限数值，以及补齐到 30 通道后的动作掩码是否有效。外部数据的相机数量、每个 latent 帧对应的动作采样数可能与 RoboTwin 不同。

结果写入 `external-validation.json`。这是按动作配置分组的代表性抽样检查，不代表已读取全部样本内容，也不代表已为所有数据目录建立索引。如需建立全部目录的索引，并从每个目录读取一个样本：

```bash
HF_HOME="$PWD/outputs/hf-cache" HF_DATASETS_CACHE="$PWD/outputs/hf-cache/datasets" \
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 OMP_NUM_THREADS=2 \
.venv/bin/python script/check_humangen_training.py --all-repos
```

首次真正启动训练时，rank 0 会先建立缺失的 Arrow 和数据索引缓存，完成后其他 rank 再继续。`INIT_WORKERS` 默认为 4。每个进程会缓存 manifest 索引，避免上千个数据目录反复解析同一份大型 JSON；文件大小或修改时间变化时，缓存失效并重新加载。

## 默认采样策略与启动

默认使用 **HumanGen 五个外部来源 + RoboTwin 43 个训练任务联合训练**，从 `checkpoints/zero-wam-wan-init-fp32-seed42` 初始化。两级概率合并后交给现有分布式来源采样器：

1. HumanGen : RoboTwin = **4:1**，即 80% : 20%。
2. HumanGen 内部按论文各来源任务数的 **平方根** 分配概率。
3. 选中来源后，在该来源内部均匀抽取样本。

配置文件为 [`humangen_robotwin_sampling.json`](../wan_va/configs/humangen_robotwin_sampling.json)，解析脚本为 [`resolve_training_sampling.py`](../script/resolve_training_sampling.py)。计算使用完整浮点精度，以下百分比仅用于展示：

| 来源 | 论文任务数先验 | 最终采样概率 |
| --- | ---: | ---: |
| AgiBot | 3,354 | 33.1128% |
| InternData-A1 | 261 | 9.2371% |
| OXE / Bridge | 438 | 11.9661% |
| RoboCOIN | 512 | 12.9375% |
| RoboMIND | 497 | 12.7466% |
| RoboTwin | 43 个训练任务 | 20.0000% |

对于 HumanGen 来源 `s`，其概率为 `0.8 * sqrt(N_s) / sum(sqrt(N))`；RoboTwin 概率固定为 0.2。这是每次抽样的概率，不保证每个短批次或每轮的实际数量精确相等。

**任务数来自论文完整数据集合，只作为来源多样性的先验；它不是本地重新统计出的任务数。这一方案没有实现来源内部严格的 task-balanced。** RoboTwin 的各任务基本各有 50 个样本，来源内均匀抽样已接近其任务均衡。它仍使用训练专用 manifest，并由 loader 独立执行 43/7 任务隔离。

当前可用数据共 37,387 个训练片段。需要完成 [RoboTwin 数据准备](robotwin-wan-training.md)与上文的外部 HumanGen 准备。`--human-gen-root` 为所有来源解析本地路径。

直接查看默认命令和实际概率，不需要再手写 `DATASETS`：

```bash
bash script/train_humangen_wan_npu.sh --dry-run
```

CPU 启动前检查会打印策略，并依次检查外部 HumanGen 与 RoboTwin：

```bash
bash script/train_humangen_wan_npu.sh --check-only
```

在已分配六张空闲卡的主机上，执行 10 步训练验证。以下设备编号只是示例，需替换为实际分配的物理卡：

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5 \
ZERO_WAM_SAVE_ROOT="$PWD/train_out/wan-humangen-robotwin-4to1-sqrt-seed42-pilot" \
bash script/train_humangen_wan_npu.sh --run
```

未指定设备列表时，dry-run 默认展示六个进程；实际启动时进程数由 `ASCEND_RT_VISIBLE_DEVICES` 决定。六卡来自之前的 FSDP 经验，不表示已重新验证当前完整配置的显存或吞吐。只有 `--run` 启动训练，数据准备、CPU 检查和 dry-run 均不进行 NPU 训练、显存测试或仿真器评测。

默认 10 步只用于流程验证。正式实验可通过 `NUM_STEPS`、`SAVE_INTERVAL` 和新的 `ZERO_WAM_SAVE_ROOT` 设置时长及保存位置；学习率、随机种子等参数与 RoboTwin 启动入口一致。每次实际运行额外保存 `sampling.json`，记录任务数先验、配置内容、最终六源权重及分组概率。

如需更换策略，可用 `SAMPLING_CONFIG` 指定另一份同结构 JSON。仍支持显式 `DATASETS` 覆盖，例如五源等权对照：

```bash
DATASETS='agibot:1,robocoin:1,robomind:1,interna1:1,oxe:1' \
bash script/train_humangen_wan_npu.sh --dry-run
```

覆盖后报告中的策略名为 `custom_DATASETS`，实际概率重新归一化，不再使用默认 4:1 配比。无效、重复、非正或非有限权重会在启动分布式进程前报错。

## 长序列与显存

8 卡启动时默认启用按容量随机窗口裁剪：保留全部样本及完整人类示范，仅对超预算的机器人视频、动作和掩码同步裁剪。其他卡数不自动使用该容量配置，可设置 `MAX_TRAIN_FRAMES`。启用方式、适用配置、数据影响与实测依据统一见[训练显存与长序列处理](training-memory.md)。

## 与论文训练范围的区别

公开的 HumanGen 外部 ICL 子集不等于论文完整的预训练数据。论文还使用了 Task-diverse VA 和内部 HumanGen 数据。论文中的 `VA:HumanGen = 1:5` 无法通过调整这五个外部来源的采样权重来复现。论文 RoboTwin 后训练的比例是 `VA:HumanGen:RoboTwin = 2:10:3`；当前的 HumanGen:RoboTwin=4:1 是接近其中两者相对比例的实验选择，不是原始配方。

当前入口没有补充缺失的 VA-only 数据集，也不会把丢弃人类视频条件后的 ICL 样本当成该数据集。它支持基于本机已有公开数据开展可复现的实验。

IFP 权重、RoPE 偏移和单轨迹训练方式继续沿用公开代码，详见 [Wan 初始化说明](wan-initialization.md)与 [RoboTwin 训练说明](robotwin-wan-training.md)。实际启动后会保存训练指标、源码快照、初始化记录和数据来源。模型快照仍不支持完整优化器与随机数状态的恢复。

## 对齐前的历史验证记录

全部 35,228 条 manifest 人类 latent 路径，以及 35,238 个机器人片段对应的动作和相机 latent 引用，均通过准备检查。

真实 loader 抽样覆盖全部 22 组机器人动作配置，涉及 22 个数据目录、668 个已索引片段，每个目录读取一个样本。所选目录没有发生静默丢样本。其余目录的索引可按需提前建立，或在首次训练时建立。

检查时发现并修复了 parquet 兼容问题：外部数据带有新版 Hugging Face `List` 特征元数据，而 LeRobot 0.3.3 限制 `datasets<=3.6`。外部 latent loader 现在只读取所需的数值型状态/动作列，并从实际 Arrow 类型推导特征，保留数值和列表形状，避开不兼容的视频及特征元数据解析。公共 parquet 文件和原有依赖版本均未修改。

该读取方式有专门的回归测试，覆盖变长/定长数值列表及 Arrow 缓存回读。数据准备、混合采样、manifest 缓存、动作转换和 parquet 兼容相关的 43 项测试通过。五源与六源启动命令均通过 dry-run 和 shell 语法检查；未启动 NPU 训练。

默认联合采样策略落地后，另运行采样策略、混合采样与公开 ICL 配置相关测试，28 项通过。其中验证了六个 rank 的交错采样序列与单进程全局序列一致，以及 120,000 次抽样的来源频率符合配置概率。统一 `--check-only` 入口通过，共实际读取 65 个样本：外部 HumanGen 22 个，RoboTwin 43 个。完整六卡训练的显存、吞吐、反向传播和保存流程仍需在分配到空闲卡后通过默认 10 步训练验证。
