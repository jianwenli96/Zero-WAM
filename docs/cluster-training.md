# Ascend 单机八卡训练

此分支交付从原始 Wan 初始化的 HumanGen + RoboTwin 联合训练。目标配置为一台节点上的 **8×Ascend 910B3、每卡 64 GiB**。入口是 `script/train_humangen_wan_npu.sh`，由该入口启动八个训练进程。集群调度器负责分配整台节点/设备；当前入口没有配置多节点 rendezvous。不同硬件、卡数或 FSDP 配置不能直接沿用这份容量标定。

## 训练设计

| 项目 | 当前设置 |
| --- | --- |
| 初始化 | 原始 Wan2.2-TI2V-5B 转换，seed 42，FP32 初始化权重；不使用训练后的 Zero-WAM Transformer 初始化 |
| 数据 | HumanGen 五个外部来源 + RoboTwin 43 个训练任务；7 个测试任务由 manifest 与 loader 双重隔离 |
| 来源采样 | HumanGen:RoboTwin=4:1；HumanGen 内按论文来源任务数的平方根分配概率；来源内均匀抽样 |
| 模型与目标 | 视频、动作、4 个 IFP/MCP 模块联合训练；IFP 接收层索引 3/11/19/29 的特征，各一个 block |
| IFP 损失权重 | 0.5 / 0.25 / 0.15 / 0.1；总损失为视频、动作及加权 IFP 损失之和 |
| 条件 | 人类视频 dropout 0.1、目标文字 dropout 0.4；human RoPE 偏移 24 |
| 优化器 | fused AdamW，LR 1e-4，betas 0.9/0.95，weight decay 0.01，预热 200 步后恒定 LR |
| 梯度 | max norm 1；范数非有限或超过阈值 20 时跳过优化器更新并记录 |
| 并行与精度 | FSDP2 sublayer，BF16 计算、FP32 reduce，主干及 MCP 块激活检查点，forward 后 reshard |
| 批次 | 每卡一条轨迹、梯度累积 1；一次优化器步共八条轨迹，token 数可不同 |
| 数据加载 | `INIT_WORKERS=1`、每卡 `LOAD_WORKERS=2`；冷缓存时 rank 0 先构建索引，其他 rank 随后读取 |
| 长度分桶 | 混合来源默认 `LENGTH_BUCKET_STEPS=10`；在同一批加权抽样结果内重排，按裁剪后的 2R+H 估计成本 |
| 长序列 | 八卡默认加载 `sequence_capacity_8npu.json`；保留短样本，超预算时随机取连续机器人窗口 |

裁剪在 CPU 上同步作用于机器人 latent、动作与动作掩码；不因过长删除样本。窗口裁剪不缩短人类视频与文字；上述条件 dropout 仍按训练配置独立执行。人类视频与机器人轨迹没有逐帧对齐关系。

当前 NPU 内存优化还包括：按行分块生成稠密注意力 mask，减少整数临时矩阵；同一步 MCP 各深度复用不可变 mask；NPU 全屏蔽 padding 行输出清零，保持参考输出/梯度语义。分块构建后仍保存完整二维 bool mask，不是稀疏注意力。数据侧保留索引/Arrow 缓存与 manifest 索引复用。单进程数据初始化避免任务对象经多进程返回时复制大型 manifest；训练本身仍为八进程。

权重映射见 [Wan 初始化说明](wan-initialization.md)。此配方没有补充未公开的 VA-only/内部 HumanGen 数据，没有实现论文的 160K token 多样本打包；RoPE 偏移与 IFP 最后一项损失权重继续沿用公开代码，不能标为完整论文配方复现。

## 数据采样

默认策略由 [`humangen_robotwin_sampling.json`](../wan_va/configs/humangen_robotwin_sampling.json) 定义，启动时由 `script/resolve_training_sampling.py` 解析。来源概率为：AgiBot 33.1128%、InternData-A1 9.2371%、OXE 11.9661%、RoboCOIN 12.9375%、RoboMind 12.7466%、RoboTwin 20%。任务数是论文完整数据集的先验；来源内部按样本均匀抽取，没有实现严格 task-balanced，也不保证短程抽样数量精确符合概率。

`SAMPLING_CONFIG` 可替换策略 JSON，`DATASETS='agibot:1,interna1:1'` 可显式覆盖来源及相对权重。RoboTwin 始终由 loader 排除七个测试任务。各来源保留自己的相机布局、动作步长、转换配置和归一化统计。

## 环境与外部资产

Git 分支包含代码、配置、空文本特征和说明。初始化权重、HumanGen 原始数据、可写数据视图、Arrow 缓存、训练输出均不在 Git 中。目标节点必须能访问真实数据及权重；如果共享存储挂载位置不同，应重新运行数据准备脚本，避免复制带有旧主机绝对路径的软链接视图。

使用已经验证过的 Ascend 训练环境。当前本机栈为 Python 3.12.13、PyTorch/torch_npu 2.9.0、CANN 9.1.0、LeRobot 0.3.3；依赖列表见 `requirements-npu.txt`、`pyproject.toml`。LeRobot 的 torch 版本约束与本环境冲突，应在已准备好其依赖的环境中使用 `pip install --no-deps lerobot==0.3.3`，避免自动降级 torch。仓库不提供驱动/CANN，目标节点需要准备兼容环境。

在仓库根目录设置路径；以下 `/cluster/...` 均需替换。`PYTHON_BIN` 指向 mentor 已准备好的环境，不要求目标机器使用本机 `.venv`。

```bash
export PYTHON_BIN=/cluster/envs/zero-wam/bin/python
export CANN_ENV_PATH=/usr/local/Ascend/cann-9.1.0/set_env.sh
export HUMANGEN_SOURCE=/cluster/datasets/HumanGen
export HUMANGEN_ROOT=/cluster/work/lianjie/HumanGen-view
export MODEL_PATH=/cluster/checkpoints/zero-wam-wan-init-fp32-seed42
export WAN_SOURCE=/cluster/checkpoints/Wan-AI/Wan2.2-TI2V-5B
export HF_HOME=/cluster/work/lianjie/hf-cache
export HF_DATASETS_CACHE="$HF_HOME/datasets"
```

原始 HumanGen 目录需要解压后的机器人数据、动作 parquet、机器人 latent、人类 latent、配对 manifest 和动作转换/归一化配置。预计算 latent 训练无需 VAE 或文本编码器在线编码。下载与解压目录结构见仓库 README。

已有有效 FP32 初始化目录可复用；否则执行一次转换，输出目录必须尚不存在：

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 OMP_NUM_THREADS=8 \
  "$PYTHON_BIN" -m wan_va.wan_init \
  --source "$WAN_SOURCE" --output "$MODEL_PATH" --seed 42 --dtype float32
```

数据视图应与原始数据分开，并可写入索引缓存：

```bash
"$PYTHON_BIN" script/prepare_humangen_training.py \
  --source "$HUMANGEN_SOURCE" --output "$HUMANGEN_ROOT"
"$PYTHON_BIN" script/prepare_robotwin_training.py \
  --source "$HUMANGEN_SOURCE" --output "$HUMANGEN_ROOT"

bash script/train_humangen_wan_npu.sh --check-only
```

`--check-only` 检查外部来源的代表动作配置与 RoboTwin 各训练任务，写入 `external-validation.json`、`validation.json`，不启动训练，也不逐条校验全量张量。它要求模型初始化与数据准备已经完成。

## 节点启动

调度器分配八卡后，在任务节点上设置其实际可见设备编号。以下 0–7 仅为设备分配示例；不同集群由作业脚本注入对应编号，不在八个独立调度任务里重复运行这个 torchrun 启动器。

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export INIT_WORKERS=1 LOAD_WORKERS=2 LENGTH_BUCKET_STEPS=10
export FSDP_GRANULARITY=sublayer GRAD_ACCUM=1
export SEQUENCE_CAPACITY_PROFILE="$PWD/wan_va/configs/sequence_capacity_8npu.json"
export TRAIN_SEED=42 LEARNING_RATE=0.0001
export MASTER_PORT=29617

# 10 步流程检查，包括第 10 步的模型权重保存；输出目录必须是新的。
export NUM_STEPS=10 SAVE_INTERVAL=10
export ZERO_WAM_SAVE_ROOT=/cluster/runs/lianjie/wan-six-source-pilot-001
bash script/train_humangen_wan_npu.sh --dry-run
bash script/train_humangen_wan_npu.sh --run
```

`--dry-run` 只显示配置，默认按八进程展示；真正运行仍要求显式设备分配。`--run` 自动加载 CANN 环境，默认启用 `expandable_segments:True`，并拒绝复用已有输出目录。首个优化器步包含较大的首次执行开销，不能直接当作稳定吞吐。

正式训练使用新的目录，并明确总步数/保存间隔。下面 50,000 步沿用训练配置中的总步数，仅作为启动示例；实际运行预算由集群实验安排决定：

```bash
export NUM_STEPS=50000 SAVE_INTERVAL=1000
export ZERO_WAM_SAVE_ROOT=/cluster/runs/lianjie/wan-six-source-train-001
bash script/train_humangen_wan_npu.sh --dry-run
bash script/train_humangen_wan_npu.sh --run
```

省略 `NUM_STEPS` / `SAVE_INTERVAL` 时仍为 10 步试跑。`LENGTH_BUCKET_STEPS=0` 可关闭分桶；单来源 `DATASETS` 默认不分桶。`MAX_TRAIN_FRAMES` 是可选的额外机器人帧数上限，交接配置未设置。容量配置要求 batch=1、八卡、累积=1、sublayer 和指定模型结构；改变这些设置时不要直接复用标定。`TRAIN_DIAGNOSTICS=1` 可记录逐卡样本/形状/显存，但增加同步和日志开销，正式训练默认关闭。

## 其他入口与检查

纯 RoboTwin 消融使用 `script/train_robotwin_wan_npu.sh`，只需 RoboTwin 数据准备与检查；该独立入口不自动启用混训入口的容量配置、分桶或诊断选项。原生数据检查脚本分别为 `script/check_humangen_training.py` 与 `script/check_robotwin_training.py`，可通过 `--help` 查看全量检查选项。

开发时可运行 CPU 回归，隔离 Ascend 迁移层的全局 CUDA 替换：

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 OMP_NUM_THREADS=2 \
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 "${PYTHON_BIN:-python}" - <<'PY'
import sys
sys.modules['torch_npu.contrib.transfer_to_npu'] = None
import pytest
raise SystemExit(pytest.main(['-q', 'tests', '-k', 'not cuda and not npu']))
PY
```

CPU 回归不代替目标设备验证。

## 运行产物与限制

最近的八卡真实子集训练已验证裁剪、前向、反向及优化器更新；容量子集和吞吐子集均在用户要求下提前停止，完成范围见 [容量规则与验证范围](training-memory.md)。已有结果不覆盖全量数据、长期收敛、多机或本次八卡配置的 checkpoint 保存阶段，因此目标节点的流程检查保留保存步骤。

每个运行目录包含 `command.txt`、`sampling.json`、数据/初始化来源、`code-head.txt`、`code-changes.patch`、`source.tar.gz`、`train.log` 和 `metrics.jsonl`。模型权重保存在 `checkpoints/checkpoint_step_N/transformer`。保存间隔需要整除总步数才能在最后一步保存。

当前 checkpoint 是 BF16 模型快照，**没有完整优化器、scheduler、采样器和随机数状态，不支持精确断点续训**。不要把从权重重新启动描述为完整恢复。
