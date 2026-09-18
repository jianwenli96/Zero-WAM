# 昇腾上的 Wan 初始化 RoboTwin ICL 训练

本入口保留了仅使用 RoboTwin 的消融实验，对应论文中的 **Zero-WAM w/o pretrain**：从原始 Wan 视频模型出发，只用 43 个已见 RoboTwin 任务训练，另外 7 个任务留作评测。它不复现完整的 HumanGen / Task-diverse VA 预训练混合数据。

若要使用已准备好的五个外部来源，或将它们与 RoboTwin 混训，请查看 [HumanGen 预训练说明](humangen-wan-training.md)。

## 数据准备

在仓库根目录执行：

```bash
python script/prepare_robotwin_training.py \
  --source /mnt/sfs_turbo/public/datasets/HumanGen \
  --output data/HumanGen
```

脚本检查每条选中轨迹的动作 parquet、三个相机的 latent，以及配对的人类视频 latent。随后创建本地任务目录，通过符号链接引用现有数据。数据索引缓存写入本地目录，不修改公共数据。

生成的 `ICL_config_robotwin_train.json` 只包含 43 个训练任务，loader 还会独立排除 7 个测试任务。`preparation.json` 记录数据划分、数量和来源路径。

本地发布数据共 2,499 对样本，其中训练集 2,149 对、测试集 350 对，对应 7,497 个机器人相机 latent 文件。使用预计算 latent 训练时，无需重新下载或编码视频。

建立 Arrow 与数据索引缓存，并从每个训练任务读取一个真实配对样本：

```bash
bash script/train_robotwin_wan_npu.sh --check-only
```

此命令要求下文所述的 FP32 初始化目录已存在。检查会确认 loader 保留全部 2,149 个训练样本，并对抽样读取的张量检查视频/动作对齐、通道数、动作掩码和数值是否有限。结果保存在 `data/HumanGen/validation.json`。

如需读取全部训练样本：

```bash
HF_HOME="$PWD/outputs/hf-cache" HF_DATASETS_CACHE="$PWD/outputs/hf-cache/datasets" \
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 OMP_NUM_THREADS=2 \
.venv/bin/python script/check_robotwin_training.py --samples-per-task 0
```

检查与训练入口默认使用本地 Hugging Face 缓存，并启用离线模式。索引本地 parquet 时，依赖库的进度条可能显示 `Downloading data`，不表示正在从网络下载数据。

## 初始化与训练配置

`MODEL_PATH` 默认指向 `checkpoints/zero-wam-wan-init-fp32-seed42`。该权重通过 `--dtype float32` 直接从原始 Wan 转换而来，并非将之前的 BF16 初始化转回 FP32。具体见[权重转换说明](wan-initialization.md)。

FP32 初始化/主权重与 BF16 FSDP 前向计算承担不同作用：读取 FP32 权重可以避免在混合精度训练开始前就对初始化数值进行舍入。

启动脚本沿用公开训练配置：视频分支、动作分支及四个 IFP 模块联合训练；人类视频条件丢弃概率为 0.1，目标文字条件丢弃概率为 0.4；AdamW 学习率为 `1e-4`，预热 200 步，每个 rank 的 batch size 为 1，并启用激活检查点及 FSDP2/HCCL。

默认的 10 步试跑只用于检查训练流程和显存，仍处于学习率预热阶段。公开代码按单条轨迹训练，**没有实现论文中多样本打包至 160K token 的机制**。IFP 损失权重和人类视频 RoPE 偏移也保留公开代码的取值，差异详见转换说明。

训练随机种子默认为 42，各 rank 使用独立的随机数流。这有助于比较实验，但不保证 NPU 算子逐比特确定性。

## 检查命令与启动

只查看启动命令，不访问 NPU、不启动训练：

```bash
bash script/train_robotwin_wan_npu.sh --dry-run
```

实际启动时，必须指定已分配给当前任务的设备。以下仅以八张空闲卡为例：

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
ZERO_WAM_SAVE_ROOT="$PWD/train_out/wan-robotwin-icl-seed42-pilot" \
bash script/train_robotwin_wan_npu.sh --run
```

示例不表示物理卡 0–7 当前可用。`--run` 要求显式指定可见设备，根据列表推导进程数，并拒绝复用已有运行目录。

当前纯 RoboTwin 入口未随最近六源子集测试重新验收；最近的完整模型八卡混合子集测试见 [显存记录](training-memory.md)，不应当作纯 RoboTwin 全量训练结论。约 108 亿个独立参数的 FP32 权重、梯度和两个 Adam 动量合计约 173 GB（十进制），还不包括 BF16 参数聚合及激活，因此两张 64 GB 卡不足以承载当前配置。此脚本未启用优化器 CPU 卸载，应先安排足够资源再试跑。

支持通过环境变量覆盖以下参数：`MODEL_PATH`、`HUMANGEN_ROOT`、`ZERO_WAM_SAVE_ROOT`、`TRAIN_SEED`、`NUM_STEPS`、`SAVE_INTERVAL`、`GRAD_ACCUM`、`LOAD_WORKERS`、`LEARNING_RATE`、`MASTER_PORT` 和 `PYTHON_BIN`。

每次运行保存启动命令、数据与模型来源、源码归档、`train.log`，以及由 rank 0 写入的 `metrics.jsonl`。指标包括视频/动作/IFP 损失、各 IFP 分项损失、梯度范数、学习率和是否跳过优化器更新。

默认试跑在第 10 步保存权重。现有训练 checkpoint 是 BF16 模型快照，**不包含完整优化器和随机数状态，尚不支持精确断点续训**。若需要在最后一步保存，应将保存间隔设为计划总步数的约数。

评估 ICL 时，应在相同训练 checkpoint 上比较正确人类示范、无示范和错误任务示范。仅看训练 loss 无法证明模型在使用示范；测试任务上的闭环成功率仍需单独评测。

## 对齐前的历史验证记录

实际 loader 已索引全部 2,149 个训练样本，没有静默过滤样本。43 个训练任务各读取一个样本，均通过张量、动作掩码及对齐检查。抽样轨迹在 padding 前包含 4,290–20,300 个主干 token，计入了人类上下文及带噪/干净数据流。

43 个任务的本地索引和 Arrow 缓存均已生成，但尚未逐一读取全部 2,149 个样本的张量内容。

数据准备、任务隔离和训练对齐相关的 17 项测试通过；启动脚本通过 shell 语法检查与 dry-run。未启动完整模型 NPU 训练。
