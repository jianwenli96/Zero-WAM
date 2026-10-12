# 自采 ALOHA 与 HumanGen 协同训练

## 数据契约与选择

原始目录 `/mnt/sfs_turbo/lijianwen/Datasets/aloha_icl_data` 保持不变。输出目录默认
`/mnt/sfs_turbo/lijianwen/Datasets/aloha_zerowam_v3`，可用 `ALOHA_PREPARED_ROOT` 修改。

原始数据有 50 条 30 Hz 轨迹，三个 480×640 相机，4 条训练人类视频和 2 条评测人类视频。
`action` 是 info.json 描述的 28 维绝对值：双臂 xyz+xyzw 四元数、两组六关节、两个夹爪。
没有 `observation.state`。转换保留原始单位，不把 action 推断成测量状态，不做相对化或毫米/米转换。
新版在整条 episode 上把 xyzw 归一化，并固定首帧符号、沿时间保持相邻四元数内积非负；
此操作不改变物理旋转，避免 q/-q 双覆盖导致回归目标跳变。切片和统计均在此操作之后。
泛化动作加载器要求 state/action 配对，所以 YAML 的 state 是 action 的显式读取别名，
并设置 `use_absolute: true`，不会用于计算状态差。模型槽位 0–25 对应原始前 26 维，
28–29 对应夹爪；26–27 填零且 loss mask 关闭。归一化与服务反归一化使用同一统计文件。

准备阶段检查全部机器人视频的帧数、fps、首尾帧。损坏相机对应的整条 episode 排除，
记录在 `rejected_videos.json`。本次发现 episode 7 的 cam_high 文件无法解码（moov atom missing）。
修复后用 `prepare --rebuild-metadata` 重新生成元数据并重新编码/检查。
编码期间还逐帧检查实际读取，发现中间损坏会失败，不会静默跳过。

固定 seed=42 从可用机器人 episode 中保留 5 条做验证，其余训练；先划分再切片。
每个训练 episode 固定随机配对一条 `human_videos/train` 视频；验证配对只使用 `eval` 视频。
本批数据为同一个 breakfast 任务，因此这样配对具有任务语义一致性。
扩展到多任务时应按任务分组抽样，不能全局随机匹配。

机器人每 4 帧抽一帧，129 个 RGB 帧组成一段（最大约 17 秒），编码成 33 个 latent 帧。
每 latent 帧对应 16 个 30 Hz 动作，与现有模型一致。相邻段只共享边界 RGB，
有效动作目标不重复，末尾不足 16 帧的部分不训练。三个视角分别 resize 为 224×288，
用模型对应的 Wan VAE 编码并归一化，加载器沿宽拼接。动作首块零 padding 与既有训练逻辑一致。

人类示范保留完整时间覆盖，最多均匀抽 129 帧，resize 为 320×480。
这与 HumanGen 的 latent 文件契约一致，但不保证时域分辨率相同：长视频的细节会丢失。
采用这一上限是为了限制长示范 attention 显存；后续可对任务阶段做更精细标注和配对。
机器人与人类文本使用同一段完整任务说明，经匹配的 UMT5 编码（512×4096）；
当前没有人工细分的局部动作文本。

`action_stats.json` 的 q01/q99/mean/std 只使用训练 episodes 完成上述四元数处理后的绝对动作计算；
验证数据及 HumanGen 不参与。各 HumanGen 来源保留自己的动作变换和统计量。

## 准备与检查

```bash
# 首次准备或补齐中断的编码；默认使用本机 8 张 NPU
bash script/prepare_aloha.sh

# 深入检查全部 latent，并逐项通过真正的训练 Dataset 加载
PY=/mnt/sfs_turbo/public/apps/miniforge3/envs/lingbot-vggt/bin/python
$PY script/check_aloha.py --deep --load-samples
```

可用 `PYTHON_BIN`、`NGPU`、`DEVICE_TYPE=cuda`、`MODEL_PATH` 调整运行环境。
编码分片、每文件原子保存，重跑仅复用来源与模型指纹一致的 latent；
来源指纹使用路径、文件大小和 mtime_ns（并非内容哈希），旁车记录编码版本、任务参数及输出文件大小。
修改后保留大小和时间戳的文件不在此轻量检查的保证范围内。不要在同一输出目录同时运行两批编码。
如果更换 VAE/T5 权重或预处理实现，应使用新的输出目录，不能复用旧 latent。
`encoding_jobs.json` 是完整预期文件列表；训练启动前检查缺失文件，防止加载器静默丢样本。

## 启动协同训练

```bash
bash script/train_dist.sh
```

默认 `CONFIG_NAME=aloha_train`、`MODEL_PATH=/mnt/sfs_turbo/public/ckpts/Zero-WAM/zero-wam-pretrain`，
不是 scratch 权重。保存到 `outputs/aloha_humangen_train_dist/<timestamp>`。
默认单机 8 卡，也兼容原集群 `MA_NUM_HOSTS/VC_TASK_INDEX/VC_WORKER_HOSTS/MA_NUM_GPUS`，
或显式 `NNODES/NODE_RANK/MASTER_ADDR/NGPU`；多节点仍需在每个节点执行脚本。

默认权重 `aloha:18,robotwin:7,agibot:7,robocoin:7,robomind:7,interna1:7,oxe:7`，
表示 ALOHA 30%，其余六组共 70%，不是按原始样本数量占比采样。
小集合被重复采样，故学习率降为 1e-5，先训练 5000 步、每 500 步保存，
每卡 batch=1，累积=1，warmup=200，机器人上限 33 latent 帧，保留 MCP/ICL 训练。
这组参数是待验证的微调起点，不能声称最优。

```bash
# 例如调低真机过采样至 20%，或修改训练步数/学习率
DATASETS='aloha:3,robotwin:2,agibot:2,robocoin:2,robomind:2,interna1:2,oxe:2' \
  bash script/train_dist.sh --learning-rate 1e-5 --num-steps 3000

# 旧的纯 HumanGen 训练仍可显式选择
CONFIG_NAME=robotwin_train bash script/train_dist.sh
```

优先对比 ALOHA 10%/30%/50% 和多个 checkpoint 的独立验证误差，检查过拟合与 HumanGen 遗忘。
短窗口训练、人类降采样以及原有按完整 30 槽位平均的 action loss 都可能影响效果；
未修改所有集合的 loss 标度，避免引入未经验证的全局训练行为变更。

## 开环推理验证

使用生产 `VA_Server` 的真实视频条件、ICL cache、去噪和反归一化过程，ALOHA 服务明确返回
`aloha_absolute_28_continuous_v2` 合约及统计量；客户端拒绝误连 Robotwin 服务或使用不一致统计量。
每个独立窗口 reset 缓存，只发送窗口首帧三视角及人类示范。
丢弃第一个 latent 对应的 padding 动作，默认评分接下来 16 个动作（约 0.53 秒）。
没有把未来机器人图像或真实动作传给模型，不调用机器人接口。
这是独立窗口短时预测，不是把未来观测当成闭环 rollout，也不覆盖长程控制累积误差。
可用服务 `--chunk-frames 4` 评分 48 个动作；客户端会跳过不足预测时长的尾段，并在报告中列出，避免长 horizon 评测中断。

训练 checkpoint 只有 transformer；先组合模型目录（以下路径替换成实际 checkpoint）：

```bash
MODEL_DIR=/mnt/sfs_turbo/lijianwen/Datasets/aloha_eval_model
BASE=/mnt/sfs_turbo/public/ckpts/Zero-WAM/zero-wam-pretrain
CKPT=/absolute/path/to/checkpoint_step_500
mkdir -p "$MODEL_DIR"
ln -s "$CKPT/transformer" "$MODEL_DIR/transformer"
for component in vae tokenizer text_encoder; do
  ln -s "$BASE/$component" "$MODEL_DIR/$component"
done

# 终端一：ALOHA 服务；必须使用同一份准备数据及动作统计
PY=/mnt/sfs_turbo/public/apps/miniforge3/envs/lingbot-vggt/bin/python
$PY -m torch.distributed.run --nproc_per_node=1 --master_port=29611 \
  -m evaluation.aloha.server --model "$MODEL_DIR" --port 29536

# 等终端一出现 server listening 后再运行终端二（加载模型可能需数分钟）
# 终端二：独立验证机器人 + eval 人类示范
$PY -m evaluation.aloha.open_loop --output outputs/aloha_open_loop --max-windows 20
# 训练集拟合检查（不能作为泛化成绩）
$PY -m evaluation.aloha.open_loop --split train --output outputs/aloha_open_loop_train
# 同窗口、同噪声 seed、不使用人类示范的消融
$PY -m evaluation.aloha.open_loop --no-icl --output outputs/aloha_open_loop_no_icl
```

`--human-latent` 可选择准备出的任意自采人类示范 `.pth`；优先用此方式保持训练/推理预处理一致。
直接走通用服务器 MP4 fallback 会使用不同的抽帧率，故此验证入口使用预编码示范。
服务每个窗口设置相同可复现 seed，报告模型路径、推理步数、窗口来源、原始预测/真值 npz、
四元数对齐等价符号后的每通道 MAE/RMSE、q01/q99 范围归一化 MAE（另保留 raw_normalized_mae）、关节/夹爪/位置误差、四元数范数与角度误差，
并给出“保持首个动作”的基线。位置指标是原始录制单位，确认单位前不要标为米。
归一化 MAE 对近常量通道可能敏感，需与各组误差一起看。

## 验证边界与后续真机工作

开环正确性检查不等于操控成功率。动作可能存在多种合理解，示范轨迹误差不能完全判断任务成功。
上机前仍需确认控制器使用末端目标还是关节目标、单位/坐标系、四元数次序、夹爪量程、
相机/控制时延及时间戳对齐；本次只保留 info.json 的字段语义和采集的时间对齐。
模型输出四元数未强制归一化，开环报告其范数；未来控制适配器必须根据实际控制协议处理。
目前不存在真实 observation.state，保持动作基线是录制 action 的基线而非状态估计。
生产服务历史 cache 使用预测动作，独立窗口评估不验证持续闭环的这一行为。

现有训练仍只保存模型权重，不完整保存 optimizer/sampler 状态，不能精确断点续训。
本任务提供训练与评测链路，不把未经完整微调的预训练模型成绩当成真机适配效果。

本次还修复了实际环境中发现的兼容问题：原始 episodes_stats 缺少 count，准备阶段现在从
parquet 重算标准逐 episode 统计；Ascend 的 attention mask 后端选择使用实际 tensor.device，
避免 transfer_to_npu 的 cuda 别名进入 FlexAttention，并显式清零 SDPA 全遮罩行输出及梯度。

## 首轮实现的历史运行记录（v2 数据，不代表 v3 验证）

已生成 750 个预期 latent（744 个机器人视角切片、6 个人类示范），全量深检通过。
真实 Dataset 逐项加载通过：44 条训练轨迹的 221 个窗口，5 条验证轨迹的 27 个窗口。
完整测试套件通过 121 项，日志在 `outputs/aloha_preparation/all_tests.log`。
数据检查日志在 `outputs/aloha_preparation/validation.log`。

真实预训练模型已通过单窗口开环服务冒烟测试，结果为
`outputs/aloha_open_loop_smoke/{metrics.json,predictions.npz}`。
历史测试使用 2 次去噪步、未经 ALOHA 微调的模型，只说明输入、ICL、推理和输出评分链路运行成功；
不得把它当成训练完成或控制效果达标。

已通过 `train_dist.sh` 完成 8 卡混合训练单步与 checkpoint 保存（进程退出码 0）：

```bash
OMP_NUM_THREADS=4 DATASETS='aloha:1,robotwin:1' \
ZERO_WAM_SAVE_ROOT="$PWD/outputs/aloha_train_smoke" \
  bash script/train_dist.sh --num-steps 1 --length-bucket-steps 0 \
    --load-worker 0 --disable-wandb --save-interval 1000
```

该测试使用默认 33 帧上限、全部 43 个 Robotwin 训练任务及 ALOHA，2370 个混合样本；
video/action/MCP loss 为 0.2092/0.0579/0.2984，梯度范数 0.12，无跳步。
单步位于学习率 warmup 起点，不能作为有效微调成果；它验证分布式前向、反向、优化器路径与保存。
checkpoint 在 `outputs/aloha_train_smoke/20261010_132832_897496/checkpoints/checkpoint_step_1`，
8 个 safetensors 分片完整。详细日志见 `outputs/aloha_preparation/train_smoke.log`。
额外开环训练集/无 ICL 测试保存在 `outputs/aloha_open_loop_train_smoke` 和
`outputs/aloha_open_loop_no_icl_smoke`。

上述历史测试未执行完整 5000 步微调，也未进行默认七来源混合的收敛/效果比较；
它只覆盖 ALOHA+Robotwin。二次审查的默认七来源测试见下文。
训练后请按正式 50 次去噪设置重新做多窗口评测，保留对应的动作统计文件再进行模型部署。


## 二次审查修正（2026-10-11）

首轮“完成”应理解为数据/训练/开环链路跑通，不能理解为表示设计和部署可靠性已充分验证。
复查在 50 条原始轨迹中发现 2440 次相邻 quaternion 内积 < -0.9 的近似反号跳变。
排除损坏 episode 7 后，训练集 2128 次、验证集 270 次；v3 连续符号处理后均为 0。
独立检查确认：物理旋转等价，位置/关节/夹爪逐值保持，q01/q99 与 v3 训练 parquet 重算一致。
采用 qw>=0 的逐帧处理仍会在跨 180° 处出现跳变，因此这里采用整条 episode 连续处理。
它仍是依赖历史的 quaternion 表示，尚未通过真机闭环检验，不能宣称最优。

v3 默认输出为 `aloha_zerowam_v3`；旧 `aloha_zerowam`、旧训练结果和历史评测保留。
旧 checkpoint 不能直接配新统计量使用。新训练会在 run 目录及每个 checkpoint 根保存
`aloha_action_contract.json`，ALOHA 服务检查该快照与所选准备数据一致。
若部署目录的 transformer 是指向 checkpoint 的 symlink，服务也会在其真实父目录寻找契约。
如果复制了 transformer 目录，请同时复制该 JSON。
基础预训练权重或旧 checkpoint 没有绑定文件，必须明确加 `--allow-unbound-model`，
且报告标记 `checkpoint_contract_bound=false`，不能将这种测试视为归一化契约验证通过。

```bash
# 推荐用 ALOHA 入口。evaluation/robotwin/launch_server_multigpus.sh 是 16 维 Robotwin 协议，不能代替它。
MODEL_PATH=/path/to/complete/aloha/model bash evaluation/aloha/launch_server.sh
# 仅用于未绑定基础模型的管线检查
MODEL_PATH=/mnt/sfs_turbo/public/ckpts/Zero-WAM/zero-wam-pretrain \
  bash evaluation/aloha/launch_server.sh --allow-unbound-model --steps 2
```

此外，v3 在 float32 下完成 RGB resize/归一化后才转 BF16，消除旧编码路径过早降精度造成的额外偏差。
编码恢复会验证视频、prompt、帧号、尺寸、编码版本和模型文件来源；启动前验证来源未变。
修复视频或增加数据后，已有验证 episode ID 保持不变；若验证 episode 本身变得不可用，
要求使用新目录显式重建评测划分，避免历史分数被悄悄改变。

开环仍是独立窗口验证；并没有覆盖持续多 chunk 的观测回传、cache 更新、控制时延、控制器限幅、
关节/末端两种输出的一致性或实物安全约束。默认每窗口只有 16 个动作（约 0.53 秒），
不能据此推断长任务成功率；可先用 --chunk-frames 4 做 48 动作的延长预测检查。
人类视频降采样到最多 129 帧、缺少阶段标注、绝对动作混合其他机器人的相对动作表示，
以及 ALOHA 30% 的过采样比例，仍需正式训练与独立验证决定，不能视为已证实最优。

准备目录是本项目专用的 LeRobot view，保留原始稀疏 episode ID；本项目按 episodes.jsonl 选择轨迹。
不要把 info.json 的连续 split 区间直接交给假定 episode ID 连续的其他工具；跨工具导出需要重新编号。

本轮实际验证记录保存在 `outputs/aloha_audit/`：

- 125 项回归测试通过；最后的缓存恢复调整后，8 项 ALOHA 专项测试再次通过。
- v3 的 750 个 latent 全部深检通过；真实 Dataset 逐项加载 221 个训练窗口和 27 个验证窗口通过。
- 七来源索引、首个样本及全部样本的长度成本检查通过。该独立检查的 ALOHA 使用 v2；v3 由上述逐项加载及下述训练另外覆盖。
- 通过默认 `train_dist.sh` 的七来源比例、33 帧上限、8 步长度分桶，在 8 张 NPU 完成两步训练，退出码 0。
  共 37,608 样本、2,041 个子数据集；为避免冒烟任务启动额外读取进程，设置 `--load-worker 0`。
  两步的 video/action/MCP loss 为 0.1772/0.0153/0.1344 和 0.2520/0.0506/0.4092，梯度范数 0.17/0.14，没有跳步。
  warmup 第二步实际学习率为 5e-8（日志显示的是 scheduler 更新后的下一步学习率）。
  保存后独立对比确认 `action_embedder.bias` 有 21 个 BF16 元素改变；这仅验证参数更新，不代表有效收敛。
- checkpoint 位于 `outputs/aloha_audit/train/20261011_004309_124286/checkpoints/checkpoint_step_2`，包含 8 个模型分片和动作契约快照。
- 默认 4 个 DataLoader worker 在 NPU 初始化后连续读取 8 个 ALOHA batch 通过，记录在 `loader_workers.log`；这不是四 worker 的完整七来源训练测试。
- 新 checkpoint 的实际服务与独立验证集开环通过：`open_loop/metrics.json`、`open_loop/predictions.npz`。
  3 个窗口来自 episode 8/41/48，每个输出 48 个动作，`checkpoint_contract_bound=true`。
  此次仍只用 2 次去噪；归一化 MAE 为 0.22688，保持首个动作的基线为 0.05653，模型效果没有达标。
  只能认定保存/重载/ICL/统计量/动作顺序/评分链路运行成功，不能认定已经完成微调或可上机。
  首次客户端在服务加载完成前连接失败；等待监听后重试成功，已补充服务日志及等待说明。

复现本轮训练冒烟命令：

```bash
OMP_NUM_THREADS=4 ZERO_WAM_SAVE_ROOT="$PWD/outputs/aloha_audit/train" \
  bash script/train_dist.sh --num-steps 2 --save-interval 2 --load-worker 0 --disable-wandb
```
