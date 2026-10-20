# PiperPika 混合后训练与 RTC

代码目录：`/mnt/sfs_turbo/guzhai/code/Zero-WAM-lijianwen`。
本实现接入旧目录的 PiperPika 数据准备、20%/80% 混训和 cosine 学习率设置，并复用当前 RTC 协议。

## 数据和训练设置

PiperPika 原始动作有 28 维；当前训练只监督其中的 12 个关节和 2 个夹爪，保留记录单位和绝对目标语义。
模型 30 维槽位中，关节使用 14～25、夹爪使用 28～29，其他通道不参与动作损失。
部署返回 `[14,F,A]`，顺序为左臂 6 关节、右臂 6 关节、左夹爪、右夹爪。

默认数据根为 `/mnt/sfs_turbo/lijianwen/Datasets/piperpika_10_16_zerowam_joint_v2`；
按人类示范组划分，训练 108 条机器人 episode、132 个原生片段，验证 12 条、14 个片段。
这不是 ALOHA 的 breakfast 数据；统计和 checkpoint 契约分别保存，不能互换。

| 设置 | PiperPika 默认值 |
|---|---|
| 初始化 | `zero-wam-pretrain` |
| 数据混合 | `piperpika:5,agibot:4,robocoin:4,robomind:4,interna1:4,oxe:4` |
| 抽样概率 | PiperPika 20%；其余五个来源各 16%；不含 RoboTwin |
| 学习率 | `2e-6`，warmup 100 步，cosine 降至 10% |
| 参数更新次数 / 保存间隔 | 5000 / 500 |
| 每卡 batch / 累积次数 | 1 / 1 |
| 视频 / 动作 / ICL / MCP | 全部保留训练 |
| 训练 chunk 分组 | 随机 1～4 个 latent；机器人窗口最多 60 个 latent |

`20+20` 是该配置的普通推理去噪默认值，与 cosine 学习率以及训练的 1000 步噪声网格是不同设置。
`script/train_dist.sh` 的默认配置仍是 `aloha_train`；显式选择 `piperpika_train` 才使用上表设置。

## 与旧脚本一致的原生布局训练

```bash
cd /mnt/sfs_turbo/guzhai/code/Zero-WAM-lijianwen
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

CONFIG_NAME=piperpika_train PREPARE_DATA=0 \
MODEL_PATH=/mnt/sfs_turbo/public/ckpts/Zero-WAM/zero-wam-pretrain \
ZERO_WAM_SAVE_ROOT=/mnt/sfs_turbo/guzhai/modelarts_outputs/piperpika_native \
bash script/train_dist.sh --disable-wandb
```

已有完整 prepared 数据时用 `PREPARE_DATA=0`，启动前检查 provenance 和全部 latent 旁车。
首次准备可用 `PREPARE_DATA=1`：全局 rank 0 生成 metadata，各卡编码互不重叠的任务，最后检查真实加载器。
默认读取原始目录 `/mnt/sfs_turbo/lijianwen/Datasets/piperpika_10_16`；可设置 `PIPERPIKA_SOURCE_ROOT` 和 `PIPERPIKA_PREPARED_ROOT`。
不要让多个训练作业同时准备同一个输出目录。

多节点与可见 NPU 识别沿用 `distributed_env.sh`；每节点执行一次入口。公共 `python_runtime.sh`
在 NPU 探测和模型导入前预加载所选 Python 环境的 `libstdc++.so.6`，保留既有 `LD_PRELOAD`。
训练和两种机器人推理入口均复用此处理。

按用户要求，数据索引已恢复原来的 `multiprocessing.Pool`，DataLoader 使用原来的默认进程上下文。
ModelArts RTC 入口恢复 `--init-worker 2 --load-worker 2` 和 `OMP_NUM_THREADS` 默认 4。
启动命令显式加 `OMP_NUM_THREADS=1`，单独验证 CPU 单线程配置；这不改变可见 NPU 数量、
混合比例、有效 batch、损失或 RTC 时间布局。

若旧作业停在 `Loading ICL dataset indexes: 0/187` 并打印 `popen_fork.py` 的多线程 fork 警告，
停止旧作业后使用下面带 `OMP_NUM_THREADS=1` 的命令重新提交；正在运行的 Python 进程不会自动应用文件修改。
新日志会记录 `Loading dataset source`、任务完成数量、`Dataset source ready` 和
`Training data loader ready`，第一步参数更新完成后出现 `Training metrics`。
每次作业同时保存 `effective_config.json` 和 `runtime_config.json`，记录实际参数、CPU 线程数和加载 worker 数。

对旧目录的受控对照已复现：CPU 并行运行时预热后，旧代码在 `OMP_NUM_THREADS=1` 时通过，
设为 4 时卡在 fork 子进程返回 Tensor 的 `_share_fd_cpu_`；修复前的新代码同样阻塞，
曾测试的线程池版本在 4 线程下通过，该替代实现现已撤回。因此警告本身不能判定死锁，关键是多线程运行时与 fork 的组合。
PiperPika 只有一个任务，顺序加载先完成；187 个 AgiBot 任务随后进入池，才触发这一问题。

2026-10-18 的历史线程池版本验证：先在空闲 NPU 初始化运行时，再加载 PiperPika 和全部 187 个 AgiBot 任务，
188 个任务全部命中索引和 Arrow 缓存，耗时 31.84 秒；两个来源的真实 batch 和当时的 `spawn`
DataLoader 均通过。该记录用于排查参考，不能作为当前恢复版本的完整训练验证。64 卡 ModelArts
作业须重新提交并观察首个 `Training metrics`。历史记录见
`/mnt/sfs_turbo/guzhai/tmp/zerowam_loader_fix_20261018/loader_probe.json`。

## 面向 10 Hz、A=32 的 RTC 后训练

ModelArts 或 Notebook 使用同一入口，通过配置名切换数据：

```bash
OMP_NUM_THREADS=1 CONFIG_NAME=piperpika_train PREPARE_DATA=0 \
bash /mnt/sfs_turbo/guzhai/code/Zero-WAM-lijianwen/script/train_modelarts_rtc.sh
```

PiperPika 模式默认从预训练模型开始，保留上述混合比例、`2e-6` 和 cosine，
同时传入 `--action-stride 3 --actions-per-latent 32 --action-sampling-datasets piperpika`。
默认输出 `/mnt/sfs_turbo/guzhai/modelarts_outputs/piperpika_rtc_a32_stride3`。
先检查容器挂载和写权限；10 步检查可在命令末尾增加 `--num-steps 10 --save-interval 10`。

也可通过 `DATASETS=piperpika:1` 选择单来源；单来源自动关闭长度分桶，多来源默认分桶窗口 8 步。
PiperPika 为 30 Hz，所以 stride 3 对应 10 Hz。
RTC 入口只对 PiperPika 改变时间布局，其余回放数据保留自己的原生采样与动作块长度。
实际回放文件中存在 A=4、8、12 等不同布局，OXE 还包含 5 Hz 数据，不能强行按原生 A=16 处理。
数据加载器携带来源标识，使同一模型交替处理 PiperPika 的 A=32 和回放数据的原生变长动作 token；
每卡 batch 仍为 1。长度分桶按各来源的时间配置计算。
非 RTC 通用训练默认不改变任何来源的时间布局；已有 ALOHA 命令的默认行为保持原样。

PiperPika 每个保留 latent 的 32 个目标覆盖 3.2 秒，训练和反馈均选择原生 `z0,z6,z12,...`。
VAE 权重、编码频率和维度不变。
checkpoint 同时保存 `piperpika_action_contract.json` 和版本 3 的动作时间契约，
其中 `datasets=["piperpika"]` 标明改变时间布局的部署来源。

每份完整 checkpoint 约 22 GB，默认 5000 步保存 10 份约需 220 GB；输出盘须有足够空间，
也可显式增加 `--save-interval`。当前脚本不会自动清理旧权重。

## 推理和 RTC 客户端

将选中的 checkpoint 的 `transformer/`、`piperpika_action_contract.json` 与底座的
`vae/`、`tokenizer/`、`text_encoder/` 组合成完整 `MODEL_PATH`。
匹配训练后的时间布局会自动读取；不要用旧原生权重的时序探测代替新布局训练。

```bash
NGPU=1 MASTER_PORT=29761 PORT=29762 \
bash evaluation/piperpika/launch_server.sh \
  --enable-rtc --steps 3 --action-steps 10 --chunk-frames 4 \
  --rtc-execution-horizon 64 --rtc-control-hz 30 \
  --rtc-attn-window 4 --rtc-icl-guidance-scale 1
```

未开启 RTC 时保持 `20+20` 默认值。RTC 默认关闭；开启后保留启动预热、延迟 buffer、
soft-mask/VJP guidance、实际动作历史提交和超时失败检查。
14 维输出的关节进行线性插值，夹爪保持左端目标；不对关节执行四元数 SLERP。
ALOHA 的 28 维位姿插值路径保持可用。

另一终端可运行不连接机器人的 30 Hz 回放：

```bash
/mnt/sfs_turbo/public/apps/miniforge3/envs/lingbot-vggt/bin/python \
  -m evaluation.piperpika.rtc_replay \
  --port 29762 --episode 15 --ticks 600 --warmup-rounds 3 \
  --output /mnt/sfs_turbo/guzhai/tmp/piperpika_rtc_replay
```

回放记录的图像独立于模型动作，只验证软件和时间条件，不表示实机操控成功率。
原生 `open_loop` 入口仅用于原生 A=16、stride=1 的独立窗口评估。

## 时间预算

使用当前 20% 乘数预留、50 ms 余量以及 stride>1 的一个插值锚点：

| 布局 | H / s | 最大原始请求延迟 |
|---|---|---:|
| 原生 30 Hz、A=16、F=4 | 64 / 32 | 0.847 秒 |
| 10 Hz、A=32、F=4 | 128 / 64 | 5.208 秒 |

判断条件仍为 `d <= s <= H-d`。学习率、混合比例不会自动缩短执行时间或让 20+20 满足 RTC；
需要用部署权重、实际人类示范和设备重新测量。

## 2026年10月18日验证结果

单张空闲 NPU 5（910B3、64 GB），`action_stride=3、A=32、F=4、s=64`，原始请求预算 5.208 秒：

| video + action | 人类 latent 数 | 实测延迟 | RTC 结果 |
|---|---:|---|---|
| 3+10 | 22 | 在线 4.895、4.862 秒 | 600 个控制 tick，2 次换队列通过 |
| 3+10 | 25，当前数据最长示范 | 在线 4.328、4.361 秒 | 600 个控制 tick，2 次换队列通过 |
| 20+20 | 22 | guided 预热 9.289、9.008、9.041 秒 | 预算不满足，控制开始前报错 |

两组通过的输出均为 `[600,14]`，全部有限，实际夹爪保持检查通过。
不同轮次负载与热身不同，不应把两组 3+10 的差异解释成上下文变长会加速。
25 帧压力测试显式替换了验证 episode 15 的人类示范，只用于时间测试。

训练侧通过 157 项 CPU 检查（另4项未选择），含不同动作块混合的小模型真实反向/优化器与 MCP。
真实加载器保留 115 个 PiperPika 片段、过滤 17 个过短片段；全部 438 个机器人 latent 的独立动作/时间对齐审计通过。
本次没有启动新的 5000 步训练。实测权重来自旧目录的原生布局 step5000，显式时间覆盖仅用于时延探测，
新 A=32/10 Hz 的动作质量需要匹配后训练和任务评测。

完整结果：`/mnt/sfs_turbo/guzhai/tmp/zerowam_piperpika_rtc_20261018/rtc_validation.md`；同目录保存 JSON 与原始日志。

## 2026年10月20日延迟优化

本轮使用匹配 A=32、stride=3 时间布局的 PiperPika step5000 权重：
`/mnt/sfs_turbo/guzhai/tmp/zerowam_posttrain_eval_20261019/model_step5000`。
保持 video 3 步、action 10 步、F=4、128 个动作目标，以及现有 soft-mask/VJP guidance。

此前的统一缓存保留索引、连续切片、请求内 RoPE / mask / 文本投影复用已经启用。
本轮新增各层 cross-attention 的静态文本 K/V 复用：只在 RTC 模型被冻结、处于 eval 且文本无需梯度时复用，
缓存随请求结束释放；文本或权重原地变化会使缓存失效。动作输入相关计算和输入 VJP 保持动态。
RTC 默认启用此项（`job_config.rtc_cache_text_kv=True`），普通训练和未开启 RTC 的推理保持原有路径。

在同一张 NPU 0、同一份历史缓存和随机种子上，交错测量四组配置，每组预热后测 4 次。
下表只计 video + action，不包含反馈 VAE、历史更新、IPC 或交接等待：

| 配置 | video + action 中位数 |
|---|---:|
| 原实现，已含既有索引优化 | 3.172 秒 |
| 额外复用网格索引并避免 padding 类型标量同步 | 3.190 秒 |
| 仅新增文本 K/V 复用，当前默认 | 3.075 秒 |
| 两项新增复用同时开启 | 3.076 秒 |

文本 K/V 复用约节省 0.098 秒（3.1%），该固定上下文测试的峰值已分配显存增加约 189 MiB。
额外网格索引复用没有测得收益，因此 `job_config.rtc_stream_layout` 默认关闭。
确定性模式下，原实现与两项复用同时开启的完整视频和动作输出最大差异均为 0；
51 项 CPU 检查通过，覆盖重复 action VJP、CFG 元数据和缓存失效。
原始配对结果：`/mnt/sfs_turbo/guzhai/tmp/zerowam_rtc_latency_opt_20261020/paired_npu0/report.json`。

另用两张完整模型副本（NPU 0、3）测试更早的固定交接，维持 s=32 个目标，即每 3.2 秒发起一次请求：

```bash
cd /mnt/sfs_turbo/guzhai/code/Zero-WAM-lijianwen
source /usr/local/Ascend/ascend-toolkit/set_env.sh
MODEL_PATH=/mnt/sfs_turbo/guzhai/tmp/zerowam_posttrain_eval_20261019/model_step5000 \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
bash script/probe_piperpika_rtc_pipeline.sh \
  --devices 0 3 --ticks 600 --check-equivalence --handoff-latency 4.4 \
  --output /mnt/sfs_turbo/guzhai/tmp/zerowam_rtc_latency_opt_20261020/two_npu_fixed4p4
```

| 时间指标 | 此前固定 5 秒交接的回放 | 本轮固定 4.4 秒交接的回放 |
|---|---:|---:|
| 观测到新计划提交 | 5.000 秒 | 4.400 秒 |
| 观测到首个可修改动作目标 | 5.2 秒 | 4.9～5.0 秒 |

交接设置变化带来的 0.6 秒缩短不能归因于上述 0.098 秒的计算优化。
本轮原始观测到结果就绪为 3.967～4.129 秒；实际交接前仍等待 0.271～0.433 秒。
RTC 保留 20% action 预算、50 ms 余量和右侧插值锚点，冻结前 49～50 个目标，
因此“4.4 秒提交”不等于“4.4 秒后动作就响应新观测”；首个可修改的目标仍在 4.9～5.0 秒。

20 秒回放完成 600 个控制 tick、4 次计划提交；提交间隔约 3.2 秒，最大控制线程迟到 26.1 ms，
最小剩余动作覆盖 5.233 秒，已提交前缀、绝对时间对齐、实际执行历史和队列覆盖检查通过。
结束时还有 2 个在途请求，随测试停止，未计为提交。
这是独立的双 NPU 实验调度器，只验证录制图像的软件和时序；没有改动生产单请求 RTC 的交接默认值，
也不构成实机任务成功率或最坏延迟保证。
完整结果：`/mnt/sfs_turbo/guzhai/tmp/zerowam_rtc_latency_opt_20261020/two_npu_fixed4p4/report.json`。

## 双 NPU 缩短 s 的实验接口

2026-10-20 使用同一份 step5000、两张 910B3（NPU 0、3），保持 video 3 步、action 10 步、
F=4、每 latent 32 个动作、stride=3。**本轮最低重复通过的更新间隔为 s=28 个目标，即 2.8 秒。**
每次仍预测 128 个目标，覆盖约 12.8 秒；每 84 个 30 Hz 控制 tick 发起一次重规划。

原增量模式按整段 32 个动作提交反馈，因此最小 s 为 32，不能只把启动间隔改成 28。
新增 `--history-mode snapshot` 后，每个请求从真实时间窗口重建机器人上下文：

1. 请求观测时刻为 T，取已经捕获的 T−6.4、T−3.2、T 三个视频锚点。
   控制刚开始、没有足够历史时，只取可得的完整区间；最早一次可以仅使用当前观测与初始 PAD。
2. 窗口内仍每 4 个原生控制 tick 输入一张图像；相邻锚点之间输入 24 张反馈图像，
   经过原生 VAE 得到 6 个 latent，取末个 latent，与该区间实际执行的 32 个动作配对。
   VAE 权重、视频采样间隔、3.2 秒动作/latent 时间布局均不改变。
3. 每次保留不可变的人类示范缓存，重建该窗口的机器人 VAE/KV；机器人位置编码使用窗口内相对坐标，
   返回动作在控制器中映射到 T 开始的绝对时间。不同请求的 T 可以相隔 2.8 秒。
4. 视频生成完成后才绑定控制器最新已提交计划；action 解码与提交保持串行。
   冻结前缀包含延迟预留和插值右锚点，提交时验证版本及前缀逐元素相同，已发出的控制命令不可修改。

这里的时间检查属于多在途请求调度：每张卡必须赶上 2s=5.6 秒后的再次使用，串行 action 与交接必须赶上
相邻 2.8 秒更新。冻结 56～61 个目标可以大于 s=28，因为它绑定的是最新已提交计划，
可重叠区间为 H−s=100 个目标；同时逐次验证冻结前缀不超过重叠区间、右锚点未被改动、全局队列不断供。
因此不能直接套用单请求的 `d<=s` 判断，也不能只凭 `d<2s` 宣称通过，实际验证需包含上述全部条件。

该窗口重建策略与原增量 VAE/KV 状态不同，需要单独评估动作质量；采样步数与预测长度相同不意味着策略输出等价。
这是独立的双卡实验接口，生产单请求客户端的 `RTCConfig` 仍要求整 latent 区间，不能仅将其
`--rtc-execution-horizon` 改成 28 来替代此调度器。

本轮也修正了控制路径的阻塞：

- `--shared-recording` 将预加载 RGB 存为只读 mmap，IPC 传图像时间索引及动作数组。
  在线回放只选择截至请求时刻的捕获；离线预热上下文在控制开始前丢弃。
- action 文件、JSONL 和逐次提交日志统一在计时循环结束后写出，避免共享盘落盘阻塞 30 Hz 控制。
- 在实际发出控制命令前测量迟到，超过一个控制周期（33.33 ms）立即失败。
  动作覆盖统计扣除不可继续插值的最后锚点。

最终版本的实卡结果：

| 模式 / s | 固定观测到提交延迟 | 验证结果 |
|---|---:|---|
| 原增量模式 / 3.2 秒 | 4.4 秒 | 600 tick、4 次提交通过 |
| 新窗口模式 / **2.8 秒** | **5.4 秒** | **3 次独立 20 秒回放全部通过，共 1800 tick、15 次提交** |
| 新窗口模式 / 2.7 秒 | 5.3 秒 | 错过第 3 个请求交接；早期版本曾单次通过，未作为可用下限 |
| 新窗口模式 / 2.6 秒 | 5.2 / 5.1 秒 | 分别出现无空闲副本、错过交接 |
| 新窗口模式 / 2.5 秒 | 5.0 秒 | 无空闲副本 |

三次 2.8 秒回放最大控制迟到分别为 1.615、16.434、0.690 ms，最小有效插值覆盖均为 4.533 秒。
独立审计从保存的 bootstrap 计划开始，重建全部提交与控制命令：1800 条命令、15 次冻结前缀与计划版本检查均逐元素一致。
18 项 CPU 检查通过，覆盖非整 latent 的绝对时间窗口、共享图像一致性、串行 action 的失败边界、
控制超时与慢文件写入隔离。

同一轮中，原增量模式观测到首个可修改目标为 4.6～4.9 秒；新窗口模式早期为 5.6 秒，
完整历史时为 5.9～6.1 秒。因此 2.8 秒提高了未来计划更新频率，也增加了历史重建成本和观测到动作目标的延迟。
若优先缩短新观测的响应时间，当前原增量 3.2 秒模式更合适；2.8 秒用于研究更密集的计划更新。

复现最低重复通过的配置：

```bash
cd /mnt/sfs_turbo/guzhai/code/Zero-WAM-lijianwen
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
MODEL_PATH=/mnt/sfs_turbo/guzhai/tmp/zerowam_posttrain_eval_20261019/model_step5000 \
bash script/probe_piperpika_rtc_pipeline.sh \
  --devices 0 3 --video-steps 3 --action-steps 10 \
  --history-mode snapshot --snapshot-history-intervals 2 \
  --receding-actions 28 --handoff-latency 5.4 --shared-recording \
  --ticks 600 --output /mnt/sfs_turbo/guzhai/tmp/piperpika_rtc_s28_replay
```

固定交接延迟、最大历史窗口和延迟 buffer 在本轮测试中显式记录；20% action 预留、50 ms 余量未调小。
3 次 20 秒测试不是连续 60 秒测试，也不能作为最坏延迟保证。以上结果只验证录制 RGB 的软件时序，
尚未评估新窗口策略的闭环任务成功率。

最终逐轮结果：`/mnt/sfs_turbo/guzhai/tmp/zerowam_rtc_min_s_20261020/final/search_results.json`。
独立动作审计：同目录 `artifact_audit.json`；各 `trial*/` 保存 bootstrap、生成动作、已发出动作、控制迟到和报告。
较早的直接 RGB 传输及同步落盘测试保留在上级目录，判断下限时以 `final/` 的严格验证为准。

## RTC 与普通预测相对 GT 的质量对照

2026-10-20 使用匹配时间布局的 step5000、全部 12 条验证 episode，保持 3+10、F=4、A=32、
10 Hz 目标、30 Hz 控制、s=2.8 秒及相同随机种子。此处“普通预测”指同一模型关闭 action VJP/前缀纠偏，
依然按绝对时间丢弃已过期目标，提交时只保留已经用过的控制目标和插值右锚点。

本次为离线固定时间对照：使用已经测过的 5.4 秒交接时钟，RTC 首次冻结 56 个目标，随后冻结 61 个目标，
普通分支仅保留 55 个已经使用的目标/锚点。首轮 action 绑定设在观测后 0.8 秒，后续设在 2.6 秒，
与前一计划提交重合。普通分支也使用同一交接时钟，尚未利用关闭 VJP 后更快推理的优势。
该离线评估用于隔离算法质量，不是所有验证 episode 都通过实际墙钟 RTC 的证明。

前缀与反馈动作全部来自模型自己的输出，GT 仅供评分；验证集动作经 PyArrow 26.0.0 单独导出。
评分使用训练集 `q99-q01` 归一化的绝对误差，平均全部 14 个使用通道，越低越好。
只计 prepared `action_config` 的真实区间，关节按生产逻辑线性插值、夹爪保持，按 GT 的 30 Hz 时间戳评分。

| 对照范围 | 普通预测 nMAE | RTC nMAE | RTC−普通，百分点 |
|---|---:|---:|---:|
| 整段实际发出的动作，12 条、5456 个控制命令 | 9.273% | 9.382% | +0.109 |
| 首次交接之后，11 条、2510 个控制命令 | 10.618% | 10.854% | +0.236 |
| 同上下文、视频和噪声，仅开关纠偏，排除冻结前缀 | 11.622% | 11.581% | −0.041 |

首次交接后，RTC 的动作误差相对增加 2.22%；同上下文的单次新预测区间误差相对减少 0.36%，两者非常接近。
后者按每次 proposal 的有效新目标评分，共 1370 个目标，预测窗口之间可以重叠；
前两行按整段唯一控制命令评分，不能将它们作为同一统计范围直接比较。
连续结果同时包含冻结/旧计划延续、后续实际动作历史变化等影响，不能将误差变化全部归因于纠偏计算本身。

12 条轨迹中，RTC 整段误差更低 4 条、更高 7 条、相同 1 条；相同的一条在首次交接前就结束，未触发 RTC 纠偏。
逐条变化范围为 −0.685～+0.993 个百分点，方向并不一致。按 6 个人类示范组进行探索性配对 bootstrap，
整段误差差值的 95% 区间约为 [−0.171,+0.354] 个百分点，首次交接后约为 [−0.357,+0.847]，均跨过 0。
这些结果没有证明 RTC 的 GT 动作误差稳定改善或稳定恶化。

接续方面，首次交接后、按训练范围归一化并平均 12 个关节的相邻控制指令变化，
P99 从普通分支的 2.832% 降为 RTC 的 1.188%（本次样本约减少 58%）；GT 对应值为 0.855%。
这是步间变化指标，较小的变化本身不能证明任务执行更正确。

验证包括 23 项 CPU 检查、两张 NPU 上普通采样与零权重 RTC 的输出最大差异均为 0，
以及从保存的 bootstrap 重建两分支全部 10912 条控制命令，逐元素一致。
之前的 4.622% 表测的是独立窗口首个 chunk 的前 0.53 秒，当时没有旧计划、纠偏尚未介入，
不能用它代表连续 RTC 的质量或与本表整段误差直接比较。

复现实验入口为 `python -m evaluation.piperpika.rtc_quality`，通过 `--device`、`--episodes`、
`--model`、`--root`、`--gt`、`--output` 选择设备、验证轨迹和安全导出的 GT。
本轮 GT 为 `/mnt/sfs_turbo/guzhai/tmp/zerowam_rtc_gt_comparison_20261020/ground_truth.npz`，
`ground_truth_metadata.json` 记录读取版本、真实区间、归一化定义和校验和。

结果目录：`/mnt/sfs_turbo/guzhai/tmp/zerowam_rtc_gt_comparison_20261020/`。
`comparison_report.json` 保存汇总、实验限制与独立核对；`per_episode.csv` 保存每条误差；
`episode15_gt_vs_rtc.png` / `.svg` 对照固定选取的两个关节、夹爪与逐帧误差。
录制 RGB 不会响应模型动作；本次只有一组配对随机种子，尚无真实机器人或仿真闭环任务成功率。

## s=3.2 秒与 s=6.4 秒的 GT 质量补测

2026-10-20 对同一份 step5000、同样 12 条验证轨迹补测两种部署配置，继续使用 video 3 步、
action 10 步、F=4、A=32、stride=3、10 Hz 动作目标和 30 Hz 控制。每个分支评分 5456 条唯一有效控制命令，
使用同一份由 PyArrow 26.0.0 导出的 GT 和训练集 `q99-q01` 归一化，越低越好。

两种配置采用各自已经实测的时序，普通预测与 RTC 在每种配置内使用相同交接时钟：

| 配置 | 历史缓存 | 固定观测到提交 | RTC 冻结目标数 | 首次提交 / 首个可修改目标时间 |
|---|---|---:|---|---|
| `s32_dual`：每 3.2 秒重规划，双副本调度 | 每次增量更新 32 个已执行动作与 1 个 latent | 4.4 秒 | 依次 46、46、49、48 | 7.6 / 7.8 秒 |
| `s64_single`：每 6.4 秒重规划，单请求调度 | 每次在同一 phase 更新 64 个已执行动作与 2 个 latent | 5.0 秒 | 依次 58、61 | 11.4 / 12.2 秒 |

3.2 秒使用 `zerowam_rtc_min_s_20261020/final/trial00_incremental_s32_handoff4.4/report.json`
的冻结轨迹；绑定时间由实际 action 就绪时间减 action 用时推得，再向上取整到 30 Hz tick。
6.4 秒使用 `zerowam_posttrain_eval_20261019/rtc_step5000/rtc_replay_report.json` 中的
58/61 目标预留，把此前 4.895/4.995 秒延迟统一按 5 秒交接。
这些数值是离线回放的固定时序模板，并非在每条轨迹上重新校准的延迟 buffer，也不是动画示意的固定 62 目标。
虚拟控制时钟从共同初始预测可用后开始，模型加载和首轮预测耗时不计入动作质量评分。

普通分支关闭 VJP 和前缀纠偏，只在提交时保留已经用于控制的目标及右侧插值锚点：
3.2 秒组保留 45 个，6.4 秒组保留 51 个。前缀和历史动作全部来自各分支模型实际发出的命令，GT 不进入模型。
同一轨迹在两种配置中的 bootstrap 完全相同；后续两分支各自维护历史。

| 评分范围 | 3.2 秒普通 | 3.2 秒 RTC | 6.4 秒普通 | 6.4 秒 RTC |
|---|---:|---:|---:|---:|
| 整段有效控制命令，各 5456 条 | 8.401% | 8.680% | 9.341% | 9.541% |
| 各自首次提交以后，分别 2720 / 1454 条 | 9.041% | 9.601% | 10.328% | 11.080% |
| 同一时间范围：12.2 秒以后，各 1190 条 | 8.998% | 9.097% | 10.495% | 10.474% |
| 同上下文、视频和噪声，只开关纠偏，排除冻结前缀 | 9.696% | 9.927% | 10.383% | 10.406% |

整段 RTC 相对普通误差分别增加 **0.279 / 0.200 个百分点**，相对增加 3.32% / 2.15%。
各自首次提交后，相对增加约 6.19% / 7.28%；6.4 秒组在首次提交到首个可修改目标之间，
仍较长时间延续旧动作。只从 12.2 秒以后的共同时间范围比较，6.4 秒 RTC 与普通预测接近，
3.2 秒 RTC 误差比 6.4 秒 RTC 低约 1.377 个百分点。
同上下文对照按 proposal 的新目标评分，可以重复计算不同 proposal 的重叠目标，不能与整段唯一控制命令直接混用。

3.2 秒组整段误差改善 2 条、变差 10 条；6.4 秒组改善 6 条、变差 5 条、相同 1 条，
相同的一条在首次提交前结束。按 6 个人类示范组进行探索性配对 bootstrap，
整段 `RTC−普通` 的 95% 区间分别为 `[+0.118,+0.559]` 和 `[-0.069,+0.662]` 个百分点。
该样本中，3.2 秒 RTC 的误差代价方向较一致，6.4 秒组的变化方向仍有不确定性。

接续平滑度方面，各自首次提交后，12 个关节按训练范围归一化的相邻控制命令变化 P99：
3.2 秒普通 `2.664%`、RTC `1.085%`、GT `0.822%`；
6.4 秒普通 `2.175%`、RTC `1.244%`、GT `0.792%`。
RTC 减少较大的步间变化，但并未在此次测试中降低整段 GT 模仿误差。

目前优先采用 **3.2 秒增量历史 RTC** 继续做闭环验证：它的整段误差为 8.680%，
低于此前 2.8 秒窗口重建的 9.382% 和本轮 6.4 秒的 9.541%，共同时间范围也优于 6.4 秒。
各组同时改变了历史组织、观测延迟和冻结预留，不能把差异全部归因于重规划间隔 s。
若只考虑此次 GT 误差，3.2 秒普通预测仍比同组 RTC 略低；实际执行还需权衡接续平滑度。

本轮验证包括 30 项 CPU 检查，独立于 `PlanLedger` 重建两种配置、两分支全部 **21824** 条命令，
冻结前缀、绑定版本、插值右锚点及安全 GT 一致性均通过。
另在 NPU 4 对 6.4 秒的首个重规划做生产 `RTCSession` 与评估路径配对：
48 张反馈图像、2 个保留 latent、64 个动作在单个 phase 更新，视频、缓存元数据逐元素一致，动作最大差异为 0。
这项等价性检查覆盖一个实际上下文；12 条轨迹的质量回放分摊在 NPU 0、3 完成，
没有声称全部轨迹通过双卡或单卡实际墙钟的截止时间验证。

复现质量评估时，在已配置 CANN 和 Python runtime 的环境运行：

```bash
python -m evaluation.piperpika.rtc_quality \
  --device 0 --episodes 2 3 14 15 30 31 \
  --profiles s32_dual s64_single \
  --model /mnt/sfs_turbo/guzhai/tmp/zerowam_posttrain_eval_20261019/model_step5000 \
  --root /mnt/sfs_turbo/lijianwen/Datasets/piperpika_10_16_zerowam_joint_v2 \
  --gt /mnt/sfs_turbo/guzhai/tmp/zerowam_rtc_gt_comparison_20261020/ground_truth.npz \
  --output /mnt/sfs_turbo/guzhai/tmp/zerowam_rtc_gt_s32_s64_20261020/npu0
```

另一张卡设置 `--device 3 --episodes 34 35 80 81 94 95`，输出目录改为同级 `npu3`。
汇总入口为 `python -m evaluation.piperpika.rtc_quality_summary`，使用 `--root` 选择上述结果根目录，
`--gt` 选择安全 GT，`--metadata` 选择同级 `ground_truth_metadata.json`。

结果目录：`/mnt/sfs_turbo/guzhai/tmp/zerowam_rtc_gt_s32_s64_20261020/`。
`comparison_report.json` 为汇总，`per_episode.csv` 为逐条结果，`quality_comparison.png` / `.svg` 为指标对比，
`episode15_comparison.png` / `.svg` 展示与 GT 的动作轨迹，`production_parity.json` 为生产路径等价性检查。
该结果仍是单组随机种子、录制 RGB 的离线对照，不是机器人闭环成功率；普通分支也尚未利用关闭 VJP 后更早交接的优势。

## 四 NPU 的最小 s 搜索与持续时序验证

2026-10-20 在四张空闲 910B3（0、3、4、6）运行完整模型副本，保持 step5000、video 3 步、
action 10 步、F=4、每 latent 32 个动作目标、stride=3、10 Hz 目标及 30 Hz 控制。
每次仍预测 128 个目标，名义覆盖约 12.8 秒；本轮仅使用现有 action RTC。
非整 latent 的重规划沿用双卡实验的 snapshot 模式，每轮重建最多两个完整的 3.2 秒真实历史区间。
延迟乘数 1.2、50 ms 预留、插值右锚点和最新计划绑定检查均保持原规则。

**本轮未验证出可靠低于 2.8 秒的 s。通过完整筛选的组合为 s=2.8 秒、观测到提交 D=5.8 秒。**
这里 D 是计算及交接延迟，不是重规划间隔 s，也不是首个可修改目标的时间。

| s | 固定 D | 实测结果 |
|---|---|---|
| 2.4 秒 | 6.4 秒 | 错过第 2 次交接 |
| 2.5 秒 | 6.2 秒 | 错过第 3 次交接 |
| 2.6 秒 | 5.4 / 6.0 秒 | 两组均曾通过 3 次独立 20 秒短测；重新启动持续测试后分别在约 18.4 / 13.8 秒超时，不能作为可靠下限 |
| 2.7 秒 | 5.4 / 5.8 秒 | 5.8 秒预算短测通过；持续测试分别在约 51.3 / 13.9 秒超时 |
| 2.8 秒 | 5.4 秒 | 短测通过，持续测试约 30.6 秒超时 |
| **2.8 秒** | **5.8 秒** | **同一参数通过连续 60 秒，以及额外 3 次独立 20 秒回放** |

最初短测得到 2.6 秒，随后持续测试将其排除。结果目录中的 `selection.json` 仅记录最初的短测筛选，
已经标明其范围；最终结论以 `four_npu_report.json` 为准，所有失败保留在 `all_trials.csv` 中。
本轮采用的经验筛选为：同一历史策略、s 和 D 至少通过三次独立 20 秒及一次连续 60 秒测试，
且该组合在本轮没有记录到失败。这是经验验证范围，不是最坏延迟保证。

最终组合的四次通过测试合计 3600 个控制 tick、34 次计划提交：

| 指标 | 结果 |
|---|---:|
| 连续 60 秒回放 | 1800 tick，19 次提交 |
| 最大控制线程迟到 | 14.214 ms，低于一个 33.333 ms 控制周期 |
| 最小有效插值覆盖 | 4.133 秒 |
| 提交间隔相对 2.8 秒的最大误差 | 0.137 ms |
| action 解码中位数 / 最大值 | 2.504 / 2.718 秒 |
| 历史准备中位数 / 最大值 | 1.942 / 2.159 秒 |
| video 阶段中位数 / 最大值 | 0.462 / 0.509 秒 |
| 观测到提交 | 5.800 秒 |
| 观测到首个可修改目标 | 6.0～6.5 秒；延迟 buffer 退出旧预热样本后为 6.3～6.5 秒 |

四个副本能并行准备历史与 video，但当前协议仍要求下一次 action 在上一计划提交后才绑定最新前缀，
action 解码与计划交接依次进行。新增副本没有把单次 action/VJP 的约 2.5 秒计算时间分摊到多卡。
2.6 秒失败的一次，视频已经在前一计划提交前准备好，action 绑定到终止时已等待约 2.635 秒仍未返回；
2.7 秒、D=5.8 秒的一次也在完整 action 时间窗口内超时。增大 D 可以缓解历史/video 准备迟到，
无法增加稳态相邻 action 交接之间的 s 时间窗口。

因此，保持当前 3+10、128 目标和最新前缀串行绑定协议，四卡没有测出比此前双卡短测 2.8 秒更低的可靠间隔。
本轮四卡采用的验证更长；此前双卡 2.8 秒、D=5.4 秒只有三次 20 秒记录，
不能把两轮不同时间、负载和验证长度的结果当成卡数变化的严格因果对照。
若继续缩短 s，重点需要放在单次 action/VJP 耗时，或允许动作计算并行且保持前缀一致的新协议。

原增量历史模式仍按完整的 32 个目标更新，其最小时间格是 3.2 秒，不随副本数增加而改变。
本轮四卡增量模式 D=4.4 秒出现超时，增加到 D=4.8 秒后通过 600 tick、4 次提交；
这项记录也表明不能只凭副本更多就沿用较紧的时延预算。

实验入口现在接受 2～4 个独立设备，并记录世界预测就绪、action 绑定时刻和失败时的在途任务状态。
35 项 CPU 检查通过，包含四卡对较长准备阶段的覆盖，以及四卡无法掩盖串行 action 超时的边界。
独立于 `PlanLedger` 重建全部 21 次测试的 14015 条已发出命令，冻结前缀、插值锚点和计划版本检查通过；
最终组合的 3600 条命令和 34 次提交也全部逐元素一致。

20 秒复现命令：

```bash
cd /mnt/sfs_turbo/guzhai/code/Zero-WAM-lijianwen
source /usr/local/Ascend/ascend-toolkit/set_env.sh
MODEL_PATH=/mnt/sfs_turbo/guzhai/tmp/zerowam_posttrain_eval_20261019/model_step5000 \
bash script/probe_piperpika_rtc_pipeline.sh \
  --devices 0 3 4 6 --video-steps 3 --action-steps 10 \
  --history-mode snapshot --snapshot-history-intervals 2 \
  --receding-actions 28 --handoff-latency 5.8 --shared-recording \
  --ticks 600 --output /mnt/sfs_turbo/guzhai/tmp/piperpika_four_npu_s28_replay
```

结果目录为 `/mnt/sfs_turbo/guzhai/tmp/zerowam_rtc_four_npu_20261020/`：
`four_npu_report.json` 为最终筛选及汇总，`all_trials.csv` 保存所有通过和失败，
`artifact_audit.json` 保存独立重建结果，`followup/` 保存最终组合的原始动作和时序。
`search.py`、`stress.py`、`followup.py` 分别保存初始搜索、持续测试和追加预算实验，
`source_initial/` 与 `source_stress/` 保存相应调度器版本。

连续 60 秒测试将一段 20 秒录制 RGB 周期性输入，控制时间戳、模型输出动作历史、延迟 buffer 和调度状态连续推进，
中途没有重置。这是实际墙钟的负载与时序测试，输入不是一条真实 60 秒 episode，
也不会根据模型动作改变场景；没有据此推断机器人任务成功率。四张实验卡已释放，其余已有任务未停止。

## 多副本实验的 device_count 参数

`script/probe_piperpika_rtc_pipeline.sh` / `evaluation.piperpika.rtc_pipeline_probe`
现在支持 `--device-count`，也接受 `--device_count` 写法，**默认 2**。
它表示完整模型副本数：每张 NPU 独立加载模型、VAE 和历史缓存，由 CPU 协调器统一管理计划及动作前缀。
回放支持 2、3、4 张卡；`--profile-only` 额外允许 1 张。现有单卡生产 `launch_server.sh`
仍使用单卡请求协议，这个参数用于多副本回放实验入口。

| 参数写法 | 行为 |
|---|---|
| 不传数量和卡号 | 自动选择 2 张可见、健康、无进程且空闲显存至少 48 GiB 的 NPU |
| `--device_count 4` | 自动选择 4 张满足上述条件的 NPU |
| `--devices 0 3` | 按指定顺序使用设备 0、3，自动推导 `device_count=2` |
| `--devices 0 3 4 6` | 兼容旧命令，自动推导 `device_count=4` |
| `--device_count 4 --devices 0 3 4 6` | 数量与列表必须一致；不一致立即报错 |

启动前通过 `npu-smi info -m` 和 `npu-smi info` 只读查询设备映射、健康、HBM 和进程。
未设置可见性限制时，自动选择优先使用空闲显存最多的卡，相同则按设备编号排序；
设置 `ASCEND_RT_VISIBLE_DEVICES=0,3,4,6` 时，只在这份列表中选择，保留列表顺序。
设备编号是 Ascend 运行时编号，对应 `npu-smi info -m` 的 `Chip Logic ID`；当前单芯片 910B3 机器上
它与卡号相同。不会把非连续的设备列表改成 `0,1,...`，也不会突破可见性限制。

自动选择跳过已有进程的卡。手动 `--devices` 表示明确指定设备，因此允许同卡已有进程，
但仍须健康且至少有 48 GiB 空闲显存。数量不足、卡号不存在、状态无法识别或 `npu-smi` 查询失败
都会在读取 RGB、加载模型前报错，不会静默减少副本。查询不构成设备预留，子进程加载前还会再检查一次显存。

下面沿用此前双卡短测的 `s=2.8s, D=5.4s` 配置，省略设备参数即默认双卡：

```bash
cd /mnt/sfs_turbo/guzhai/code/Zero-WAM-lijianwen
source /usr/local/Ascend/ascend-toolkit/set_env.sh
MODEL_PATH=/mnt/sfs_turbo/guzhai/tmp/zerowam_posttrain_eval_20261019/model_step5000 \
bash script/probe_piperpika_rtc_pipeline.sh \
  --video-steps 3 --action-steps 10 \
  --history-mode snapshot --snapshot-history-intervals 2 \
  --receding-actions 28 --handoff-latency 5.4 --shared-recording \
  --ticks 600 --output /mnt/sfs_turbo/guzhai/tmp/piperpika_default_two_npu
```

要跑此前四卡的 `s=2.8s, D=5.8s` 组合，使用上节完整命令，把
`--devices 0 3 4 6` 替换为 `--device_count 4` 即可。
若只允许特定卡，命令前设置 `ASCEND_RT_VISIBLE_DEVICES=0,3,4,6`；需要固定每个副本的卡号和顺序时，
继续使用 `--devices`。

设备数量仅改变副本容量与轮转，不会自动更改 `s`、3+10 去噪、F、动作目标数、
snapshot 历史策略或 D。每次启动日志及 `report.json` 会记录 `device_count`、最终 `devices`
和选择依据 `device_selection`；原有交接超时、冻结前缀、版本和动作队列检查继续生效。
新增参数经过 CPU 回归和本机只读设备查询验证；没有为本次参数修改重新运行完整模型时序测试。
上述双卡配置仍只有此前的短测验证范围，四卡持续测试结果及其局限见上一节。
