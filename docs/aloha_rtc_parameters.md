# Zero WAM ALOHA RTC 参数与训练说明

更新日期：2026年10月18日。代码目录：`/mnt/sfs_turbo/guzhai/code/Zero-WAM-lijianwen`。

当前保留10 Hz动作目标、最多4个未来视频latent、video 3步和action 10步去噪的已测组合为：`action_stride=3`、`actions_per_latent=32`、`F=4`、`s=64`。一次预测128个未来动作目标，覆盖12.8秒，每推进6.4秒启动下一轮规划。2026年10月16日的录制图像回放通过RTC时间检查，在线平均4.429秒、最大4.725秒，上限5.208秒。该结果使用旧step-6000权重测量计算时间；新时间布局的任务质量需要匹配后训练与独立评测。

## 参数含义和取值

| 参数 | 当前取值 | 在哪里设置 | 含义及影响 |
|---|---:|---|---|
| 原始控制频率 | 30 Hz | 推理 `--rtc-control-hz 30` | 数据采集和底层控制频率 |
| action_stride | 3 | 训练 `--action-stride 3` | 相邻模型动作目标间隔3个原始控制步；ALOHA模型目标频率为30/3=10 Hz |
| A | 32 | 训练 `--actions-per-latent 32` | 每个保留视频latent对应32个动作目标，覆盖3.2秒 |
| F | 4 | 推理 `--chunk-frames 4` | 每次预测4个未来视频latent；初始观测及PAD单独缓存 |
| H | 128 | 由F×A计算 | 每次联合去噪128个未来动作目标，覆盖12.8秒 |
| s | 64 | 推理 `--rtc-execution-horizon 64` | receding量；每推进64个模型目标启动下一轮规划，名义间隔6.4秒 |
| video去噪步数 | 3 | 推理 `--steps 3` | 视频分支的flow迭代次数；也作为动作步数的默认值 |
| action去噪步数 | 10 | 推理 `--action-steps 10` | 单独覆盖动作分支的flow迭代次数 |
| RTC开关 | True | 推理 `--enable-rtc` | 默认False；开启训练免修改的soft-mask/VJP指导、异步队列和时间检查 |
| 历史窗口 | 4 | 推理 `--rtc-attn-window 4` | 保留的模型视频历史窗口；当前低延迟实测使用此值 |
| ICL CFG | 1 | 推理 `--rtc-icl-guidance-scale 1` | 当前实测的ICL指导尺度 |
| 延迟buffer长度 | 10 | 推理 `--rtc-delay-buffer-size 10` | 用最近最多10次请求延迟的最大值预测下一次延迟 |
| 延迟乘数 | 1.2 | 推理 `--rtc-latency-multiplier 1.2` | 在实测延迟上增加20%预留 |
| 额外时间 | 0.05秒 | 推理 `--rtc-latency-margin 0.05` | 在乘数预留后额外增加50 ms |
| 初始延迟估计 | 2秒 | 推理 `--rtc-initial-latency 2` | 这类时间布局的自动默认值；必须由客户端预热实测校验 |
| RTC指导权重上限 | 5 | 推理 `--rtc-max-guidance-weight 5` | 裁剪training-free action指导的权重 |
| 机器人图像尺寸 | 224×288 | 当前ALOHA配置，RTC客户端自动匹配 | 每个视角的高×宽；三个视角分别编码 |
| 推理卡数 | 1 | `NGPU=1` | RTC路径使用单张完整模型，不使用推理FSDP |

3+10表示两条分支各自的去噪迭代次数。每次动作迭代处理整个F×A动作张量；32和128都不是去噪次数。更多去噪步数的动作精度收益尚需单独评估。

## 动作与视频的时间对应

ALOHA原始数据为30 Hz。现有机器人RGB每4个原始控制步取一组，原生VAE输入为7.5 fps；VAE在首帧之后每4张RGB生成一个原生latent，对应16个原始动作间隔。

`action_stride=3`和`A=32`组合的原生latent保留跨度为：

```text
latent_stride = A × action_stride / native_A
              = 32 × 3 / 16 = 6

保留视频：z0, z6, z12, z18, ...
第一个动作块：a0, a3, ..., a93        共32个目标
第二个动作块：a96, a99, ..., a189    共32个目标
```

VAE网络、权重、原生编码频率和latent维度保持原样。训练从现有VAE文件选取上述latent；在线先按原生频率编码反馈，再做同相位选择。这里3.2秒是动作块与保留视频端点的时间对齐，不表示VAE重新把整段3.2秒压缩进一个latent。

底层仍以30 Hz输出控制指令：位置和关节线性插值，单位四元数SLERP，夹爪保持左端目标。完整episode的物理时长保持原样；当前训练集44条episode约60.8～92.5秒，平均78.2秒。

## RTC时间预算

设L为延迟buffer中的最大请求延迟，d为预留的模型动作步数：

```text
f = 30 / action_stride = 10 Hz
H = F × A = 128
d = ceil(f × (1.2 × L + 0.05)) + 1
条件：d ≤ s ≤ H - d

最大原始延迟 = ((min(s, H-s) - 1) / f - 0.05) / 1.2
             = 5.208333秒    （s=64）
```

额外的1步冻结插值端点，避免换队列修改已经开始执行的插值段。客户端在机器人静止时至少完成3轮guided预热，并将最后两轮预热测量纳入初始buffer。启动或运行时超预算会报错。

目前实现要求H和s都是A的整数倍。s=32时每3.2秒重新规划，但允许原始延迟只有2.542秒，当前3+10实测无法满足。s=64通过的是动作队列与时间检查；6.4秒的观测重规划间隔是否适合接触任务，需要评估任务效果。

## 已通过时间检查的候选组合

下表均为F=4，使用单卡、历史窗口4、ICL CFG=1、224×288输入和真实soft-mask/VJP。不同实验的设备负载不同，延迟仅用于核对各自预算。

| video + action | A | action_stride | 模型目标频率 | H / 预测时间 | s / 重规划间隔 | 在线最大延迟 | 验证范围 |
|---|---:|---:|---:|---|---|---:|---|
| 3+10 | 32 | 3 | 10 Hz | 128 / 12.8秒 | 64 / 6.4秒 | 4.725秒 | 85.3秒回放，12次换队列 |
| 2+2 | 16 | 3 | 10 Hz | 64 / 6.4秒 | 32 / 3.2秒 | 2.095秒 | 42.7秒回放，12次换队列 |
| 1+1 | 16 | 3 | 10 Hz | 64 / 6.4秒 | 32 / 3.2秒 | 1.427秒 | 21.3秒回放，6次换队列 |
| 3+3 | 16 | 3 | 10 Hz | 64 / 6.4秒 | 32 / 3.2秒 | 2.400秒 | 12.8秒短回放，3次换队列 |
| 4+2 | 16 | 3 | 10 Hz | 64 / 6.4秒 | 32 / 3.2秒 | 2.095秒 | 12.8秒短回放，3次换队列 |
| 3+10 | 16 | 6 | 5 Hz | 64 / 12.8秒 | 32 / 6.4秒 | 4.936秒 | 85.3秒回放，12次换队列 |

延迟统计从发起重规划到控制循环获取响应，包含反馈打包、往返、VAE编码、历史提交和去噪；相机采集、回放预解码、此前逐帧缩放、模型加载与文本reset不计入这些数值。录制观测与模型输出无关，回放验证软件与时间可行性，不是实机操控成功率测试。

## 四卡训练命令

以下从现有ALOHA step-6000权重开始新布局后训练。只选择ALOHA，避免把不同来源的原始频率混为统一10 Hz。原始数据、latent和旧checkpoint不覆盖。

```bash
cd /mnt/sfs_turbo/guzhai/code/Zero-WAM-lijianwen
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4
export ASCEND_RT_VISIBLE_DEVICES=1,5,6,7
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ALOHA_PREPARED_ROOT=/mnt/sfs_turbo/lijianwen/Datasets/aloha_zerowam_v3
export MODEL_PATH=/mnt/sfs_turbo/lijianwen/Codes/Zero-WAM/outputs/aloha_step6000_validation/model
export ZERO_WAM_SAVE_ROOT="$PWD/outputs/aloha_rtc_a32_stride3"

NNODES=1 NODE_RANK=0 MASTER_ADDR=127.0.0.1 MASTER_PORT=29648 NGPU=4 \
CONFIG_NAME=aloha_train DATASETS=aloha:1 \
bash script/train_dist.sh \
  --action-stride 3 --actions-per-latent 32 \
  --length-bucket-steps 0 --learning-rate 1e-5 \
  --batch-size 1 --gradient-accumulation-steps 1 \
  --num-steps 5000 --save-interval 2500 \
  --init-worker 2 --load-worker 2 --disable-wandb
```

NPU编号是2026年10月18日本次检查到的空闲卡；以后启动前用`npu-smi info`核对，训练使用本机HCCL通信。命令沿用本次短训练的累积1，全局batch=4×1×1=4。此前八卡、每卡batch1且累积1的全局batch为8；若要改成累积2，应另测FSDP梯度累积的显存开销。每卡batch必须为1。5000步是首轮预算，按验证集挑选checkpoint。

本次每个checkpoint约22 GB，短训练保存后工作盘约剩106 GiB。正式示例每2500步保存一次，5000步共两份；若改回每500步保存，将产生约220 GB权重，需要更大的输出空间或及时归档旧checkpoint。当前训练器不会自动删除旧checkpoint。

| 训练设置 | 作用 |
|---|---|
| `DATASETS=aloha:1` | 只用ALOHA；默认脚本否则会混入多个HumanGen来源 |
| `--length-bucket-steps 0` | 单数据源关闭长度分桶，避免入口校验报错 |
| `--learning-rate 1e-5` | 后训练目标学习率；当前默认warmup为200步 |
| `--gradient-accumulation-steps` | 每卡累积多少个microbatch再更新一次；训练num-steps按更新轮计数 |
| `--num-steps` | 本次新增训练轮数；从旧权重初始化不等于恢复优化器和全局步数 |
| `--save-interval` | 每多少轮保存一次；最后一轮也会保存 |
| `--init-worker / --load-worker` | 数据集初始化和DataLoader并发度 |
| `--disable-wandb` | 关闭W&B；本地train.log仍保存 |

训练继续使用FSDP、activation checkpointing及全部video/action/ICL/MCP训练分支。训练chunk大小默认在1～4随机抽取，匹配推理F=4上限。3+10去噪步数、s和enable_rtc均属于推理设置；不需要在训练时加入RTC开关或改为3/10步训练schedule。

输出为`ZERO_WAM_SAVE_ROOT/<运行时间>/checkpoints/checkpoint_step_*/`。新checkpoint保存版本3时间契约，包含`action_stride=3`、`actions_per_latent=32`、`native_actions_per_latent=16`和`latent_stride=6`；推理自动读取并拒绝冲突覆盖。

## 在 ModelArts 创建训练作业

Notebook和ModelArts训练作业均可使用`script/train_modelarts_rtc.sh`。单机或多节点使用同一条命令，默认每个节点使用全部可见NPU，不固定为4卡或8卡。`train_dist.sh`通过共享的`script/distributed_env.sh`读取平台分配、可见设备和节点信息；ModelArts入口负责准备Ascend环境及当前RTC后训练参数。

### 控制台配置

| 字段 | 当前配置 |
|---|---|
| 训练模式 | 自定义作业；由下面的bash脚本自行启动torchrun |
| 镜像 | 能运行已验证Ascend环境的兼容镜像；本次本机验证使用PyTorch 2.6和torch_npu 2.6 |
| 实例数 | 按资源需要选择单机或多机，由平台变量自动读取 |
| 加速卡 | 每实例分配所需的Ascend 910B，单卡64 GB；默认使用该节点全部可见卡，各节点卡数需一致 |
| 主机内存 | 按每节点进程数为CPU加载模型及checkpoint汇总留足内存；每份完整FP32参数约45.7 GB（按权重元素数估算） |
| 存储 | 将下面三个路径所在的SFS Turbo挂载进训练容器，并为输出目录配置写权限 |
| 启动命令 | 先使用下方10步短训练命令 |
| 作业日志路径 | 选择可写的OBS日志目录，训练stdout在作业日志中查看 |

容器内需要同时能看到以下目录：

| 挂载路径 | 用途 |
|---|---|
| `/mnt/sfs_turbo/guzhai` | 当前代码和新的训练输出 |
| `/mnt/sfs_turbo/lijianwen` | ALOHA数据、step-6000初始化权重及其符号链接目标 |
| `/mnt/sfs_turbo/public` | 共享Python环境、底座VAE/T5/tokenizer及已有缓存 |

这些路径是训练容器里的路径，Notebook中存在不代表新作业容器已挂载。现有数据的编码契约、缓存和模型符号链接含绝对路径，保持原路径可直接复用。已有模型目录本身的符号链接不包含真实权重文件，必须让链接目标也可访问。

本机已测环境的Python为`/mnt/sfs_turbo/public/apps/miniforge3/envs/lingbot-vggt/bin/python`。共享环境需要镜像的系统库、CANN与torch_npu兼容；若在镜像中另装环境，用`PYTHON_BIN`指向其Python。当前`requirements.txt`含torch 2.9和flash_attn等GPU侧依赖，Ascend镜像依赖以已跑通环境及对应CANN兼容关系为准。

### 启动命令

在控制台“启动命令”填入以下内容，先验证作业容器的挂载、权限和环境：

```bash
bash /mnt/sfs_turbo/guzhai/code/Zero-WAM-lijianwen/script/train_modelarts_rtc.sh \
  --num-steps 10 --save-interval 10
```

通过后，新建正式训练版本，启动命令改为：

```bash
bash /mnt/sfs_turbo/guzhai/code/Zero-WAM-lijianwen/script/train_modelarts_rtc.sh
```

默认参数为5000步、每2500步保存、每卡batch1、累积1、ALOHA单来源、stride3、A32。脚本只在每个容器运行一次，内部`train_dist.sh`负责创建每张卡的训练进程；如控制台有外部自动多卡启动方式，选择自定义bash入口，保持单层torchrun。

脚本保留平台的`ASCEND_RT_VISIBLE_DEVICES`。未显式指定数值`NGPU`（或指定`NGPU=auto`）时：有可见卡掩码就调用`torch_npu.npu.device_count()`取得掩码后的实际数量；无掩码时优先使用平台有效的`MA_NUM_GPUS`，平台未提供正卡数则通过同一接口探测。探测不到设备会报错，不会猜成固定卡数。显式设置数值`NGPU`仍可覆盖自动发现。

例如，平台分配8卡但掩码只开放4卡时启动4个进程；平台每实例分配8卡、实例数2时启动2×8=16个训练进程。每卡batch1、累积1时，全局batch为`节点数×每节点卡数`，会随资源规格变化。

### 多节点自动配置

| 参数 | 自动取值顺序 |
|---|---|
| 节点数 | 显式`NNODES` → `MA_NUM_HOSTS` → `VC_WORKER_NUM` → `VC_WORKER_HOSTS`中的主机数 → 单机1 |
| 节点rank | 显式`NODE_RANK` → `VC_TASK_INDEX`；仅单机可默认0 |
| 主节点地址 | 显式`MASTER_ADDR` → `VC_WORKER_HOSTS`第一个主机；仅单机可默认127.0.0.1 |
| 主节点端口 | 显式`MASTER_PORT` → `MA_TORCHRUN_MASTER_PORT` → ModelArts入口默认29648 |

ModelArts会在每个实例的容器执行同一启动命令，因此增加实例数无需修改启动命令。所有节点应使用相同版本代码、共享的模型/数据路径及同一个输出根目录。多节点缺少rank或主节点地址、rank越界、无可用NPU等情况会在创建训练进程前报错。

当前torchrun要求各节点的本地进程数一致：例如2个节点各8张可见卡可用，1个节点8卡、另1个节点4卡的混合规格不属于该启动方式的支持范围。框架约束见[torchrun文档](https://docs.pytorch.org/docs/stable/elastic/run.html)。

非ModelArts环境需要在每个节点提供节点信息，卡数仍自动发现。例如两个节点分别执行：

```bash
# 节点0：MASTER_ADDR替换成两个节点都能访问的节点0地址。
NNODES=2 NODE_RANK=0 MASTER_ADDR=10.0.0.1 \
bash /mnt/sfs_turbo/guzhai/code/Zero-WAM-lijianwen/script/train_modelarts_rtc.sh
```

```bash
# 节点1：使用相同的MASTER_ADDR和MASTER_PORT。
NNODES=2 NODE_RANK=1 MASTER_ADDR=10.0.0.1 \
bash /mnt/sfs_turbo/guzhai/code/Zero-WAM-lijianwen/script/train_modelarts_rtc.sh
```

这里“可见”由容器设备分配和可见卡掩码决定，不代表空闲卡扫描。此前四卡命令保留为本机已测的显式资源示例；通用入口不设置其物理卡号。

### 可选环境变量与输出

| 环境变量 | 默认值或用途 |
|---|---|
| `PYTHON_BIN` | 上述共享环境Python；镜像自带环境可覆盖 |
| `ALOHA_PREPARED_ROOT` | `/mnt/sfs_turbo/lijianwen/Datasets/aloha_zerowam_v3` |
| `MODEL_PATH` | `/mnt/sfs_turbo/lijianwen/Codes/Zero-WAM/outputs/aloha_step6000_validation/model` |
| `ZERO_WAM_SAVE_ROOT` | `/mnt/sfs_turbo/guzhai/modelarts_outputs/aloha_rtc_a32_stride3` |
| `ASCEND_ENV_FILE` | 默认`/usr/local/Ascend/ascend-toolkit/set_env.sh` |
| `MASTER_PORT` | 优先读取平台端口，否则默认29648，各节点一致 |
| `NGPU` | 默认自动发现全部可见NPU；可显式设置数值覆盖 |

输出默认放在代码目录之外的SFS路径，便于使用ModelArts容器用户写入。当前仓库的`outputs/`是root组的770权限，默认`ma-user`未必能写；新输出路径也须确认训练容器用户具备创建和写入权限。脚本会在加载模型前检查主要路径，并把HF/XDG缓存默认放到输出根目录下。

输出结构为`ZERO_WAM_SAVE_ROOT/<运行时间>/train.log`及`checkpoints/checkpoint_step_*/`。训练器直接写上述SFS目录，控制台填写OBS日志路径不会把这些checkpoint自动转换为OBS模型输出。

本入口仍按现有编码契约检查数据源；如果改挂载路径或复制数据，应先处理绝对路径和来源指纹，避免只改数据根目录后误用旧缓存。以下官方文档说明控制台字段、SFS挂载和平台变量；本机四卡短训练已验证，多节点已检查启动参数传递，尚未进行真实跨节点训练。ModelArts独立作业需通过首次10步检查确认其容器环境和跨节点通信。

- [创建自定义训练作业](https://support.huaweicloud.com/develop-modelarts/develop-modelarts-0006.html)
- [管理训练容器环境变量](https://support.huaweicloud.com/intl/zh-cn/develop-modelarts/develop-modelarts-0104.html)
- [PyTorch与NPU分布式训练示例](https://support.huaweicloud.com/intl/en-us/usermanual-standard-modelarts/modelarts-distributed-0012.html)

## 组合推理目录并启动服务

`RTC_CKPT`替换为选中的checkpoint绝对路径；`RTC_MODEL`使用尚未组合过的新目录。

```bash
cd /mnt/sfs_turbo/guzhai/code/Zero-WAM-lijianwen
RTC_CKPT="/实际路径/checkpoints/checkpoint_step_5000"
RTC_BASE=/mnt/sfs_turbo/public/ckpts/Zero-WAM/zero-wam-pretrain
RTC_MODEL="$PWD/outputs/aloha_rtc_a32_stride3/model"

mkdir -p "$RTC_MODEL"
ln -s "$RTC_CKPT/transformer" "$RTC_MODEL/transformer"
ln -s "$RTC_CKPT/aloha_action_contract.json" "$RTC_MODEL/aloha_action_contract.json"
for component in vae tokenizer text_encoder; do
  ln -s "$RTC_BASE/$component" "$RTC_MODEL/$component"
done

source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export ALOHA_PREPARED_ROOT=/mnt/sfs_turbo/lijianwen/Datasets/aloha_zerowam_v3
export MODEL_PATH="$RTC_MODEL"
export ASCEND_RT_VISIBLE_DEVICES=1
NGPU=1 MASTER_PORT=29649 PORT=29536 \
bash evaluation/aloha/launch_server.sh \
  --enable-rtc --steps 3 --action-steps 10 --chunk-frames 4 \
  --rtc-execution-horizon 64 --rtc-control-hz 30 \
  --rtc-attn-window 4 --rtc-icl-guidance-scale 1 \
  --rtc-delay-buffer-size 10 \
  --rtc-latency-multiplier 1.2 --rtc-latency-margin 0.05
```

启动服务后，在另一个终端运行录制数据RTC回放，重新核对部署设备上的延迟。此命令不连接机器人：

```bash
cd /mnt/sfs_turbo/guzhai/code/Zero-WAM-lijianwen
PY=/mnt/sfs_turbo/public/apps/miniforge3/envs/lingbot-vggt/bin/python
$PY -m evaluation.aloha.rtc_replay \
  --port 29536 --episode 1 --ticks 2560 --warmup-rounds 3 \
  --output outputs/aloha_rtc_a32_stride3/replay
```

32动作布局使用RTC客户端。普通open_loop入口当前只支持原生动作布局；不能把旧模型只改推理参数的时序探测当成新布局训练完成。

## 实测记录

- 32动作、10 Hz、3+10时间验证：`/mnt/sfs_turbo/guzhai/tmp/zerowam_a32_20261016/a32_report.md`。
- 16动作、10 Hz候选：`/mnt/sfs_turbo/guzhai/tmp/zerowam_time_stride_20261015/rtc_time_stride_report.md`。
- 保留3+10的5 Hz候选：`/mnt/sfs_turbo/guzhai/tmp/zerowam_rtc_hparams_20261016/rtc_hparams_report.md`。
- 本次四卡短训练命令及原始日志：`/mnt/sfs_turbo/guzhai/tmp/zerowam_rtc_train_20261018/`，最终结论见该目录的`training_validation.md`。

## 2026年10月18日四卡短训练结果

物理NPU 1、5、6、7使用上述时间布局完成10轮真实训练，四个rank正常结束。video/action/MCP loss及梯度均有限，优化器未跳过更新；保存的1880个权重张量全部可读且有限，代表性权重相对初始化模型发生变化，版本3时间契约读回通过。后续轮次平均间隔约18.67秒。短训练使用累积1、全局batch4，尚未验证收敛和任务成功率。

详细结果：`/mnt/sfs_turbo/guzhai/tmp/zerowam_rtc_train_20261018/training_validation.md`。短训练checkpoint：`/mnt/sfs_turbo/guzhai/code/Zero-WAM-lijianwen/outputs/aloha_rtc_a32_stride3_smoke_20261018/20261018_040052_512912/checkpoints/checkpoint_step_10`。
