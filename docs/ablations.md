# Zero-WAM 消融训练

所有实验继承 `script/train_dist.sh` 的模型初始化、优化器、训练步数及分布式配置。
在集群每个节点执行同一条实验命令；节点变量沿用 `NNODES`、`NODE_RANK`、
`MASTER_ADDR`、`MASTER_PORT`、`NGPU`（或原脚本支持的集群环境变量）。

| 实验 | 一键命令 | 控制 flag |
| --- | --- | --- |
| 全功能基线 | `bash script/train_dist.sh` | 无 |
| 仅 text prompt | `bash script/train_dist_text_only.sh` | `--disable-human-video` |
| 仅 human video prompt | `bash script/train_dist_human_video_only.sh` | `--disable-text` |
| 移除 MCP | `bash script/train_dist_no_mcp.sh` | `--disable-mcp` |
| 仅 HumanGen Robotwin 数据 | `bash script/train_dist_robotwin_only.sh` | `--robotwin-only` |

四个 flag 可独立组合，也可直接传给 `python -m wan_va.train`。
脚本支持追加原训练参数，例如 `--num-steps 20000 --save-interval 1000`。
单机 8 卡示例：

```bash
NNODES=1 NODE_RANK=0 MASTER_ADDR=127.0.0.1 NGPU=8 \
  bash script/train_dist_text_only.sh
```

默认输出分别位于 `outputs/ablation_{text_only,human_video_only,no_mcp,robotwin_only}`，
每次运行仍创建时间戳子目录，可通过 `ZERO_WAM_SAVE_ROOT` 覆盖。
运行目录的 `ablations.json` 记录实际条件开关、数据源及 dropout 设置。

## 实验语义

- text-only：不向 transformer 输入 human ICL latent 或 human 文本；机器人观测、
  action/video 训练目标与 MCP 保留。数据仍使用同一份配对索引及 human 缓存，
  保持训练样本集合和长度分桶不变，因此仍需要完整的 HumanGen 缓存。
- human-video-only：机器人文本与 human 缓存文本都替换为预训练的空文本 embedding，
  保留 cross-attention 结构，但不输入任务文本信息。`--droptext-target` 无法覆盖该消融。
- no-MCP：关闭 MCP 模块加载、MCP 噪声调度器和辅助损失；主 video/action 分支保留。
- robotwin-only：强制数据源为 `robotwin:1.0`，使用 HumanGen 下的 `robotwin_data`、
  `ICL_config_robotwin.json` 和 `human_latents/robotwin`。保留已有 43 个训练任务与
  7 个留出任务划分。自动关闭仅支持混合数据的长度分桶。

剩余条件沿用基线 dropout（text 0.4、human ICL 0.1），以便每组仅消融指定因素；
“仅某条件”指可用的 prompt 模态，并非每步都强制保留该条件。
前 3 组默认使用基线的六类数据及其原权重；第 4 组覆盖 `DATASETS` / `--datasets`。
比较时保持相同初始化、步数、全局 batch size；Robotwin-only 在相同步数下会更频繁重复
Robotwin 样本，这是固定训练预算下的数据组成对照。

## 对应推理配置

评估时在传给 `VA_Server` 的 Robotwin job config 中设置相应字段：

```python
job_config.enable_human_video = False  # text-only
job_config.enable_text = False         # human-video-only
job_config.enable_mcp = False          # no-MCP
```

每组只设置对应字段；未设置的字段默认启用。Robotwin-only 不需要额外推理开关。
训练开关不会自动修改推理配置。human-video-only 的 prompt 可以为空，且缓存中的
human 文本也会被替换；text-only 即便请求 `use_icl=True` 也不会使用 human 条件。


## 多卡推理启动脚本

在仓库根目录执行下列命令，`MODEL_PATH` 指向对应实验的模型根目录
（含 `transformer/`，并满足现有服务的 VAE、文本编码器等组件加载要求）：

```bash
MODEL_PATH=/path/to/text_only_model bash evaluation/robotwin/launch_server_multigpus_text_only.sh
MODEL_PATH=/path/to/human_video_only_model bash evaluation/robotwin/launch_server_multigpus_human_video_only.sh
MODEL_PATH=/path/to/no_mcp_model bash evaluation/robotwin/launch_server_multigpus_no_mcp.sh
MODEL_PATH=/path/to/robotwin_only_model bash evaluation/robotwin/launch_server_multigpus_robotwin_only.sh
```

脚本自动传入对应的 `--disable-human-video`、`--disable-text`、`--disable-mcp`，
无需手改 Python 配置。Robotwin-only 保留完整推理条件，区别是加载的训练权重。
原 `launch_server_multigpus.sh` 继续用于全功能基线，也支持追加以上命令行开关。

各脚本沿用原来的 GPU 0–7，每卡启动一个服务，默认服务端口 29556–29563，
分布式端口 29661–29668。各实验应分别运行；日志目录区分并不意味着 GPU 或端口隔离。
可通过 `START_PORT`、`MASTER_PORT` 覆盖端口，并与客户端设置保持一致。
客户端及 human video 请求方式沿用现有评估流程，由服务端开关决定是否使用条件。

默认日志和 PID 文件位于 `logs/ablation_<实验名>/`，可视化结果位于
`visualization/ablation_<实验名>/`，编译缓存也按实验分开。
可以用 `LOG_ROOT`、`SAVE_ROOT`、`TORCHINDUCTOR_CACHE_ROOT` 覆盖。
