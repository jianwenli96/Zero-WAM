# 昇腾实现：以 mentor 分支为基线

当前开发分支 `lianjie/mentor-aligned` 基于 `mentor/main_ascend` 的
`d9a2177`（Adapt to NPU for core models）。原有独立 NPU 适配保存在
`backup/pre-mentor-alignment` 和 `lianjie/ascend-training`。

## 核心实现

- 训练、推理入口沿用 mentor 的 `torch_npu.contrib.transfer_to_npu`，由迁移层
  转换 CUDA 设备、同步和分布式调用；不再使用本地 `runtime.py`。
- ICL 注意力沿用完整稠密布尔 mask + SDPA；CUDA 分支使用 FlexAttention。
  不再包含 query 分块实现，`ZERO_WAM_SDPA_CHUNK_SIZE` 不生效。
- RoPE 沿用 mentor 的复数路径及 NPU FP32 调整。
- FSDP2、激活重算、默认 mesh 和 `AdamW(fused=True)` 沿用 mentor。
- `wan_va/modules/model.py`、`icl_model.py`、`wan_va/distributed/`、服务端以及
  mentor 的 `tests/test_icl_model.py` 保持基线实现。

## 保留的本地功能

Wan 原始权重转换、HumanGen/RoboTwin 数据准备、43/7 任务隔离、数据读取修复、
六源采样策略、训练命令归档、随机种子、逐步 loss/耗时记录和可选机器人片段裁剪。
裁剪同时作用于视频、动作及掩码，保留完整人类示范，默认不开启。

旧版 NPU 专用 smoke/self-rollout 脚本未迁入此分支，避免混用独立设备管理逻辑；
可在备份分支查看，其历史输出仍在本地 `outputs/`。mentor 的原推理服务保留。

## 环境和启动

复用本机 `.venv`，无需重装。新环境使用 `requirements-npu.txt`，包元数据将
`flash-attn` 放在 CUDA extra 中；这仅调整安装依赖，不改变 mentor 的运行实现。
LeRobot 依赖与共享 Python 说明见对应数据文档和 `shared-environment.md`。

```bash
source script/activate_shared_env.sh
source setup_npu_env.sh
bash script/train_humangen_wan_npu.sh --dry-run
```

实际启动仍需显式指定已分配的 `ASCEND_RT_VISIBLE_DEVICES` 并传 `--run`。
`MAX_TRAIN_FRAMES=16` 可为混训入口开启训练片段上限，未设置时保持完整序列。
`ZERO_WAM_DEVICE` 不再控制训练设备。

## 验证范围

对齐前独立适配的 NPU 测试、离线推理及六卡训练记录，不证明本分支已通过同样验证。
此前 165 个机器人 latent 帧样本在 MCP 调制处 OOM；mentor 的对应计算保持相同，
并且稠密注意力 mask 的内存随 token 数平方增长，因此不能把本次对齐视为 OOM 修复。

本次只进行 CPU 回归、启动命令与核心文件一致性检查，不启动 NPU 训练。
CPU 回归需要隔离 `transfer_to_npu` 的全局 CUDA 替换，并只选择 CPU 测试参数；
不能把这种隔离测试视为迁移层或 NPU 算子的硬件验证。

可复现的 CPU 回归命令：

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 OMP_NUM_THREADS=2 \
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 .venv/bin/python - <<'PY'
import sys
sys.modules['torch_npu.contrib.transfer_to_npu'] = None
import pytest
raise SystemExit(pytest.main(['-q', 'tests', '-k', 'not cuda and not npu']))
PY
```

本次结果（2026-09-17）：94 项通过、18 项 CUDA/NPU 参数测试未运行。
两个训练入口的 dry-run、可选裁剪参数和 shell 语法检查通过；核心模型、分布式、
服务端、公共工具及 ICL 模型测试文件与 `mentor/main_ascend` 无差异。
