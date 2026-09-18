# 昇腾训练实现

当前 `lianjie-dev` 交付分支保留与 mentor `main_ascend`（`d9a2177`）对齐的设备迁移方式，并加入已实现的显存、数据与训练功能。Git 历史中的 NPU 适配提交为 `878e40c`；这里的“对齐”描述实现来源，不表示当前核心文件与 mentor 分支逐字相同。集群启动统一见 [八卡交接说明](cluster-training.md)。

## 当前实现

- 训练、推理入口使用 `torch_npu.contrib.transfer_to_npu`，由迁移层转换 CUDA 设备、同步和分布式调用；没有另一套 `runtime.py` 设备管理。
- NPU 使用稠密 bool mask + SDPA，CUDA 使用 FlexAttention。大 mask 按行分块构建以限制临时整数矩阵，最终二维 bool mask 仍保留；MCP 同一步复用不可变 mask。
- 全屏蔽 padding 行在 NPU 上显式清零，保持参考输出与梯度语义；RoPE 保留复数路径，在 NPU 使用 FP32。
- FSDP2 默认 sublayer，BF16 计算、FP32 reduce；主干与 MCP 激活重算、`AdamW(fused=True)` 均启用。
- 八卡 HumanGen 混训入口默认加载实测容量配置，CPU 上随机同步裁剪机器人视频、动作和掩码，保留完整人类示范与文字（条件 dropout 独立执行）。
- 六源权重采样、长度分桶、索引/Arrow 缓存、manifest 复用、Wan 原始权重转换、真实动作转换及 RoboTwin 43/7 任务隔离均保留。

`ZERO_WAM_SDPA_CHUNK_SIZE`、`ZERO_WAM_DEVICE` 不控制当前训练路径。`MAX_TRAIN_FRAMES` 是可选额外帧数上限，不设置时八卡仍按容量配置裁剪。

## 环境与检查

依赖见 `requirements-npu.txt` 与 `pyproject.toml`；目标节点使用兼容的驱动/CANN/torch_npu。已有相同挂载路径的 aarch64 环境可参考 [共享环境说明](shared-environment.md)。其他节点设置 `PYTHON_BIN`、`CANN_ENV_PATH`、模型及数据路径，不能直接依赖本机路径。

```bash
# 已完成数据准备和模型初始化后，只展示启动命令。
bash script/train_humangen_wan_npu.sh --dry-run
```

实际启动需显式分配 `ASCEND_RT_VISIBLE_DEVICES` 并传 `--run`。默认试跑十步，八卡默认容量裁剪、混合来源分桶窗口 10、初始化 worker 1、每卡加载 worker 2。

CPU 回归可隔离迁移层的全局 CUDA 替换，排除硬件测试参数：

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 OMP_NUM_THREADS=2 \
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 .venv/bin/python - <<'PY'
import sys
sys.modules['torch_npu.contrib.transfer_to_npu'] = None
import pytest
raise SystemExit(pytest.main(['-q', 'tests', '-k', 'not cuda and not npu']))
PY
```

CPU 回归不代替迁移层或 NPU 算子的硬件验证。当前八卡真实子集验证结果、未完成范围和容量边界集中记录在 [训练显存说明](training-memory.md)。此前更早的独立适配/其他卡数结果仅作为历史记录。
