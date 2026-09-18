# 从原始 Wan 权重初始化 Zero-WAM

按 [训练说明](cluster-training.md)设置 `PYTHON_BIN`、`WAN_SOURCE` 和 `MODEL_PATH`，在仓库根目录执行一次转换：

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 OMP_NUM_THREADS=8 \
  "$PYTHON_BIN" -m wan_va.wan_init \
  --source "$WAN_SOURCE" --output "$MODEL_PATH" --seed 42 --dtype float32
```

输入为原始 Wan2.2-TI2V-5B 权重；输出包含 `transformer/config.json`、分片 safetensors 和 `initialization.json`。记录包括源路径、种子、参数映射和新增层初始化规则。转换拒绝覆盖已有目录，先写临时目录，完成序列化与结构检查后生成目标目录。

显式指定 `--dtype float32` 保留源权重的 FP32 精度；转换器省略该选项时默认 BF16。分片上限默认 3 GB，可用 `--shard-size-gb` 调整。FP32 初始化权重与训练的 BF16 计算精度是不同设置。

## 参数映射

Diffusers 的 Wan 转换函数完成参数重命名；Conv3d patch 核按通道、时间、高度、宽度展平，填入前向实际使用的 patch MLP。视频 Transformer 权重复制到动作分支，文本交叉注意力的 K/V 保持共享。四个 IFP 模块各一个 block，继承主干最后一个 block，包括动作分支。源参数缺失、形状不兼容或存在未使用的源参数时直接报错。

动作输入/输出层使用 PyTorch Linear 默认均匀初始化；IFP 特征融合与投影层权重使用标准差 0.02 的正态分布、bias 为零。新增参数使用独立 CPU 随机数生成器，避免受模型构造消耗随机数的影响。这些分布是当前实现选择，论文未完整规定。

## 训练接入

本流程只转换 Transformer。预计算 latent 与文字特征训练可直接将输出目录作为 `MODEL_PATH`，不需要在线加载 VAE 或文本编码器。原始视频编码和推理仍需另行准备 VAE、文本编码器及 tokenizer。

human-video RoPE 偏移为 24，IFP 损失权重为 `0.5/0.25/0.15/0.1`，沿用公开训练代码；论文分别写为 32 和 `0.5/0.25/0.15/0.15`。权重转换不会修改这两处配置。初始化及映射回归覆盖在 `tests/test_wan_init.py`，CPU 测试命令见训练说明。
