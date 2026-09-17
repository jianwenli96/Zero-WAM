# 从原始 Wan 权重初始化 Zero-WAM

在仓库根目录使用项目环境执行：

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 OMP_NUM_THREADS=8 \
.venv/bin/python -m wan_va.wan_init \
  --source /mnt/sfs_turbo/public/ckpts/Wan-AI/Wan2.2-TI2V-5B \
  --output checkpoints/zero-wam-wan-init-fp32-seed42 \
  --seed 42 --dtype float32
```

输出包含 `transformer/config.json`、分片 safetensors 权重和 `initialization.json`。后者记录源权重路径、随机种子、每个 state-dict 条目的映射关系，以及新增层的初始化规则。

上述命令显式指定 `--dtype float32`，保留源权重的 FP32 精度；转换器未指定该参数时仍默认使用 BF16。分片大小默认上限为 3 GB，可通过 `--shard-size-gb` 调整。脚本拒绝覆盖已有输出，先写入同级临时目录，完成序列化与结构检查后再生成最终目录。

## 权重映射与初始化

转换输入是原始 Wan TI2V 权重。脚本调用 Diffusers 的 Wan 转换函数进行参数重命名，将 Conv3d patch 卷积核按通道、时间、高度、宽度的顺序展平，填入实际前向使用的 patch MLP，然后将视频 Transformer 权重复制到动作分支。

文本交叉注意力的 K/V 按 ICL 模型要求保持共享。四个 IFP 模块各含一个 Transformer block，均继承主干最后一个 block 的权重，包括已初始化的动作分支。源参数缺失、形状不兼容或存在未使用的源参数时，转换会报错。

新增动作输入输出层采用 PyTorch `Linear` 默认的均匀分布初始化，包括 bias。新增 IFP 特征融合层和投影层参考仓库原有模型实现，权重使用标准差为 0.02 的正态分布，bias 置零。**这些新增层的初始化规则是明确记录的实现选择，论文未完整说明其具体分布。** 初始化使用独立的 CPU 随机数生成器，不受模型构造过程消耗随机数的影响。

## 接入训练

本次只转换 Transformer。现有训练入口读取预计算的视频 latent 和文本特征，可直接使用输出根目录：

```bash
export MODEL_PATH="$PWD/checkpoints/zero-wam-wan-init-fp32-seed42"
```

设置环境变量不会启动训练。若要处理原始视频或执行推理，仍需分别准备转换后的 VAE、文本编码器和 tokenizer 组件目录；此初始化过程不使用已训练的 Zero-WAM Transformer 权重。

训练配置仍沿用公开代码：human-video RoPE 偏移为 24，IFP 损失权重为 `0.5/0.25/0.15/0.1`；论文分别写为 32 和 `0.5/0.25/0.15/0.15`。权重转换不会自动修改这两处差异。

## 对齐前的验证方法与结果

运行相关测试：

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 OMP_NUM_THREADS=2 \
.venv/bin/python -m pytest -q tests/test_wan_init.py tests/test_icl_model.py -k "not cuda and not npu"
```

安装了昇腾环境时，torch_npu 可能在 CPU 梯度检查点重计算期间查询设备能力。若 CPU 反向测试因沙箱无法访问驱动而失败，需要加载 CANN 环境、指定已分配的可见设备，并在允许访问设备的环境中运行。本机验证通过的命令为：

```bash
source setup_npu_env.sh
ASCEND_RT_VISIBLE_DEVICES=6 OMP_NUM_THREADS=2 \
.venv/bin/python -m pytest -q \
  tests/test_wan_init.py tests/test_icl_model.py tests/test_model_paths.py
```

13 项测试全部通过，包括转换并重新加载小模型后的视频、动作和 IFP 联合前向/反向。

本地生成过两份随机种子为 42 的初始化权重：

| 精度 | 分片数 | 张量总字节数 | 验证结果 |
| --- | ---: | ---: | --- |
| BF16 | 8 | 22,862,256,572 | 1,864 个映射/复制条目、16 个新增条目；复制结果与源权重转换为 BF16 后一致 |
| FP32（推荐） | 16 | 45,724,513,144 | 直接从原始 Wan 生成；1,864 个复制条目回读后与 FP32 源值完全一致，16 个新增条目均为有限数值 |

条目数包含共享参数的别名，不代表独立参数数量。FP32 版本的校验结果保存在输出目录的 `validation.json`。转换期间未运行完整模型训练。
