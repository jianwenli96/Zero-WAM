# 跨主机复用开发环境

当前使用 Python venv，并非 Miniconda。基础 Python 3.12.13 已复制到共享目录
`/mnt/sfs_turbo/lianjie/runtimes/python3.12.13`，项目 `.venv` 指向这份解释器，保留原有依赖。
无需在另一台机器安装 Miniconda。

在另一台 Linux aarch64 主机上，将共享目录挂载到相同的绝对路径，然后执行：

```bash
cd /mnt/sfs_turbo/lianjie/Zero-WAM
source script/activate_shared_env.sh
python --version
python -m pip --version
```

使用 NPU 时，再加载该主机的 CANN：

```bash
source setup_npu_env.sh
```

默认 CANN 路径为 `/usr/local/Ascend/cann-9.1.0/set_env.sh`。如果安装位置不同，先设置
`CANN_ENV_PATH`。目标主机的系统库、驱动和 CANN 必须兼容当前 PyTorch / torch_npu。
共享目录不包含 NPU 驱动和 CANN；不适用于 x86 主机。

训练脚本仍直接使用 `.venv/bin/python`，无需预先激活环境。避免多台主机同时安装或升级同一个环境的依赖。

迁移只修改项目 `.venv/pyvenv.cfg` 和 `.venv/bin/python3` 的基础解释器路径，未移动或删除原系统 Python。
共享 Python 可执行文件的动态库搜索路径从系统绝对路径改为 `$ORIGIN/../lib`，使其加载共享目录内的 libpython。
迁移前配置、符号链接目标及依赖清单保存在 `outputs/shared-env-migration/`，便于核对和恢复。
Python 的编译配置仍可能含有原安装路径；今后在此环境源码编译扩展前应检查头文件和链接路径。

本机验证不能代替目标主机验证。换机后先运行训练入口的 `--dry-run` / `--check-only`，
再在实际分配的 NPU 上验证训练。
