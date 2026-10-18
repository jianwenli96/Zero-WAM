<h1 align="center">Zero-WAM:<br>In-Context World-Action Modeling from Human Videos for Open-Ended Task Generalization</h1>

<p align="center">
  <strong>
    <a href="https://jiaming-zhou.github.io/">Jiaming Zhou</a> &nbsp;
    <a href="https://zqh0253.github.io/">Qihang Zhang</a><sup>*</sup> &nbsp;
    <a href="https://gangweix.github.io/">Gangwei Xu</a> &nbsp;
    <a href="https://alfayoung.github.io/">Cunxin Fan</a> &nbsp;
    <a href="https://github.com/robbyant-research/Zero-WAM">Yujie Zhao</a> &nbsp;
    <a href="https://github.com/robbyant-research/Zero-WAM">Ruilin Wang</a> &nbsp;
    <a href="https://ymluo1214.github.io/">Yiming Luo</a> &nbsp;
    <a href="https://yangs03.github.io/">Shuai Yang</a> &nbsp;
    <a href="https://scholar.google.com/citations?user=Hnh87z4AAAAJ&hl=en">Xing Zhu</a> &nbsp;
    <a href="https://shenyujun.github.io/">Yujun Shen</a> &nbsp;
    <a href="https://junweiliang.me">Junwei Liang</a><sup>&dagger;</sup> &nbsp;
    <a href="https://justimyhxu.github.io/">Yinghao Xu</a><sup>&dagger;</sup>
  </strong>
  <br>
  Robbyant &nbsp;&middot;&nbsp; HKUST (GZ) &nbsp;&middot;&nbsp; HKUST
  <br>
  <sup>*</sup>Project Lead &nbsp;&nbsp; <sup>&dagger;</sup>Corresponding Authors
</p>

<p align="center">
  <a href="https://robbyant-research.github.io/Zero-WAM/"><img src="https://img.shields.io/badge/Project-Page-blue" alt="Project page"></a>
  <a href="https://arxiv.org/pdf/2608.26103"><img src="https://img.shields.io/badge/Paper-arXiv-b31b1b" alt="Paper"></a>
  <a href="https://github.com/robbyant-research/Zero-WAM"><img src="https://img.shields.io/badge/Code-available-brightgreen" alt="Code"></a>
  <a href="https://huggingface.co/robbyant-research/zero-wam-pretrain"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Model-pretrain-yellow" alt="Pretrained model"></a>
  <a href="https://huggingface.co/robbyant-research/zero-wam-posttrain-robotwin"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Model-robotwin-yellow" alt="Robotwin model"></a>
  <a href="https://huggingface.co/datasets/robbyant-research/HumanGen"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Data-HumanGen-yellow" alt="HumanGen dataset"></a>
  <a href="LICENSE.txt"><img src="https://img.shields.io/badge/License-Apache_2.0-blue" alt="License"></a>
</p>

<p align="center">
  <a href="https://robbyant-research.github.io/Zero-WAM/">
    <img src="assets/figures/framework-v1.0.webp" alt="Zero-WAM framework overview" width="95%">
  </a>
</p>

## Overview

Zero-WAM targets zero-shot cross-task robotic manipulation, where a policy must execute tasks that were never practiced during training using only deployment-time context. It brings in-context learning to robotics by treating human demonstration videos as visual task specifications, enabling a causal video-action policy to predict future robot observations and executable actions from either language instructions or human video prompts.

## Highlights

- We formulate zero-shot robotic task generalization as in-context world-action modeling, where one causal policy supports both language and human videos as task instructions.
- We introduce a scalable HumanGen pipeline that converts task-sampled robot trajectories into semantically matched human video instructions, yielding 74.2K human-robot ICL pairs over 8.6K tasks.
- We propose Zero-WAM with an in-context future chunk prediction objective to reduce shortcut learning and strengthen the use of human video prompts.
- We demonstrate zero-shot cross-task generalization in RoboTwin 2.0 and real-world unseen task configurations without collecting corresponding robot data or updating model parameters.

## Data Overview

Zero-WAM builds its training data around task diversity rather than raw trajectory count. Task-diverse VA re-samples five public robotic video-action datasets at the task level, yielding more than 6K manipulation tasks and about 400K robot trajectories per training epoch. HumanGen complements this robot-domain corpus with 74.2K human-robot ICL pairs over 8.6K tasks, spanning external, in-house, simulation, and real-world sources.

<p align="center">
  <img src="assets/figures/data_overview.png" alt="Zero-WAM data overview" width="95%">
</p>

## Data Pipeline

The in-context human video generation pipeline starts from task-sampled robot trajectories that retain executable actions. A VLM parses task semantics and object-state changes, an image editor converts the first robot frame into a human observation, and a video generation model synthesizes the corresponding human manipulation video. Each generated video is filtered for semantic preservation and physical plausibility before being paired back with the original robot trajectory as an ICL sample.

<p align="center">
  <img src="assets/figures/data_pipeline.png" alt="Zero-WAM data pipeline" width="95%">
</p>

## RoboTwin 2.0 Zero-Shot Evaluation

Zero-WAM achieves 46.95% average zero-shot success on seven unseen RoboTwin 2.0 tasks, outperforming LingBot-VA by 29.50 percentage points.

| Task | WAN-Action | LingBot-VA | **Zero-WAM** |
| --- | ---: | ---: | ---: |
| Place object on scale | 3.00 &plusmn; 2.16 | 6.17 &plusmn; 4.87 | **24.67 &plusmn; 2.05** |
| Stamp seal | 7.33 &plusmn; 1.25 | 3.67 &plusmn; 2.49 | **47.00 &plusmn; 4.55** |
| Open microwave | 2.26 &plusmn; 1.60 | 29.33 &plusmn; 10.66 | **59.00 &plusmn; 2.83** |
| Move stapler to pad | 10.67 &plusmn; 1.70 | 23.33 &plusmn; 8.22 | **69.14 &plusmn; 2.93** |
| Place bread in basket | 15.26 &plusmn; 2.55 | 17.33 &plusmn; 6.18 | **35.00 &plusmn; 3.74** |
| Place empty cup | 38.33 &plusmn; 2.05 | 42.33 &plusmn; 7.85 | **84.87 &plusmn; 0.18** |
| Stack blocks three | 0.00 &plusmn; 0.00 | 0.00 &plusmn; 0.00 | **9.00 &plusmn; 2.16** |
| **Average** | **10.98 &plusmn; 1.07** | **17.45 &plusmn; 1.40** | **46.95 &plusmn; 0.72** |

## Environment

For CUDA/RoboTwin, use Python 3.10 with a CUDA-enabled PyTorch environment, then install the project dependencies. For Ascend/ALOHA and ModelArts, see [ALOHA post-training and RTC](#aloha-post-training-and-real-time-chunking).

```bash
python -m pip install -r requirements.txt --no-build-isolation
```

Robotwin evaluation also requires a working Robotwin 2.0 installation. Follow the official Robotwin installation guide:

https://robotwin-platform.github.io/doc/usage/robotwin-install.html

See [INSTALL.md](INSTALL.md) for the exact tested environment (Python 3.10, PyTorch 2.9.0, CUDA 12.6), the tested Robotwin revision, and **the two unseen-task success-condition updates**. Run all commands below from the repository root.

```bash
export PROJECT_ROOT=/path/to/Zero-WAM
cd "${PROJECT_ROOT}"
```

## HumanGen Data

This repository releases two HumanGen subsets used by Zero-WAM: **Simulation ICL (RoboTwin)** and **Pre-training ICL (External)**. Install the Hugging Face CLI first:

```bash
python -m pip install "huggingface_hub[cli]"
```

### Simulation ICL (RoboTwin)

Simulation ICL (RoboTwin) contains ~2,500 human-robot ICL pairs across 50 RoboTwin 2.0 tasks, with 43 tasks used for training and seven unseen tasks used for evaluation. We release the robot videos, robot actions, human-video latents, robot-video latents, and pairing manifest needed for RoboTwin post-training and evaluation.

If you only need Robotwin ICL data for post-training or evaluation, download the minimal subset:

```bash
hf download Robbyant-Research/HumanGen \
  --repo-type dataset \
  --local-dir "${PROJECT_ROOT}/data/HumanGen" \
  --include "icl_configs/*" \
  --include "robotwin_data/**" \
  --include "human_latents/robotwin/**"
```

The released Robotwin data follows this layout:

```text
data/HumanGen/
├── robotwin_data/<task>/{data,meta,videos,latents}
├── human_latents/robotwin/run_*/samples/*/*.pth
└── icl_configs/ICL_config_robotwin.json
```

### ICL Pre-training Data

ICL Pre-training Data provides the public-dataset ICL data used during pre-training. It is built from five robotic video-action datasets: AgiBot, InternData-A1, Open X-Embodiment (Bridge), RoboCOIN, and RoboMIND. We release the paired robot videos, robot actions, robot-video latents, and precomputed human-video latents used for co-training.

The five ICL pre-training sources use the following HumanGen dataset keys and paths:

| `DATASETS` key | Robot data and latents | Human latents | Human data | Pairing manifest |
| --- | --- | --- | --- | --- |
| `agibot` | `agibot_data/` | `human_latents/agibot/` | `human_data/agibot/` | `icl_configs/ICL_config_agibot.json` |
| `robocoin` | `robocoin_data/` | `human_latents/robocoin/` | `human_data/robocoin/` | `icl_configs/ICL_config_robocoin.json` |
| `robomind` | `robomind_data/` | `human_latents/robomind/` | `human_data/robomind/` | `icl_configs/ICL_config_robomind.json` |
| `interna1` | `interna1_data/` | `human_latents/interna1/` | `human_data/interna1/` | `icl_configs/ICL_config_interna1.json` |
| `oxe` | `oxe_data/` | `human_latents/oxe/` | `human_data/oxe/` | `icl_configs/ICL_config_oxe.json` |

To use the released Pre-training ICL data ("External" subset) for co-training, download the Robotwin subset plus the five public-dataset robot datasets and the archived human latents:

```bash
hf download Robbyant-Research/HumanGen \
  --repo-type dataset \
  --local-dir "${PROJECT_ROOT}/data/HumanGen" \
  --include "icl_configs/*" \
  --include "robotwin_data/**" \
  --include "human_latents/**" \
  --include "agibot_data/**" \
  --include "robocoin_data/**" \
  --include "robomind_data/**" \
  --include "interna1_data/**" \
  --include "oxe_data/**"
```

The public-dataset robot data and non-Robotwin human latents are stored as `tar.zst` shards to avoid transferring hundreds of thousands of small files.

Install `zstd` with apt:

```bash
sudo apt-get update && sudo apt-get install -y zstd
```

Or install it with conda:

```bash
conda install -c conda-forge zstd
```

Extract all released `tar.zst` shards from the HumanGen root:

```bash
cd "${PROJECT_ROOT}/data/HumanGen"
for dir in human_latents agibot_data robocoin_data robomind_data interna1_data oxe_data; do for f in "$dir"/part-*.tar.zst; do test -e "$f" && tar -I zstd -xf "$f" -C .; done; done
```

After extraction, `human_latents/` contains `robotwin/` as regular files and the five public-dataset latent folders from the shards. Each `*_data/` directory contains the corresponding extracted robot data and robot-video latents.

Robot-video latents are stored with each LeRobot task under `latents/`. Human-video latents use the matching `human_latents/<dataset>/` hierarchy.

The generated human videos are optional for training. Download `human_data/` if you want the original HumanGen videos:

```bash
hf download Robbyant-Research/HumanGen \
  --repo-type dataset \
  --local-dir "${PROJECT_ROOT}/data/HumanGen" \
  --include "human_data/**"
```

## Model Checkpoints

Two Zero-WAM checkpoints are released on the Hugging Face Hub:

| Model | Description | Arch | Parameters | Size |
| --- | --- | --- | ---: | ---: |
| [`zero-wam-pretrain`](https://huggingface.co/robbyant-research/zero-wam-pretrain) | Pre-trained Zero-WAM | MoT | 10.8B | ~35 GB |
| [`zero-wam-posttrain-robotwin`](https://huggingface.co/robbyant-research/zero-wam-posttrain-robotwin) | Robotwin post-trained Zero-WAM | MoT | 10.8B | ~35 GB |

Download the released Zero-WAM checkpoints as needed:

```bash
hf download robbyant-research/zero-wam-pretrain \
  --local-dir "${PROJECT_ROOT}/checkpoints/zero-wam-pretrain"
hf download robbyant-research/zero-wam-posttrain-robotwin \
  --local-dir "${PROJECT_ROOT}/checkpoints/zero-wam-posttrain-robotwin"
```

## Robotwin Inference

Configure the model and simulator paths:

```bash
export MODEL_PATH="${PROJECT_ROOT}/checkpoints/zero-wam-posttrain-robotwin"
export ROBOTWIN_ROOT=/path/to/Robotwin
```

### Single Task

Start the inference server on one GPU:

```bash
CUDA_VISIBLE_DEVICES=0 \
bash evaluation/robotwin/launch_server.sh
```

In another terminal, evaluate one unseen task for 100 trials:

```bash
ICL_CFG=5 \
TARGET_TEXT_CFG=-1 \
SEED=0 \
bash evaluation/robotwin/launch_client.sh \
  /path/to/eval_results \
  place_object_scale
```

`ICL_CFG=5` applies classifier-free guidance between the ICL-conditioned and
non-ICL robot-video branches. `TARGET_TEXT_CFG=-1` disables target-text CFG and
uses the empty-text embedding for the target robot stream. `SEED=0` fixes the
Robotwin rollout seed. The ICL demonstration for each task is fixed by
`evaluation/robotwin/robotwin_icl_human_videos.py`.

### Eight-GPU Evaluation

Run the full unseen-task evaluation:

```bash
ICL_CFG=5 \
TARGET_TEXT_CFG=-1 \
SEED=0 \
TEST_NUM=100 \
SAVE_ROOT=/path/to/eval_results \
bash evaluation/robotwin/run_icl_eval.sh
```

The launcher evaluates seven unique tasks and repeats `place_empty_cup` to fill
eight GPU slots:

```text
stack_blocks_three
place_object_scale
stamp_seal
open_microwave
move_stapler_pad
place_bread_basket
place_empty_cup
place_empty_cup
```

The launcher starts eight independent one-GPU server/client pairs, waits until
all servers are healthy, and stops them after evaluation. The repeated
`place_empty_cup` run is saved under `place_empty_cup_repeat` so it cannot
overwrite the primary result.

Each task uses the fixed human demonstration in `evaluation/robotwin/robotwin_icl_human_videos.py`. Human-video paths are relative to `data/HumanGen`, and precomputed latents are used when available. Logs are written to `./logs`, generated videos to `./visualization`, and rollout results to the selected `SAVE_ROOT`.

## Training

Both training modes start from the released `zero-wam-pretrain` checkpoint. Configure the model and output paths:

```bash
export MODEL_PATH="${PROJECT_ROOT}/checkpoints/zero-wam-pretrain"
export ZERO_WAM_SAVE_ROOT=/path/to/output
```

Human video ICL, bidirectional human-video attention, the action branch, and MCP are enabled by default. The released configuration uses `drop_icl=0.1`, `droptext_target=0.4`, four one-block MCP modules, loss weights `[0.5, 0.25, 0.15, 0.1]`, MCP SNR shift `10`, and future-chunk stride `2`.

### Robotwin-only Post-training

Start eight-GPU post-training using only Robotwin data:

```bash
NGPU=8 \
DATASETS="robotwin:1.0" \
bash script/train.sh --init-worker 1
```

Keep `DATASETS="robotwin:1.0"` explicit for Robotwin-only training.
The loader always excludes the seven unseen evaluation tasks and requires the
remaining Robotwin training split to contain exactly 43 tasks. Training stops
with an error if this isolation check fails.

### RoboTwin + Pre-training ICL Co-training

Start eight-GPU co-training with RoboTwin and all five Pre-training ICL (External) datasets:

```bash
NGPU=8 \
DATASETS="robotwin:1.0,agibot:1.0,robocoin:1.0,robomind:1.0,interna1:1.0,oxe:1.0" \
bash script/train.sh --init-worker 1
```

Each value in `DATASETS` is the relative probability of selecting that dataset for the next sample. The weights are normalized automatically; the equal weights above give each of the six sources probability `1/6`. Change the values to set another sampling mixture. After a source is selected, samples are drawn uniformly from that dataset.

Each run creates a timestamp directory. Checkpoints are written to `ZERO_WAM_SAVE_ROOT/<timestamp>/checkpoints/checkpoint_step_N/transformer`. Run the repository tests with `python -m pytest -q`.

> **Note:** The `zero-wam-posttrain-robotwin` checkpoint reported in the paper was post-trained on a mixture of Task-diverse VA data, the complete Pre-training ICL collection, and Robotwin ICL data using specified sampling ratios. The released training data contains only a subset of the Pre-training ICL collection and the Robotwin ICL data. Post-training with the released data will produce a model that differs from our released checkpoint.

## ALOHA Post-training and Real-Time Chunking

This branch supports prepared ALOHA data, joint video/action time resampling,
configurable actions per video latent, and opt-in training-free real-time
chunking (RTC). The Ascend path has been checked with Python 3.12, PyTorch 2.6,
and `torch_npu` 2.6 on Ascend 910B3 devices with 64 GB HBM. Use a compatible CANN
image and Python environment; the CUDA dependencies above are specific to that
environment.

| Guide | Contents |
| --- | --- |
| [ALOHA data and training](docs/aloha_training.md) | Data preparation, human-video pairing, action statistics, and post-training |
| [RTC parameters and ModelArts (Chinese)](docs/aloha_rtc_parameters.md) | Parameter definitions, measured configurations, ModelArts mounts, and complete commands |
| [RTC runtime](docs/rtc.md) | Timing checks, controller API, interpolation, and recorded-data replay |

### Ascend and ModelArts Training

Set the paths in the training container, or in the ModelArts job's environment
settings. The prepared dataset includes the existing native-rate robot latents,
human latents, pairing manifest, and action statistics. Model directories and
all their symlink targets must be accessible on every node.

```bash
export PYTHON_BIN=/path/to/ascend-env/bin/python
export ALOHA_PREPARED_ROOT=/path/to/aloha_zerowam_v3
export MODEL_PATH=/path/to/initialization-model
export ZERO_WAM_SAVE_ROOT=/path/to/writable/shared/aloha_rtc_output
```

From the repository root, first check ten training steps and checkpoint saving:

```bash
bash script/train_modelarts_rtc.sh --num-steps 10 --save-interval 10
```

Then start post-training:

```bash
bash script/train_modelarts_rtc.sh
```

The defaults are ALOHA-only training, `action_stride=3`,
`actions_per_latent=32`, learning rate `1e-5`, batch size 1 per device,
gradient accumulation 1, 5,000 steps, and a checkpoint every 2,500 steps.
Training retains video, action, ICL, MCP, activation checkpointing, and FSDP.
The output root can also contain the default HF/XDG caches; run logs and
checkpoints are under its timestamp directories.

For ModelArts, select a custom Bash startup command and run the same script
once per node using its absolute mounted path. The launcher automatically reads
node count, node rank, and master address from the platform. Every node uses
all its visible NPUs by default: a visibility mask is measured through
`torch_npu.npu.device_count()`; otherwise the launcher uses the platform's
allocated card count or detects it at runtime. There is no fixed four- or
eight-card fallback. `NGPU` remains an optional numeric override.

For example, two nodes with eight visible NPUs each launch 16 workers. Nodes
must have the same local worker count. All nodes need the same code version,
shared data/model paths, and output root. Outside ModelArts, set `NNODES`,
`NODE_RANK`, and `MASTER_ADDR` on each node; device count remains automatic.
The default global batch size is the total number of training workers.
Missing multi-node settings, invalid ranks, and zero visible devices fail
before workers start. See the parameter guide for mount and environment details.

### RTC Inference

RTC defaults to disabled (`enable_rtc=False`). For the 32-action layout,
post-train with the defaults above and point `MODEL_PATH` at a complete model
root containing `transformer`, `vae`, `tokenizer`, `text_encoder`, and the saved
ALOHA action contract. See the parameter guide for composing this directory
from a training checkpoint.

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export MODEL_PATH=/path/to/posttrained/complete-model
NGPU=1 bash evaluation/aloha/launch_server.sh \
  --enable-rtc --steps 3 --action-steps 10 \
  --chunk-frames 4 --rtc-execution-horizon 64
```

The checkpoint records the training time layout; inference reads it
automatically and rejects conflicting overrides. At ALOHA's native 30 Hz:

| Setting | Value | Physical time |
| --- | ---: | ---: |
| Action stride | 3 | 10 Hz model targets, interpolated to 30 Hz control |
| Actions per latent, A | 32 | 3.2 seconds per action block |
| Future video latents, F | 4 | 128 future action targets, 12.8 seconds |
| Receding amount, s | 64 | Start replanning every 6.4 seconds |
| Video / action denoising steps | 3 / 10 | Configured at inference time |

The native VAE encoding stays unchanged; training selects every sixth native
robot latent to align with the longer action blocks. Use the RTC client to
maintain the asynchronous action queue, run guided warmup, and track the
maximum of a bounded latency history. RTC inference uses one unsharded device;
training can use multiple nodes. Invalid budgets and missed deadlines raise
errors, and the 32-action layout requires the RTC client.

Recorded-data replay of this layout completed 85.3 seconds and 12 queue swaps
with 3+10 denoising steps: mean latency 4.429 seconds, maximum 4.725 seconds,
against a 5.208-second budget including the configured reserve. The timing
probe used a legacy checkpoint; task quality needs matching post-training and
robot evaluation. A separate four-NPU, ten-step training check completed real
optimizer updates and checkpoint saving. Multi-node launch parameters have
been tested; cross-node model training has not yet been measured.

## Acknowledgments

Zero-WAM is built on [Next Forcing](https://github.com/gangweix/next-forcing) and [LingBot-VA](https://github.com/Robbyant/lingbot-va). We thank the authors for releasing their work.

## Citation

If you find Zero-WAM useful, please cite:

```bibtex
@misc{zhou2026zerowam,
  title = {Zero-WAM: In-Context World-Action Modeling from Human Videos for Open-Ended Task Generalization},
  author = {Zhou, Jiaming and Zhang, Qihang and Xu, Gangwei and Fan, Cunxin and Zhao, Yujie and Wang, Ruilin and Luo, Yiming and Yang, Shuai and Zhu, Xing and Shen, Yujun and Liang, Junwei and Xu, Yinghao},
  year = {2026},
  eprint = {2608.26103},
  archivePrefix = {arXiv},
  primaryClass = {cs.RO},
  url = {https://arxiv.org/abs/2608.26103}
}
```
