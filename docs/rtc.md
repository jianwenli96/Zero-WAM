# ALOHA training-free RTC

`enable_rtc` defaults to `False`. The existing reset / infer / compute_kv_cache
protocol and default 50+50, F=2 behavior remain available with RTC disabled.
RTC is an opt-in ICL inference path with a single unsharded device. Native-rate
RTC needs no checkpoint conversion. Changing the model's temporal stride needs
matching post-training; see [action_stride](aloha_training.md#延长动作时间跨度action_stride).

The action guidance follows the **training-free** `realtime_action` branch of
[Physical Intelligence's reference](https://github.com/Physical-Intelligence/real-time-chunking-kinetix/blob/9296f31d62d5bfeb5779dcb2f9bcf71ca37f448b/src/model.py)
and Algorithm 1 of [the RTC paper](https://arxiv.org/html/2506.07339v2).
The training-time `simulated_delay` branch is not used. The MIT notice is in
[third_party/rtc_LICENSE.txt](third_party/rtc_LICENSE.txt).

## Launch

```bash
cd /mnt/sfs_turbo/lijianwen/Codes/Zero-WAM
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ALOHA_PREPARED_ROOT=/mnt/sfs_turbo/lijianwen/Datasets/aloha_zerowam_v3
export MODEL_PATH=/mnt/sfs_turbo/lijianwen/Codes/Zero-WAM/outputs/aloha_step6000_validation/model
# Select an actually available device; do not interrupt existing jobs.
export ASCEND_RT_VISIBLE_DEVICES=6
NGPU=1 bash evaluation/aloha/launch_server.sh --enable-rtc
```

The native-rate RTC preset is F=16, video/action steps=1/1, ICL CFG=1, attention window=4,
30 Hz, execution horizon s=128, and H=256 future actions. Explicit `--steps`
and `--chunk-frames` override the preset. `--enable_rtc` is an alias.
RTC bypasses FSDP and enables equivalent indexed/slice cache pruning and
per-request reuse of attention masks, RoPE and text projections.
The text encoder is moved onto the inference device only during reset and
returned to CPU after encoding; cached text embeddings stay on the device.
Its prompt computation is unchanged, and its weights do not occupy device
memory during action/video inference. Parameter
weights remain frozen. The action VJP is computed with input gradients enabled;
do not wrap RTC inference in `torch.inference_mode()`.
Unused MCP training auxiliaries stay on CPU in RTC mode; the streaming video
and action branches do not call them. These weights remain in the checkpoint.

### Longer action intervals with F=4

Post-train with `bash script/train_dist.sh --action-stride 3` to keep 16 actions
per latent while making each latent interval three times longer. The checkpoint
records the joint action/latent time stride. Inference reads it automatically:

```bash
# MODEL_PATH must point to the composed stride=3 post-trained model directory.
NGPU=1 bash evaluation/aloha/launch_server.sh --enable-rtc --steps 2
```

For a stride>1 checkpoint the defaults become F=4, H=64 model actions, s=32
model actions and seed latency=2 seconds; explicit flags override these values.
`--steps` controls both video and action denoising; `--action-steps` can override
the action branch independently (e.g. `--steps 4 --action-steps 2` means 4+2).
Calibrate on the target device. At 30 Hz native control with stride=3 the model target rate is 10 Hz,
the prediction covers 6.4 seconds, and replanning occurs every 3.2 seconds.
The prefix and initial z0/PAD are still prefilled separately, so F counts future
latent frames. This keeps the generated F within the training maximum of four.

`--action-stride 3` can explicitly probe timing on a legacy checkpoint without
sampling metadata; the server warns that this does not validate policy quality.
A saved training contract cannot be overridden with a different stride.
Stride>1 requires the new RTC client and protocol version 2; ordinary open-loop
clients and older RTC clients cannot execute this time layout.

Robot RGB acquisition and native VAE encoding are unchanged. Training selects
z0,z3,z6,... from those encoded latents. Online feedback encodes all native-rate
frames, then selects every third resulting latent using the same phase. This
is different from feeding RGB at lower fps into the VAE. A 3.2-second executed
prefix supplies 24 stride-4 RGB observations, yielding six native latents of
which two are retained. The native controller runs at 30 Hz and interpolates
the 10 Hz anchors: linear positions/joints, normalized quaternion SLERP, and
left-held grippers. Model actions and protocol timestamps use coarse action
units; the controller tracks native ticks separately.

## Timing checks and latency buffer

`d <= s <= H-d` must hold in model action units. Let
`model_hz=control_hz/action_stride`. The conservative forecast is
`d=ceil(model_hz * (max(latency_buffer)*multiplier + margin)) + lookahead`,
where `lookahead=1` for stride>1 and zero otherwise. The extra anchor keeps
an interpolation segment already being executed unchanged during a queue swap.

Defaults:

| Setting | CLI flag | Default |
|---|---|---:|
| Control frequency | `--rtc-control-hz` | 30 |
| Joint action/latent time stride | `--action-stride` | checkpoint value, otherwise 1 |
| Execution horizon | `--rtc-execution-horizon` | 128 |
| Latency history length | `--rtc-delay-buffer-size` | 10 |
| Seed raw latency, seconds | `--rtc-initial-latency` | 3.342 |
| Latency multiplier | `--rtc-latency-multiplier` | 1.2 |
| Extra latency, seconds | `--rtc-latency-margin` | 0.05 |
| Guidance clip beta | `--rtc-max-guidance-weight` | 5 |
| History window | `--rtc-attn-window` | 4 |
| ICL guidance | `--rtc-icl-guidance-scale` | 1 |

The seed gives d=122, so `122 <= 128 <= 134`. It is an initial estimate,
not a measured latency guarantee. Invalid budgets fail before model loading.
For stride=3, F=4 and s=32, seed=2 gives d=26, so `26 <= 32 <= 38`.
With the same 20% + 50ms reserve and one interpolation anchor, raw latency
must be at most about 2.542 seconds; the full 3.2 seconds is not all available
for inference. These values are budgets, not guarantees on a shared device.
The execution horizon is fixed; an increasing delay is not hidden by silently
changing the controller frequency or execution horizon.

The client must complete at least three guided warmup rounds while the robot
is idle. Their timings are retained; the final two warm timings initialize the
bounded buffer together with the configured seed. Warmup history is then reset.
The online buffer measures background queueing, image packing, network/model
inference and response availability in the control loop. Its maximum predicts
the next delay; measurements expire after the configured buffer length.

A request exceeding its forecast or timing inequality, a queue underrun,
misaligned timestamps, invalid feedback, or a background RPC failure raises
`RTCError` / `RTCTimingError`. Failures are latched. Reset the server and create
a new client before retrying. The caller must handle the exception by stopping
its controller; this library does not provide a robot-specific emergency stop.

## Model/cache alignment

The initial observation z0 and normalized action PAD are prefilled separately.
With stride=1, the first generated chunk contains z1..z16 and **256 usable actions**,
without a discarded first action block. At t=128, only the executed 128 actions
and their 32 stride-4 observations are committed as history. The new prediction
covers actions 128..383. The old unexecuted actions 128..255 guide the overlap.
The temporal RoPE offset advances by the executed prefix, not by F=16.

Guidance uses the reference exponential prefix weights, a clean-action estimate,
input VJP and clipped correction. Zero-WAM integrates sigma downward, so the
reference tau=1-sigma velocity and correction signs are converted accordingly.
After denoising, the committed prefix is restored exactly. The controller also
preserves every command already issued when replacing its action queue.

Real observations replace temporary generated-video cache entries. Only
executed commands enter action history. If a downstream controller changes a
returned command, supply its actual executed value on the next `get_action` call.

## Controller API

```python
from evaluation.aloha.rtc_client import RTCPolicy, WebsocketRTCTransport

policy = RTCPolicy(WebsocketRTCTransport('127.0.0.1', 29536))
try:
    # get_rgb returns three uint8 RGB arrays keyed by the ALOHA camera names.
    # A callable refreshes the initial observation after warmup.
    policy.start(get_rgb, {
        'prompt': task_prompt,
        'use_icl': True,
        'icl_latent_path': human_latent_path,
        'video_guidance_scale': -1,
        'seed': 42,
    })
    # Your controller schedules this callback at 30 Hz. get_action does not sleep.
    while running:
        command = policy.get_action(get_rgb(), executed_action=last_executed_action)
        last_executed_action = execute_absolute_aloha_command(command)
except Exception:
    stop_robot()
    raise
finally:
    policy.close()
```

The commands have the existing 28-channel absolute ALOHA contract. Real robot
I/O and stopping are application callbacks, not provided by this repository.
The client snapshots/downsizes observations to 224x288 as they arrive and keeps
every fourth observation. This uint8 resize can change outputs versus the old
FP32 server resize, as documented in the latency experiments.

The ordinary `evaluation.aloha.open_loop` and Robotwin client do not implement
the timestamped RTC protocol. Use the new client with an RTC-enabled server.

## Recorded-data test

```bash
python -m evaluation.aloha.rtc_replay \
  --root "$ALOHA_PREPARED_ROOT" --port 29536 \
  --episode 1 --ticks 512 --output /tmp/zero_wam_rtc_replay
```

This runs the actual background inference and VJP-guided queue swaps at 30 Hz.
It issues no robot commands. Recorded feedback is independent of predicted
actions, so successful playback validates implementation and timing, not robot
task success. Cold/warm timings, measured online delays and issued actions are
saved in the output directory, including failed runs.

The original stride=1, F=16 preset exceeds this checkpoint's training maximum chunk length of 4. The new
prefill/cache alignment and 1/1 inference also need task-quality validation.
Timing failures are detected, but finite tests cannot establish a hard
worst-case latency bound on a shared accelerator/OS.
The stride=3, F=4 path avoids this chunk-length extension but needs a model
post-trained with matching temporal sampling. Timing tests with the old
step-6000 checkpoint establish software/latency behavior only.

## Validation of this implementation

The CPU test suite covers timing guards, the reference guidance sign and mask,
input VJP through a real tiny Transformer, cache pruning equivalence, executed
prefix alignment, asynchronous failures and a real local WebSocket queue test.
It also runs the existing ALOHA/ICL regression tests. The complete step-6000
checkpoint passed recorded-data RTC playback on Ascend 910B3 at 30 Hz, with
F=16, 1/1 denoising, true VJP guidance, H=256 and s=128. Online request latency
was approximately two seconds in that run. These recordings do not establish
real-robot task success; repeat calibration on the deployment device.
