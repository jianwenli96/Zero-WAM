"""Run normal training with per-rank sample/shape/memory diagnostics.

Use TRAIN_DIAGNOSTICS=1 with script/train_humangen_wan_npu.sh. Boundary
synchronizations affect timing, so use the same entrypoint for both A/B runs.
No model, optimizer, sampling or crop behavior is replaced by this wrapper.
"""
import json
import os
import time
import traceback
from pathlib import Path

from wan_va import train
from wan_va.dataset.icl_lerobot_latent_dataset import ICLLeRobotLatentDataset

torch = train.torch
root = Path(os.environ['ZERO_WAM_SAVE_ROOT'])
root.mkdir(parents=True, exist_ok=True)
rank = int(os.environ.get('RANK', '0'))
log = root / f'diagnostics-rank{rank}.jsonl'


def memory():
    return {key: getattr(torch.npu, key)() / 2**30 for key in (
        'memory_allocated', 'memory_reserved',
        'max_memory_allocated', 'max_memory_reserved')}


def emit(event, **data):
    with log.open('a') as handle:
        handle.write(json.dumps(dict(event=event, rank=rank, pid=os.getpid(),
                                     time=time.time(), **data), default=str) + '\n')


original_item = ICLLeRobotLatentDataset.__getitem__


def item(self, idx):
    result = original_item(self, idx)
    result['_diagnostic_sample'] = json.dumps(dict(
        root=str(self.root), local_index=int(idx),
        meta=self.new_metas[idx % len(self.new_metas)]), default=str)
    return result


ICLLeRobotLatentDataset.__getitem__ = item
original_prepare = train.Trainer._prepare_input_dict


def prepare(self, batch):
    result = original_prepare(self, batch)
    video = result['latent_dict']['grid_id'].shape[-1]
    action = result['action_dict']['grid_id'].shape[-1]
    human = (result['icl_latent_dict']['grid_id'].shape[-1]
             if result['icl_latent_dict'] is not None else 0)
    emit('prepared', step=self.step + 1,
         robot_shape=list(batch['latents'].shape),
         human_shape=list(batch['icl_latents'].shape),
         video_tokens=video, action_tokens=action, human_tokens=human,
         total_backbone_tokens=2 * (video + action) + human,
         chunk_size=result['chunk_size'], window_size=result['window_size'],
         memory_gib=memory())
    return result


train.Trainer._prepare_input_dict = prepare
original_step = train.Trainer._train_step


def step(self, batch, batch_idx):
    torch.npu.synchronize()
    torch.npu.reset_peak_memory_stats()
    started = time.perf_counter()
    emit('start', step=self.step + 1,
         robot_shape=list(batch['latents'].shape),
         human_shape=list(batch['icl_latents'].shape),
         sample=batch.get('_diagnostic_sample'),
         human_path=batch.get('icl_human_latent_path'),
         learning_rate=self.optimizer.param_groups[0]['lr'], memory_gib=memory())
    try:
        result = original_step(self, batch, batch_idx)
        torch.npu.synchronize()
        emit('success', step=self.step + 1, seconds=time.perf_counter() - started,
             memory_gib=memory(),
             losses={k: float(result[k].item()) for k in ('latent_loss', 'action_loss', 'mcp_loss')},
             optimizer_step_skipped=result.get('optimizer_step_skipped'))
        return result
    except Exception as exc:
        emit('error', step=self.step + 1, seconds=time.perf_counter() - started,
             error=str(exc), traceback=traceback.format_exc(), memory_gib=memory())
        raise


train.Trainer._train_step = step
if __name__ == '__main__':
    emit('launch')
    train.init_logger()
    train.main()
