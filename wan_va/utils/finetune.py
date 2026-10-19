"""Explicit, testable post-training learning-rate schedule."""
import math


def cosine_warmup_ratio(step, warmup_steps, total_steps, min_ratio=0.1):
    if not 0 <= min_ratio <= 1 or not 0 <= warmup_steps < total_steps:
        raise ValueError('Need 0 <= warmup < total_steps and 0 <= min_ratio <= 1')
    if step < warmup_steps:
        return step / max(warmup_steps, 1)
    progress = min(1., (step - warmup_steps) / (total_steps - warmup_steps))
    return min_ratio + (1 - min_ratio) * (1 + math.cos(math.pi * progress)) / 2
