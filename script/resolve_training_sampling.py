#!/usr/bin/env python3
"""解析联合训练采样配置；只使用标准库，不初始化 NPU。"""
import argparse
import json
import math
from pathlib import Path

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / 'wan_va/configs/humangen_robotwin_sampling.json'
EXTERNAL_SOURCES = ('agibot', 'interna1', 'oxe', 'robocoin', 'robomind')


def resolve(config_path=DEFAULT_CONFIG, datasets=None):
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text())
    counts = config['paper_task_counts']
    if set(counts) != set(EXTERNAL_SOURCES):
        raise ValueError('采样配置必须包含五个 HumanGen 外部来源')
    if any(not isinstance(n, int) or isinstance(n, bool) or n <= 0 for n in counts.values()):
        raise ValueError('task 数必须为正整数')
    ratio = float(config['humangen_to_robotwin_ratio'])
    exponent = float(config['task_count_exponent'])
    if not math.isfinite(ratio) or ratio <= 0 or not math.isfinite(exponent) or not 0 <= exponent <= 1:
        raise ValueError('混合比例必须为有限正数，指数必须在 [0, 1] 内')
    if config['within_source'] != 'uniform_samples':
        raise ValueError('当前仅支持来源内均匀抽样')
    if datasets is None:
        scores = {name: counts[name] ** exponent for name in EXTERNAL_SOURCES}
        total = sum(scores.values())
        probabilities = {name: ratio / (ratio + 1) * score / total for name, score in scores.items()}
        probabilities['robotwin'] = 1 / (ratio + 1)
    else:
        probabilities = {}
        for item in datasets.split(','):
            if item.count(':') != 1:
                raise ValueError('DATASETS 格式应为 name:weight，以逗号分隔')
            name, raw = (part.strip().lower() for part in item.split(':'))
            if name not in (*EXTERNAL_SOURCES, 'robotwin') or name in probabilities:
                raise ValueError(f'未知或重复的数据来源：{name}')
            weight = float(raw)
            if not math.isfinite(weight) or weight <= 0:
                raise ValueError(f'{name} 的权重必须为有限正数')
            probabilities[name] = weight
        total = sum(probabilities.values())
        if not math.isfinite(total):
            raise ValueError('来源权重总和必须有限')
        probabilities = {name: weight / total for name, weight in probabilities.items()}
    resolved = ','.join(f'{name}:{weight:.17g}' for name, weight in probabilities.items())
    return dict(policy=config['name'] if datasets is None else 'custom_DATASETS',
                config_path=str(config_path), configured_policy=config,
                datasets=resolved, source_probabilities=probabilities,
                group_probabilities={'humangen': sum(p for n, p in probabilities.items() if n != 'robotwin'),
                                     'robotwin': probabilities.get('robotwin', 0.0)},
                within_source='uniform_samples',
                note='来源内均匀抽样；平方根任务数加权是近似策略，不代表已复现论文的任务均衡采样。')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(DEFAULT_CONFIG))
    parser.add_argument('--datasets', default=None)
    parser.add_argument('--output-format', choices=['datasets', 'json'], default='datasets')
    args = parser.parse_args()
    report = resolve(args.config, args.datasets)
    print(report['datasets'] if args.output_format == 'datasets' else json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
