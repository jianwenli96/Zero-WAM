"""Audit a capacity profile against an existing shape census, without tensor I/O."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from wan_va.dataset.sequence_crop import SequenceCapacity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--census', required=True, type=Path)
    parser.add_argument('--profile', type=Path, default=Path(__file__).resolve().parents[1]
                        / 'wan_va/configs/sequence_capacity_8npu.json')
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    capacity = SequenceCapacity.load(args.profile)
    rows = json.loads(args.census.read_text())
    summary = defaultdict(lambda: dict(samples=0, cropped=0, original_frames=0, retained_frames=0))
    records = []
    for row in rows:
        limit = capacity.frame_limit(row['robot_tokens_per_frame'],
                                     row['action_tokens_per_frame'], row['human_tokens'])
        retained = min(row['robot_frames'], limit)
        source = summary[row['source']]
        source['samples'] += 1
        source['cropped'] += retained < row['robot_frames']
        source['original_frames'] += row['robot_frames']
        source['retained_frames'] += retained
        records.append(dict(source=row['source'], root=row['root'], local_index=row['local_index'],
            robot_frames=row['robot_frames'], window_frames=retained, frame_limit=limit,
            human_frames=row['human_frames'], human_tokens=row['human_tokens'],
            robot_height=row['robot_height'], robot_width=row['robot_width'],
            robot_tokens_per_frame=row['robot_tokens_per_frame'],
            action_tokens_per_frame=row['action_tokens_per_frame'],
            cropped=retained < row['robot_frames']))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = dict(profile=capacity.name, sources=dict(summary),
                  samples=len(rows), cropped=sum(s['cropped'] for s in summary.values()),
                  dropped_samples=0)
    (args.output_dir / 'summary.json').write_text(json.dumps(report, indent=2))
    with (args.output_dir / 'windows.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
