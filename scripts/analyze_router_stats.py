#!/usr/bin/env python3
import argparse
import csv
import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np


def _load_npz(path: Path) -> Dict[str, np.ndarray]:
    data = np.load(path)
    return {key: data[key] for key in data.files}


def _ensure_counts(data: Dict[str, np.ndarray], num_experts: int) -> np.ndarray:
    if 'counts' in data:
        return data['counts']
    if 'indices' not in data:
        raise ValueError('Missing indices for computing counts.')
    return np.bincount(data['indices'].astype(np.int64), minlength=num_experts)


def _entropy(probs: np.ndarray) -> float:
    probs = probs[probs > 0]
    if probs.size == 0:
        return 0.0
    return float(-(probs * np.log(probs)).sum())


def _summarize_counts(counts: np.ndarray) -> Dict[str, float]:
    total = counts.sum()
    if total == 0:
        return {
            'total_tokens': 0.0,
            'utilization': 0.0,
            'entropy': 0.0,
            'top1_share': 0.0,
            'top4_share': 0.0,
        }
    probs = counts / total
    top1_share = float(np.sort(probs)[-1])
    top4_share = float(np.sort(probs)[-4:].sum()) if probs.size >= 4 else float(probs.sum())
    utilization = float((counts > 0).sum() / counts.size)
    return {
        'total_tokens': float(total),
        'utilization': utilization,
        'entropy': _entropy(probs),
        'top1_share': top1_share,
        'top4_share': top4_share,
    }


def _flatten_tokens(data: Dict[str, np.ndarray]) -> Iterable[Tuple[int, int, int, int]]:
    if 'input_ids' not in data or 'position_ids' not in data or 'sample_ids' not in data:
        return []
    input_ids = data['input_ids']
    position_ids = data['position_ids']
    sample_ids = data['sample_ids']
    if input_ids.ndim != 2 or position_ids.ndim != 2:
        return []
    if sample_ids.ndim == 1:
        sample_ids = np.repeat(sample_ids[:, None], input_ids.shape[1], axis=1)
    indices = data.get('indices')
    if indices is None:
        return []
    indices = indices.reshape(-1)
    flat_input = input_ids.reshape(-1)
    flat_pos = position_ids.reshape(-1)
    flat_sample = sample_ids.reshape(-1)
    if not (len(indices) == len(flat_input) == len(flat_pos) == len(flat_sample)):
        return []
    return zip(flat_sample.tolist(), flat_pos.tolist(), flat_input.tolist(), indices.tolist())


def analyze_dir(stats_dir: Path, num_experts: int, max_tokens: int) -> None:
    files = sorted(stats_dir.glob('router_stats_iter_*.npz'))
    if not files:
        raise FileNotFoundError(f'No router_stats_iter_*.npz found in {stats_dir}')

    summary_rows: List[Dict[str, float]] = []
    token_tracks: Dict[Tuple[int, int, int], List[int]] = {}

    for path in files:
        data = _load_npz(path)
        counts = _ensure_counts(data, num_experts)
        metrics = _summarize_counts(counts)
        metrics['file'] = path.name
        summary_rows.append(metrics)

        if max_tokens <= 0:
            continue
        for idx, (sample_id, pos_id, token_id, expert_id) in enumerate(_flatten_tokens(data)):
            if idx >= max_tokens:
                break
            key = (sample_id, pos_id, token_id)
            token_tracks.setdefault(key, []).append(expert_id)

    summary_path = stats_dir / 'router_stats_summary.csv'
    with summary_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    if token_tracks:
        token_path = stats_dir / 'router_stats_token_changes.csv'
        with token_path.open('w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            header = ['sample_id', 'position_id', 'input_id'] + [p.name for p in files]
            writer.writerow(header)
            for (sample_id, pos_id, token_id), experts in token_tracks.items():
                row = [sample_id, pos_id, token_id] + experts
                writer.writerow(row)

    print(f'Wrote summary: {summary_path}')
    if token_tracks:
        print(f'Wrote token changes: {stats_dir / "router_stats_token_changes.csv"}')


def main() -> None:
    parser = argparse.ArgumentParser(description='Analyze router stats dumps.')
    parser.add_argument('--stats-dir', type=Path, required=True, help='Directory with router_stats_iter_*.npz files.')
    parser.add_argument('--num-experts', type=int, required=True, help='Total number of experts.')
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=0,
        help='Max tokens per file to track for per-token expert changes (0 disables).',
    )
    args = parser.parse_args()
    analyze_dir(args.stats_dir, args.num_experts, args.max_tokens)


if __name__ == '__main__':
    main()
