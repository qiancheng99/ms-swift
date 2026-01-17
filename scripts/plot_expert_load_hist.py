#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


def _load_counts(
    *,
    counts_path: Optional[Path],
    indices_path: Optional[Path],
    num_experts: Optional[int],
) -> np.ndarray:
    if counts_path is not None:
        if counts_path.suffix in {'.npy', '.npz'}:
            data = np.load(counts_path)
            if isinstance(data, np.lib.npyio.NpzFile):
                if 'counts' not in data:
                    raise ValueError('npz file must contain a "counts" array.')
                return np.asarray(data['counts'])
            return np.asarray(data)
        if counts_path.suffix == '.json':
            with counts_path.open('r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, dict) and 'counts' in data:
                data = data['counts']
            return np.asarray(data)
        raise ValueError(f'Unsupported counts file type: {counts_path.suffix}')

    if indices_path is None:
        raise ValueError('Either --counts or --indices must be provided.')
    if num_experts is None:
        raise ValueError('--num-experts is required when using --indices.')
    indices = np.load(indices_path)
    indices = np.asarray(indices).reshape(-1)
    counts = np.bincount(indices, minlength=num_experts)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description='Plot MoE expert load histogram.')
    parser.add_argument(
        '--counts',
        type=Path,
        default=None,
        help='Path to expert counts (.npy/.npz/.json). If provided, --indices is ignored.',
    )
    parser.add_argument(
        '--indices',
        type=Path,
        default=None,
        help='Path to routed expert indices (.npy). Requires --num-experts.',
    )
    parser.add_argument('--num-experts', type=int, default=None, help='Total number of experts.')
    parser.add_argument('--normalize', action='store_true', help='Normalize counts into probabilities.')
    parser.add_argument('--title', type=str, default='Expert Load Histogram', help='Plot title.')
    parser.add_argument('--output', type=Path, default=Path('expert_load_hist.png'), help='Output image path.')
    args = parser.parse_args()

    counts = _load_counts(counts_path=args.counts, indices_path=args.indices, num_experts=args.num_experts)
    if args.normalize:
        total = counts.sum()
        counts = counts / total if total > 0 else counts

    x = np.arange(len(counts))
    plt.figure(figsize=(10, 4))
    plt.bar(x, counts, width=0.9)
    plt.xlabel('Expert ID')
    plt.ylabel('Load' if not args.normalize else 'Load Probability')
    plt.title(args.title)
    plt.tight_layout()
    plt.savefig(args.output)
    print(f'Saved histogram to: {args.output}')


if __name__ == '__main__':
    main()
