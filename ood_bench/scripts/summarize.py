import argparse
import json
from pathlib import Path
from typing import Iterable, Union
from collections import defaultdict

import numpy as np


def summarize_trials(output_dir: Union[Path, str]) -> dict:
    result_paths = list(Path(output_dir).glob('*/result.json'))
    if not result_paths:
        print(f'unable to find any completed trial under {output_dir}')
        return

    result_all = defaultdict(list)
    for i, result_path in enumerate(result_paths):
        with result_path.open('r') as f:
            result = json.load(f)
        for name, value in result.items():
            result_all[name].append(value)

    summary = {
        'completed_trials': [path.parent.name for path in result_paths],
        'result': {}
    }
    for name, values in result_all.items():
        arr = np.array(values, dtype=np.float32)
        summary['result'][name] = {'mean': float(arr.mean()), 'std': float(arr.std())}

    print(f'Summary for trials under {output_dir}')
    print(f'Result averaged over {len(result_paths)} trial(s):')
    for name, stats in summary['result'].items():
        print(f"- {name}: {stats['mean']:.4f} +/- {stats['std']:.4f}")
    with Path(output_dir, 'summary.json').open('w') as f:
        json.dump(summary, f)
    return summary


def summarize_sets_div_and_cor(output_dirs: Iterable[Union[Path, str]]) -> dict:
    div_avgs, cor_avgs = [], []
    div_stds, cor_stds = [], []
    n_trials = []
    summaries = {}
    for output_dir in output_dirs:
        summary = summarize_trials(output_dir)
        if summary is None:
            continue
        div_avgs.append(summary['result']['div']['mean'])
        cor_avgs.append(summary['result']['cor']['mean'])
        div_stds.append(summary['result']['div']['std'])
        cor_stds.append(summary['result']['cor']['std'])
        n_trials.append(len(summary['completed_trials']))
        summaries[output_dir] = summary
    div_avgs, cor_avgs = map(np.array, (div_avgs, cor_avgs))
    div_stds, cor_stds = map(np.array, (div_stds, cor_stds))
    n_trials = np.array(n_trials)
    avg_div_std = sum(div_stds * n_trials / sum(n_trials))
    avg_cor_std = sum(cor_stds * n_trials / sum(n_trials))
    print(f'Overall statistics of {len(summaries)} sets of trials:')
    print(f'set-average div: {div_avgs.mean():.4f} +/- ({avg_div_std:.4f})')
    print(f'set-average cor: {cor_avgs.mean():.4f} +/- ({avg_cor_std:.4f})')
    return summaries


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dirs', type=str, nargs='+', required=True)
    args = parser.parse_args()
    if len(args.output_dirs) == 1:
        summarize_trials(args.output_dirs[0])
    else:
        summarize_sets_div_and_cor(args.output_dirs)