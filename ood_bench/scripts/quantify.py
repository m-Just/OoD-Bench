import json
import random
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
import scipy
from scipy.stats import gaussian_kde

from ood_bench import config, utils


def compute_div(p: Sequence[float], q: Sequence[float], probs: Sequence[int],
                eps_div: float) -> float:
    if not len(p) == len(q) == len(probs):
        raise ValueError
    div = 0
    for i in range(len(probs)):
        if p[i] < eps_div or q[i] < eps_div:
            div += abs(p[i] - q[i]) / probs[i]
    div /= len(probs) * 2
    return div


def compute_cor(y_p: np.ndarray, z_p: np.ndarray, y_q: np.ndarray, z_q: np.ndarray,
                p: Sequence[float], q: Sequence[float], probs: Sequence[int],
                points: np.ndarray, eps_cor: float, strict: bool = False) -> float:
    if not len(p) == len(q) == len(probs):
        raise ValueError
    y_p_unique, y_q_unique = map(np.unique, (y_p, y_q))
    if not np.all(y_p_unique == y_q_unique):
        raise ValueError
    classes = sorted(y_p_unique)
    n_classes = len(classes)
    sample_sizes = np.zeros(n_classes, dtype=int)
    cors = np.zeros(n_classes, dtype=float)
    
    for i in range(n_classes):
        y = classes[i]
        indices_p = np.where(y_p == y)[0]
        indices_q = np.where(y_q == y)[0]
        if indices_p.shape != indices_q.shape:
            raise ValueError(f'Number of datapoints mismatch (y={y}): '
                             f'{indices_p.shape} != {indices_q.shape}')
        try:
            kde_p = gaussian_kde(z_p[indices_p].T)
            kde_q = gaussian_kde(z_q[indices_q].T)
            p_given_y = kde_p(points)
            q_given_y = kde_q(points)
        except (np.linalg.LinAlgError, ValueError) as exception:
            if strict:
                raise exception
            print(f'WARNING: skipping y={y} because scipy.stats.gaussian_kde '
                  f'failed. This usually happens when there is too few datapoints.')
            print(f'y={y}: #datapoints=({len(indices_p)}, {len(indices_q)}), '
                  f'skipped')
            continue
        sample_sizes[i] = len(indices_p)
        
        for j in range(len(probs)):
            if p[j] > eps_cor and q[j] > eps_cor:
                integrand = abs(p_given_y[j] * np.sqrt(q[j] / p[j])
                              - q_given_y[j] * np.sqrt(p[j] / q[j]))
                cor_j = integrand / probs[j]
            else:
                integrand = cor_j = 0
            cors[i] += cor_j
        cors[i] /= len(probs) * 2
        print(f'y={y}: #datapoints=({len(indices_p)}, {len(indices_q)}), '
              f'value={cors[i]:.4f}')
    cor = np.sum(sample_sizes * cors) / np.sum(sample_sizes)
    return cor


if __name__ == '__main__':
    args, _ = config.parse_argument(Path(__file__).name)
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    save_dir = Path(args.output_dir or args.feature_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    args_dir = save_dir.joinpath('args')
    args_dir.mkdir(exist_ok=True)
    with args_dir.joinpath('quantify.json').open('w') as f:
        json.dump(vars(args), f)
        
    utils.pprint_dict('Environment', {
        'Python': sys.version.split(' ')[0],
        'NumPy': np.__version__,
        'SciPy': scipy.__version__,
        'scikit-learn': sklearn.__version__,
    })
    utils.pprint_dict('Args', vars(args))
    
    data = np.load(Path(args.feature_dir, 'data.npz'))
    y_p, z_p, y_q, z_q = data['y_p'], data['z_p'], data['y_q'], data['z_q']
    print(f'features loaded: (p) {z_p.shape}, (q) {z_q.shape}')
    print(f'labels   loaded: (p) {y_p.shape}, (q) {y_q.shape}')
    if len(z_p) != len(y_p) or len(z_q) != len(y_q):
        raise RuntimeError

    z_all = np.append(z_p, z_q, 0)
    scaler = StandardScaler().fit(z_all)
    z_all, z_p, z_q = map(scaler.transform, (z_all, z_p, z_q))

    print('computing KDE for importance sampling')
    sampling_pdf = gaussian_kde(z_all.T)
    points = sampling_pdf.resample(args.sample_size, seed=args.seed)
    probs = sampling_pdf(points)

    print('computing KDE for p and q')
    p = gaussian_kde(z_p.T)(points)
    q = gaussian_kde(z_q.T)(points)

    if args.mode in ('both', 'div'):
        print('computing diversity shift')
        div = compute_div(p, q, probs, args.eps_div)
        if np.isnan(div) or np.isinf(div):
            raise RuntimeError
        print(f'div: {div}')
    else:
        div = None

    if args.mode in ('both', 'cor'):
        print('computing correlation shift')
        cor = compute_cor(y_p, z_p, y_q, z_q, p, q, probs, points, args.eps_cor,
                          strict=args.strict)
        if np.isnan(cor) or np.isinf(cor):
            raise RuntimeError
        print(f'cor: {cor}')
    else:
        cor = None
        
    with save_dir.joinpath('result.json').open('w') as f:
        json.dump({'div': div, 'cor': cor}, f)