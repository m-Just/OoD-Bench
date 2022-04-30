import json
import subprocess
import sys
import time
from datetime import timedelta
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from ood_bench import config, utils


class Quantification(utils.Task):
    def __init__(self, args_dict: dict, task_index: int, iteration: int,
                 redirect_stdout_and_stderr: bool) -> None:
        self.args_dict = args_dict
        self.task_index = task_index
        self.iteration = iteration
        self.redirect_stdout_and_stderr = redirect_stdout_and_stderr
    
    def start(self, worker_index: int) -> Tuple[Optional[float], Optional[float]]:
        start_time = time.time()
        output_dir = self.args_dict['feature_dir']
        if self.redirect_stdout_and_stderr:
            logs_dir = Path(output_dir, 'logs_quant')
            logs_dir.mkdir(exist_ok=True)
            log_path = logs_dir.joinpath(f'iteration{self.iteration}.txt')
            stdout = stderr = log_path.open('w')
        else:
            stdout, stderr = sys.stdout, sys.stderr
        subprocess_kwargs = {
            'shell': True, 'check': True, 'stdout': stdout, 'stderr': stderr}
        
        cmd = utils.compose_command('ood_bench.scripts.quantify', self.args_dict, None)
        print('executing command:', cmd, file=stdout, flush=True)
        subprocess.run(cmd, **subprocess_kwargs)
        
        with Path(output_dir, 'result.json').open('r') as f:
            result = json.load(f)
            div, cor = result['div'], result['cor']
            
        time_spent = int(time.time() - start_time)
        time_spent = timedelta(seconds=time_spent)
        print(f'quantification completed in {time_spent}', file=stdout, flush=True)
        return div, cor, self.task_index
    
    def on_start(self):
        print(f'quantification for {self.args_dict["feature_dir"]} started')
        
    def on_completion(self):
        print(f'quantification for {self.args_dict["feature_dir"]} completed')
        
    def on_failure(self):
        print(f'quantification for {self.args_dict["feature_dir"]} failed')


if __name__ == '__main__':
    args, _ = config.parse_argument(Path(__file__).name)
    utils.pprint_dict('Args', vars(args))
    
    data_paths = sorted(Path(args.root_dir).glob('**/data.npz'))
    if data_paths:
        print(f'extracted features found under ({len(data_paths)} trials in total):')
        for path in data_paths:
            print(str(path.parent))
    else:
        print(f'no data found under {args.root_dir}')
        exit(1)
    divs = np.zeros(len(data_paths), dtype=float)
    cors = np.zeros(len(data_paths), dtype=float)
    min_eps_div = args.min_eps_div
    max_eps_div = args.max_eps_div
    min_eps_cor = args.min_eps_cor
    max_eps_cor = args.max_eps_cor
    mode = 'both'
    
    for i in range(args.max_iters):
        eps_div = (min_eps_div + max_eps_div) / 2
        eps_cor = (min_eps_cor + max_eps_cor) / 2
        print(f'iteration #{i+1}: eps_div={eps_div}, eps_cor={eps_cor}')
        
        tasks = []
        for j, path in enumerate(data_paths):
            quant_args = {
                'feature_dir': str(path.parent),
                'eps_div': eps_div,
                'eps_cor': eps_cor,
                'sample_size': args.sample_size,
                'seed': args.seed,
                'mode': mode,
            }
            task = Quantification(quant_args, j, i+1, args.parallel)
            tasks.append(task)
        
        if args.parallel:
            runner = utils.ParallelRunner(args.n_workers, 'thread')
            runner.launch(tasks)
            for completed_future in runner.completed:
                div, cor, j = completed_future.result()
                if mode in ('both', 'div'):
                    divs[j] = div
                if mode in ('both', 'cor'):
                    cors[j] = cor
        else:
            for j, task in enumerate(tasks):
                div, cor, _ = task.start(0)
                if mode in ('both', 'div'):
                    divs[j] = div
                if mode in ('both', 'cor'):
                    cors[j] = cor
            
        avg_div, avg_cor = divs.mean(), cors.mean()
        print(f'iteration #{i+1} result:')
        print(f'eps_div: {eps_div:.4e}, eps_cor: {eps_cor:.4e}')
        print(f'avg_div: {avg_div:.4e}, avg_cor: {avg_cor:.4e}')
        
        lb, ub = args.lower_bound, args.upper_bound
        if min(avg_div, avg_cor) >= lb and max(avg_div, avg_cor) <= ub:
            break
        if avg_div < lb:
            min_eps_div = eps_div
        elif avg_div > ub:
            max_eps_div = eps_div
        else:
            mode = 'cor'
        if avg_cor < lb:
            max_eps_cor = eps_cor
        elif avg_cor > ub:
            min_eps_cor = eps_cor
        else:
            mode = 'div'
    else:
        print('WARNING: maximum number of iterations reached')
    print(f'search ended at iteration {i+1}')
    
    with Path(args.root_dir, 'calibration.json').open('w') as f:
        json.dump({'args': vars(args),
                   'result': {'eps_div': eps_div, 'eps_cor': eps_cor,
                              'avg_div': avg_div, 'avg_cor': avg_cor}}, f)