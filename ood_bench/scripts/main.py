import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import timedelta
from pathlib import Path
from typing import Iterable, Sequence, Tuple

import numpy as np

from ood_bench import config, utils
        
        
class Trial(utils.Task):
    def __init__(self, subprocess_args: Tuple[dict, dict, dict], output_root: str,
                 visible_devices: Sequence[str], skip_quant: bool = False,
                 redirect_stdout_and_stderr: bool = False) -> None:
        self.args_train, self.args_extract, self.args_quantify = subprocess_args
        s = ''.join([str(sorted(d.items())) for d in subprocess_args])
        self.id = hashlib.md5(s.encode('utf-8')).hexdigest()
        self.output_dir = Path(output_root, self.id)
        self.visible_devices = visible_devices
        self.skip_quant = skip_quant
        self.redirect_stdout_and_stderr = redirect_stdout_and_stderr
        
    def start(self, worker_index: int) -> None:
        start_time = time.time()
        devices = self.visible_devices[worker_index]
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.redirect_stdout_and_stderr:
            stdout = stderr = self.output_dir.joinpath('log.txt').open('w')
        else:
            stdout, stderr = sys.stdout, sys.stderr
        subprocess_kwargs = {
            'shell': True, 'check': True, 'stdout': stdout, 'stderr': stderr}
                
        # train the environment classifier
        args_dict = {'model_dir': self.output_dir, **self.args_train}
        cmd = utils.compose_command('ood_bench.scripts.train', args_dict, devices)
        print('executing command:', cmd, file=stdout, flush=True)
        subprocess.run(cmd, **subprocess_kwargs)
        
        # extract the features from the trained classifier
        args_dict = {'model_dir': self.output_dir, **self.args_extract}
        cmd = utils.compose_command('ood_bench.scripts.extract', args_dict, devices)
        print('executing command:', cmd, file=stdout, flush=True)
        subprocess.run(cmd, **subprocess_kwargs)
        
        # quantify the shifts with the extracted features
        if self.skip_quant:
            print('skipping quantification', file=stdout, flush=True)
        else:
            args_dict = {'feature_dir': self.output_dir, **self.args_quantify}
            cmd = utils.compose_command('ood_bench.scripts.quantify', args_dict,
                                        devices)
            print('executing command:', cmd, file=stdout, flush=True)
            subprocess.run(cmd, **subprocess_kwargs)
        
        time_spent = int(time.time() - start_time)
        time_spent = timedelta(seconds=time_spent)
        print(f'Trial completed in {time_spent}', file=stdout, flush=True)
        
    def on_start(self):
        print(f'trial {self.id} started')
        
    def on_completion(self):
        print(f'trial {self.id} completed')
        
    def on_failure(self):
        print(f'trial {self.id} failed')
    
    
def calibrate(args: argparse.Namespace, calib_subarg_names: Iterable[str]) -> None:
    devices = os.environ.get('CUDA_VISIBLE_DEVICES', None)
    args_dict = copy.deepcopy(vars(args))
    p = Path(args_dict['output_dir'])
    p = Path(p.parent, f'{p.name}_iid')
    args_dict.update({
        'output_dir': p,
        'calibrate': False,
        'skip_quant': True,
        'force_iid': True,
    })
    cmd = utils.compose_command('ood_bench.scripts.main', args_dict, devices)
    print('calibration starts')
    print('executing command:', cmd)
    subprocess.run(cmd, shell=True, check=True)
    
    calib_args = {n: getattr(args, n) for n in calib_subarg_names}
    args_dict = {
        'root_dir': p,
        'parallel': args.parallel,
        'n_workers': args.n_workers,
        **calib_args
    }
    cmd = utils.compose_command('ood_bench.scripts.calibrate', args_dict, None)
    print('executing command:', cmd)
    subprocess.run(cmd, shell=True, check=True)
    with p.joinpath('calibration.json').open('r') as f:
        result = json.load(f)['result']
        eps_div, eps_cor = result['eps_div'], result['eps_cor']
    print('calibration completed')
    return eps_div, eps_cor
    

if __name__ == '__main__':
    args, parser = config.parse_argument(Path(__file__).name)
    subarg_name_groups = []
    for group in parser._action_groups:
        if group.title not in ('positional arguments', 'optional arguments'):
            subarg_name_groups.append([action.dest for action in group._group_actions])
    calib_subarg_names = subarg_name_groups[-1]
    subarg_name_groups = subarg_name_groups[:-1]
    
    if args.calibrate:
        args.eps_div, args.eps_cor = calibrate(args, calib_subarg_names)
    
    def compose_subargs(names, seed):
        subargs = {n: getattr(args, n) for n in names}
        subargs['seed'] = seed
        return subargs
    
    random_state = np.random.RandomState(args.seed)
    trials, trial_seeds = [], set()
    while len(trial_seeds) < args.n_trials:
        trial_seed = int(random_state.randint(2**31))
        if trial_seed in trial_seeds:
            continue
        trial_seeds.add(trial_seed)
        subargs = [compose_subargs(names, trial_seed) for names in subarg_name_groups]
        trial = Trial(subargs, args.output_dir, utils.get_visible_cuda_devices(),
                      skip_quant=args.skip_quant,
                      redirect_stdout_and_stderr=args.parallel)
        trials.append(trial)
            
    visible_devices = utils.get_visible_cuda_devices()
    if args.parallel:
        print(f'experiment root directory: {args.output_dir}')
        print(f'#trials to run: {len(trials)}')
        utils.ParallelRunner(len(visible_devices), 'thread').launch(trials)
    else:
        for i, trial in enumerate(trials):
            print(f'trial {i+1}/{len(trials)} starts')
            trial.start(0)
    
    args_dict = {'output_dirs': args.output_dir}
    cmd = utils.compose_command('ood_bench.scripts.summarize', args_dict, None)
    print('executing command:', cmd)
    subprocess.run(cmd, shell=True, check=True)