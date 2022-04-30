import os
import random
import shlex
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import Any, Callable, DefaultDict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.utils.data as data


class Task:
    def launch(self, worker_index: int):
        raise NotImplementedError
        
    def on_start(self):
        raise NotImplementedError
        
    def on_completion(self):
        raise NotImplementedError
        
    def on_failure(self):
        raise NotImplementedError
    
    
class ParallelRunner:
    def __init__(self, max_workers: int, worker_type: str) -> None:
        self.max_workers = max_workers
        self.worker_type = worker_type
        
    def launch(self, tasks: Sequence[Task], callback: Callable = None) -> None:
        self.num_tasks = len(tasks)
        self.completed, self.failed = [], []
        self.task_futures = [None] * self.max_workers
        self.next = 0
        if self.worker_type == 'thread':
            Executor = ThreadPoolExecutor
        elif self.worker_type == 'process':
            Executor = ProcessPoolExecutor
        else:
            raise NotImplementedError
        self.executor = Executor(max_workers=self.max_workers)
        while len(self.completed) + len(self.failed) < len(tasks):
            for worker in self.idle_workers:
                self.check_completion(worker)
                if self.next == len(tasks):
                    continue
                self.assign(worker, tasks[self.next], callback)
                self.next += 1
            time.sleep(1)
        self.executor.shutdown()
        
    def check_completion(self, worker) -> None:
        if not self.task_futures[worker]:
            return
        task, future = self.task_futures[worker]
        if future.exception():
            task.on_failure()
            self.failed.append(future)
        else:
            task.on_completion()
            self.completed.append(future)
        self.task_futures[worker] = None
        print(f'{len(self.completed)} completed, {len(self.failed)} failed, '
              f'{self.num_tasks - len(self.completed) - len(self.failed)} to go')
        
    def assign(self, worker: int, task: Task, callback: Callable) -> None:
        future = self.executor.submit(task.start, worker)
        task.on_start()
        if callback:
            future.add_done_callback(callback)
        self.task_futures[worker] = (task, future)
        
    @property
    def idle_workers(self) -> List[int]:
        idle_workers = []
        for i in range(self.max_workers):
            if self.task_futures[i]:
                task, future = self.task_futures[i]
                if future.done():
                    idle_workers.append(i)
            else:
                idle_workers.append(i)
        return idle_workers


class ExponentialMovingAverage:
    def __init__(self, update_ratio: float, init_val: Optional[float] = None) -> None:
        if not(0 < update_ratio < 1):
            raise ValueError
        self._update_ratio = update_ratio
        self.ema = init_val
        
    def update(self, value: float) -> float:
        if self.ema is None:
            self.ema = value
        else:
            self.ema *= (1 - self._update_ratio)
            self.ema += self._update_ratio * value


class ConcatDataset_(data.ConcatDataset):
    def __init__(self, datasets: Iterable[data.Dataset]) -> None:
        super(ConcatDataset_, self).__init__(datasets)
        self.samples = sum([dataset.samples for dataset in datasets], [])
        
        
class Subset_(data.Subset):
    def __init__(self, dataset: data.Dataset, indices: Sequence[int]) -> None:
        super(Subset_, self).__init__(dataset, indices)
        self.samples = [dataset.samples[i] for i in indices]
        
        
class WrappedTensorDataset(data.Dataset):
    def __init__(self, dataset: data.TensorDataset) -> None:
        self.samples = [(x.numpy(), y.item()) for x, y in zip(*dataset.tensors)]
        
    def __getitem__(self, key: int) -> Tuple[Any, Any]:
        x, y = self.samples[key]
        return x, y
    
    def __len__(self) -> int:
        return len(self.samples)
    

def get_label_map(dataset: data.Dataset) -> DefaultDict[Any, List[int]]:
    label_maps = defaultdict(list)
    for i, (_, label) in enumerate(dataset.samples):
        label_maps[label].append(i)
    return label_maps


def create_label_shift_free_datasets(datasets: Iterable[data.Dataset],
                                     seed: int) -> Tuple[Subset_, ...]:
    label_maps = [get_label_map(dataset) for dataset in datasets]
    if not label_maps:
        raise RuntimeError
    labels = set(label_maps[0])
    
    min_count = {}
    for label in labels:
        min_count[label] = min(len(map_[label]) for map_ in label_maps)
    
    keys_list = [[] for _ in range(len(label_maps))]
    rng = np.random.RandomState(seed)
    for label, count in min_count.items():
        for i, map_ in enumerate(label_maps):
            rng.shuffle(map_[label])
            keys_list[i] += map_[label][:count]
    return tuple(Subset_(d, k) for d, k in zip(datasets, keys_list))


def split_dataset(dataset: data.Dataset, fraction: float,
                  seed: int) -> Tuple[Subset_, Subset_]:
    """ Randomly split a dataset with deterministic label distribution. """
    label_map = get_label_map(dataset)
    rng = np.random.RandomState(seed)
    keys_p = []
    keys_q = []
    for _, keys in label_map.items():
        n = int(len(keys) * fraction)
        rng.shuffle(keys)
        keys_p += keys[:n]
        keys_q += keys[n:]
    return Subset_(dataset, keys_p), Subset_(dataset, keys_q)
    
    
def get_dataloaders(dataset: data.Dataset, envs_p: Iterable[int], envs_q: Iterable[int],
                    holdout_fraction: float, batch_size: int, n_workers: int,
                    force_iid: bool, seed: int) -> Tuple[data.DataLoader, ...]:
    for i, env in enumerate(dataset):
        if isinstance(env, data.TensorDataset):
            dataset.datasets[i] = WrappedTensorDataset(env)
    
    dataset_p = ConcatDataset_([dataset[i] for i in envs_p])
    dataset_q = ConcatDataset_([dataset[i] for i in envs_q])
    subset_p, subset_q = create_label_shift_free_datasets([dataset_p, dataset_q], seed)
    if force_iid:
        subsets_p = split_dataset(subset_p, 0.5, seed)
        subsets_q = split_dataset(subset_q, 0.5, seed)
        subset_p, subset_q = map(ConcatDataset_, zip(subsets_p, subsets_q[::-1]))
    
    def in_out_split(dataset: data.Dataset) -> Tuple[data.DataLoader, data.DataLoader]:
        out, in_ = split_dataset(dataset, holdout_fraction, seed)
        in_dataloader  = data.DataLoader(in_, batch_size, shuffle=True,
                                         num_workers=n_workers, drop_last=True)
        out_dataloader = data.DataLoader(out, batch_size, shuffle=False,
                                         num_workers=n_workers)
        return in_dataloader, out_dataloader
        
    in_dataloader_p, out_dataloader_p = in_out_split(subset_p)
    in_dataloader_q, out_dataloader_q = in_out_split(subset_q)
    
    print(f'Envs p: #datapoints (in+out): {len(subset_p)} '
          f'({len(in_dataloader_p.dataset)}+{len(out_dataloader_p.dataset)})')
    print(f'Envs q: #datapoints (in+out): {len(subset_q)} '
          f'({len(in_dataloader_q.dataset)}+{len(out_dataloader_q.dataset)})')
    
    return in_dataloader_p, out_dataloader_p, in_dataloader_q, out_dataloader_q


def pprint_dict(title: str, d: dict) -> None:
    print(f'{title}:')
    for k, v in d.items():
        print(f'\t{k}: {v}')
        
    
def pprint_device(device: str) -> None:
    d = {'name': device}
    if device == 'cuda':
        default = ','.join(map(str, range(torch.cuda.device_count())))
        d['CUDA_VISIBLE_DEVICES'] = os.environ.get('CUDA_VISIBLE_DEVICES', default)
    pprint_dict('Device', d)
    
    
def get_visible_cuda_devices() -> List[str]:
    devices = os.environ.get('CUDA_VISIBLE_DEVICES', None)
    if devices:
        devices = devices.split(',')
    else:
        devices = [str(i) for i in range(torch.cuda.device_count())]
    return devices
    
    
def set_deterministic(seed: int = 0):
    ''' Reference: https://pytorch.org/docs/stable/notes/randomness.html '''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.use_deterministic_algorithms(True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        
        
def compose_command(module_name: str, args_dict: dict, devices: str) -> str:
    formatted_args_dict = {}
    for k, v in args_dict.items():
        if v is None or v is False:
            continue
        elif v is True:
            v = ''
        elif isinstance(v, list):
            v = ' '.join([str(v_) for v_ in v])
        elif isinstance(v, str):
            v = shlex.quote(v)
        formatted_args_dict[k] = v
    cmd = ' '.join([f'--{k} {v}' if v != '' else f'--{k}'
                    for k, v in formatted_args_dict.items()])
    cmd = f'python -m {module_name} {cmd}'
    if devices is not None:
        cmd = f'CUDA_VISIBLE_DEVICES={devices} {cmd}'
    return cmd