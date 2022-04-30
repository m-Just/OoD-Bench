import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.utils.data as data
import torchvision
import PIL

from domainbed.datasets import get_dataset_class

from ood_bench import config, utils
from ood_bench.networks import get_backbone, EnvClassifier


class Namespace_(argparse.Namespace):
    def __init__(self, d: dict) -> None:
        self.__dict__.update(d)

        
if __name__ == '__main__':
    args, _ = config.parse_argument(Path(__file__).name)
    
    if args.model_dir:
        path = Path(args.model_dir, 'args', 'train.json')
        if path.exists():
            with path.open('r') as f:
                args_train = json.load(f)
            args_train = {k: args_train.get(k, None) for k in vars(args)}
            args_train.update({k: v for k, v in vars(args).items() if v is not None
                               and v is not False})
            args = Namespace_(args_train)
        else:
            raise FileNotFoundError
    
    save_dir = Path(args.output_dir or args.model_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    args_dir = save_dir.joinpath('args')
    args_dir.mkdir(exist_ok=True)
    with args_dir.joinpath('extract.json').open('w') as f:
        json.dump(vars(args), f)
    
    utils.pprint_dict('Environment', {
        'Python': sys.version.split(' ')[0],
        'PyTorch': torch.__version__,
        'Torchvision': torchvision.__version__,
        'CUDA': torch.version.cuda,
        'CUDNN': torch.backends.cudnn.version(),
        'NumPy': np.__version__,
        'PIL': PIL.__version__,
    })
    utils.pprint_dict('Args in effect', vars(args))
    
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    utils.pprint_device(device)
    utils.set_deterministic(seed=args.seed)
    
    if args.dataset_settings:
        dataset_settings = json.loads(args.dataset_settings)
    else:
        dataset_settings = {}
    dataset_settings['data_augmentation'] = False
    utils.pprint_dict('Dataset settings in effect', dataset_settings)

    dataset_class = get_dataset_class(args.dataset)
    dataset = dataset_class(args.data_dir, [], dataset_settings)
    dataloaders = utils.get_dataloaders(
        dataset, args.envs_p, args.envs_q, args.holdout_fraction, args.batch_size,
        args.n_workers, args.force_iid, args.seed)
    in_dataloader_p, out_dataloader_p, in_dataloader_q, out_dataloader_q = dataloaders
    
    model = get_backbone(args.backbone, dataset.input_shape, args.pretrained_model_path)
    if args.model_dir is not None:
        model = EnvClassifier(model, dataset.num_classes, args.dim_z, 2)
        model.load_state_dict(torch.load(Path(args.model_dir, 'model.pth')))
        model = model.g
    model = model.to(device)
    model.eval()
    
    save_dict = {}
    for k, dataloader in enumerate([out_dataloader_p, out_dataloader_q]):
        envs_code = ('p', 'q')[k]
        print(f'extracting features from envs {envs_code}')
        y_minibatches = []
        z_minibatches = []
        for i, (x, y) in enumerate(dataloader):
            x = x.to(device)
            with torch.no_grad():
                z = model(x)
            y_minibatches.append(y)
            z_minibatches.append(z.cpu())
        y_cat = torch.cat(y_minibatches)
        z_cat = torch.cat(z_minibatches)
        save_dict[f'y_{envs_code}'] = y_cat.numpy()
        save_dict[f'z_{envs_code}'] = z_cat.numpy()
    np.savez(save_dir.joinpath(f'data.npz'), **save_dict)