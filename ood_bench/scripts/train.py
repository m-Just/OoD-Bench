import json
import sys
import time
from datetime import timedelta
from pathlib import Path
from typing import Iterable, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch.optim import SGD, swa_utils, lr_scheduler
import torchvision
import PIL

from domainbed.datasets import get_dataset_class

from ood_bench import config, utils
from ood_bench.networks import get_backbone, EnvClassifier
            
            
def evaluate(model: nn.Module, zipped_minibatches: Iterable, device: str) -> float:
    model.eval()
    
    n_examples = 0
    n_correct_preds = 0
    for i, minibatches in enumerate(zipped_minibatches):
        x, y = map(torch.cat, zip(*minibatches))
        e = torch.cat([torch.zeros(x.size(0) // 2, dtype=torch.long),
                       torch.ones (x.size(0) // 2, dtype=torch.long)])
        x = x.to(device)
        y = F.one_hot(y, num_classes=model.class_dim).float().to(device)
        e = e.to(device)
        
        with torch.no_grad():
            z, logits = model(x, y)

        _, pred = torch.max(logits.data, 1)
        n_correct_preds += (pred == e).long().sum().item()
        n_examples += x.size(0)
    
    model.train()
    return n_correct_preds / n_examples


def update_swa_bn(model: nn.Module, zipped_minibatches: Iterable, device: str) -> None:
    momenta = {}
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.running_mean = torch.zeros_like(module.running_mean)
            module.running_var = torch.ones_like(module.running_var)
            momenta[module] = module.momentum

    if not momenta:
        return

    was_training = model.training
    model.train()
    for module in momenta.keys():
        module.momentum = None
        module.num_batches_tracked *= 0

    with torch.no_grad():
        for i, minibatches in enumerate(zipped_minibatches):
            x, y = map(torch.cat, zip(*minibatches))
            x = x.to(device)
            y = F.one_hot(y, num_classes=model.class_dim).float().to(device)
            model(x, y)

    for bn_module in momenta.keys():
        bn_module.momentum = momenta[bn_module]
    model.train(was_training)


if __name__ == '__main__':
    args, _ = config.parse_argument(Path(__file__).name)
    
    if args.n_epochs % args.n_epochs_per_ckpt:
        raise ValueError
    
    save_dir = Path(args.model_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    args_dir = save_dir.joinpath('args')
    args_dir.mkdir(exist_ok=True)
    with args_dir.joinpath('train.json').open('w') as f:
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
    utils.pprint_dict('Args', vars(args))
    
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
    dataset_settings.setdefault('data_augmentation', False)
    utils.pprint_dict('Dataset settings in effect', dataset_settings)
    
    dataset_class = get_dataset_class(args.dataset)
    dataset = dataset_class(args.data_dir, [], dataset_settings)
    dataloaders = utils.get_dataloaders(
        dataset, args.envs_p, args.envs_q, args.holdout_fraction, args.batch_size,
        args.n_workers, args.force_iid, args.seed)
    in_dataloader_p, out_dataloader_p, in_dataloader_q, out_dataloader_q = dataloaders
    
    backbone = get_backbone(args.backbone, dataset.input_shape,
                            args.pretrained_model_path)
    model = EnvClassifier(backbone, dataset.num_classes, args.dim_z, 2, args.freeze_backbone)
    model = model.to(device)

    best_dict = {'acc': 0, 'step': -1, 'state': None}
    val_acc = evaluate(model, zip(out_dataloader_p, out_dataloader_q), device)
    print(f'validation accuracy before training: {val_acc:.4f}')
    
    optimizer = SGD(model.parameters(), lr=args.init_lr, momentum=args.momentum,
                    weight_decay=args.weight_decay)
    swa_model = swa_utils.AveragedModel(model)
    if not hasattr(swa_model, 'class_dim'):
        setattr(swa_model, 'class_dim', model.class_dim)
    else:
        raise Exception
    T_max = len(in_dataloader_p) * args.swa_start
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max,
                                               eta_min=args.swa_lr)
    
    training_loss = utils.ExponentialMovingAverage(0.1)
    training_acc = utils.ExponentialMovingAverage(0.1)
    for epoch in range(1, args.n_epochs + 1):
        epoch_start_time = time.time()
        for i, minibatches in enumerate(zip(in_dataloader_p, in_dataloader_q)):
            model.train()
            x, y = map(torch.cat, zip(*minibatches))
            e = torch.cat([torch.zeros(args.batch_size, dtype=torch.long),
                           torch.ones (args.batch_size, dtype=torch.long)])
            x = x.to(device)
            y = F.one_hot(y, num_classes=model.class_dim).float().to(device)
            e = e.to(device)
            
            z, logits = model(x, y)
            loss = F.cross_entropy(logits, e)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if epoch <= args.swa_start:
                scheduler.step()
            else:
                swa_model.update_parameters(model)
            
            _, pred = torch.max(logits.data, 1)
            correct = (pred == e)
            minibatch_acc = correct.float().sum().item() / correct.size(0)
            training_loss.update(loss.item())
            training_acc.update(minibatch_acc)
        epoch_time_spent = int(time.time() - epoch_start_time)
        epoch_time_spent = timedelta(seconds=epoch_time_spent)
            
        if epoch % args.n_epochs_per_ckpt == 0:
            print(f'epoch {epoch} / {args.n_epochs} finished in {epoch_time_spent}')
            print(f'training loss (ema): {training_loss.ema:.4f}')
            print(f'     accuracy (ema): {training_acc.ema:.4f}')
            val_acc = evaluate(model, zip(out_dataloader_p, out_dataloader_q), device)
            print(f'validation accuracy: {val_acc:.4f}')
            if epoch > args.swa_start:
                update_swa_bn(swa_model, zip(in_dataloader_p, in_dataloader_q), device)
                swa_val_acc = evaluate(swa_model, zip(out_dataloader_p, out_dataloader_q),
                                       device)
                print(f'validation accuracy: {swa_val_acc:.4f} (swa)')
            
    state_dict = swa_model.state_dict()
    del state_dict['n_averaged']
    for name in list(state_dict.keys()):
        if name.startswith('module'):
            state_dict[name[7:]] = state_dict[name]
            del state_dict[name]
    torch.save(state_dict, str(save_dir.joinpath('model.pth')))

    with save_dir.joinpath('result.json').open('w') as f:
        json.dump({
            'training loss': training_loss.ema,
            'training acc': training_acc.ema,
            'validation acc': val_acc,
            'validation acc (swa)': swa_val_acc
        }, f)