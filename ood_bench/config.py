import argparse
from typing import Union

from domainbed.datasets import DATASETS
from ood_bench.networks import BACKBONES


DEFAULT_SEED = 0
DEFAULT_SAMPLE_SIZE = 10000
ParserOrGroup = Union[argparse.ArgumentParser, argparse._ArgumentGroup]


def add_main_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED, help='main seed to '
                        'generate trial seeds')
    parser.add_argument('--n_trials', type=int, default=10, help='number of trials '
                        'with different random seeds to run')
    parser.add_argument('--parallel', action='store_true', help='run trials in '
                        'parallel making use all visible GPUs (one trial per GPU)')
    parser.add_argument('--output_dir', type=str, default='outputs/my_trials',
                        help='root directory to store the outputs of all trials')
    parser.add_argument('--calibrate', action='store_true', help='calibrate EPS_DIV '
                        'and EPS_COR to ensure that quantification under iid setting '
                        'yields near-zero result')
    parser.add_argument('--skip_quant', action='store_true', help='skip quantification')
    
    group = parser.add_argument_group('training arguments')
    add_training_arguments(group, export=True)
    group = parser.add_argument_group('feature extraction arguments')
    add_feature_extr_arguments(group, export=True)
    group = parser.add_argument_group('shift quantification arguments')
    add_quant_arguments(group, export=True)
    group = parser.add_argument_group('calibration arguments')
    add_calib_arguments(group, export=True)
    
    
def add_training_arguments(parser: ParserOrGroup, export: bool = False) -> None:
    if not export:
        parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
        parser.add_argument('--model_dir', type=str, required=True, help='directory '
                            'to store training outputs')
        
    parser.add_argument('--data_dir', type=str, required=True, help='root directory '
                        'storing all datasets')
    parser.add_argument('--dataset', type=str, required=True, choices=DATASETS)
    parser.add_argument('--envs_p', type=int, nargs='+', required=True,
                        help='indices of a set of environments')
    parser.add_argument('--envs_q', type=int, nargs='+', required=True,
                        help='indices of another set of environments')
    parser.add_argument('--holdout_fraction', type=float, default=0.2, help='fraction '
                        'of data used for validation')
    parser.add_argument('--dataset_settings', type=str, help='JSON-serialized dict')
    parser.add_argument('--n_workers', type=int, default=4)
    parser.add_argument('--force_iid', action='store_true', help='merge environments p '
                        'and q; then randomly split the merged dataset')

    parser.add_argument('--backbone', type=str, required=True, choices=BACKBONES)
    parser.add_argument('--pretrained_model_path', type=str,
                        help='load the pretrained model at the given path; or '
                             'load the pretrained model from torchvision model '
                             'zoo (if available) by passing "auto"')
    parser.add_argument('--dim_z', type=int, default=8, help='dimension of the feature '
                        'space in which feature densities are estimated')
    
    # Optimization hyperparameters
    # Environment classifiers are trained with SGD and Stochastic Weight Averaging (SWA)
    # which prevents the classifiers from ending up at bad local minima, especially
    # when the environments are i.i.d.
    parser.add_argument('--init_lr', type=float, default=0.05, help='initial SGD '
                        'learning rate before SWA starts')
    parser.add_argument('--swa_lr', type=float, default=0.025, help='SWA learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='ratio of '
                        'weight decay')
    parser.add_argument('--batch_size', type=int, default=32, help='size of minibatch '
                        'from either set of environments')
    
    parser.add_argument('--swa_start', type=int, default=5, help='first epoch of SWA')
    parser.add_argument('--n_epochs', type=int, default=10, help='total # of epochs')
    parser.add_argument('--n_epochs_per_ckpt', type=int, default=1, help='# of epochs '
                        'between evaluations')
    
    
def add_feature_extr_arguments(parser: ParserOrGroup, export: bool = False) -> None:
    if not export:
        # specify `model_dir` to restore the arguments used earlier during training
        # if you want to use models trained elsewhere, omit this argument
        parser.add_argument('--model_dir', type=str,
                            help='reuse the training arguments stored under the '
                                 'given directory as the default arguments')

        # if `model_dir` is omitted, you must specify the following arguments;
        # if `model_dir` is given, you can optionally provide them to selectively
        # override the training arguments
        parser.add_argument('--seed', type=int)

        parser.add_argument('--data_dir', type=str)
        parser.add_argument('--dataset', type=str, choices=DATASETS)
        parser.add_argument('--envs_p', type=int, nargs='+')
        parser.add_argument('--envs_q', type=int, nargs='+')
        parser.add_argument('--holdout_fraction', type=float)
        parser.add_argument('--dataset_settings', type=str)
        parser.add_argument('--n_workers', type=int)
        parser.add_argument('--force_iid', action='store_true')

        parser.add_argument('--backbone', type=str, choices=BACKBONES)
        parser.add_argument('--pretrained_model_path', type=str)
        parser.add_argument('--dim_z', type=int)

        parser.add_argument('--batch_size', type=int)
        parser.add_argument('--output_dir', type=str, help='redirect the output to the '
                            'specified directory from MODEL_DIR')
        
        
def add_quant_arguments(parser: ParserOrGroup, export: bool = False) -> None:
    if not export:
        parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
        parser.add_argument('--feature_dir', type=str, help='path to the directory '
                            'storing the extracted features and corresponding labels')
        parser.add_argument('--output_dir', type=str, help='redirect the output to the '
                            'specified directory from FEATURE_DIR')
        
    parser.add_argument('--mode', type=str, choices=['both', 'div', 'cor'],
                        default='both', help='quantify both kinds of shift or either '
                       'one of them')
    parser.add_argument('--eps_div', type=float, default=1e-12,
                        help='epsilon threshold for estimating diversity shift')
    parser.add_argument('--eps_cor', type=float, default=5e-4,
                        help='epsilon threshold for estimating correlation shift')
    parser.add_argument('--sample_size', type=int, default=DEFAULT_SAMPLE_SIZE,
                        help='sample size of importance sampling')
    parser.add_argument('--strict', action='store_true', help='raise error when KDE '
                        'fails for insufficient sample')

    
def add_calib_arguments(parser: ParserOrGroup, export: bool = False) -> None:
    if not export:
        parser.add_argument('--root_dir', type=str, required=True)
        parser.add_argument('--parallel', action='store_true')
        parser.add_argument('--n_workers', type=int)
        parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
        parser.add_argument('--sample_size', type=int, default=DEFAULT_SAMPLE_SIZE,
                            help='sample size of importance sampling')
    
    parser.add_argument('--max_iters', type=int, default=20)
    parser.add_argument('--upper_bound', type=float, default=1e-2)
    parser.add_argument('--lower_bound', type=float, default=5e-3)
    parser.add_argument('--min_eps_div', type=float, default=0)
    parser.add_argument('--max_eps_div', type=float, default=1e-6)
    parser.add_argument('--min_eps_cor', type=float, default=0)
    parser.add_argument('--max_eps_cor', type=float, default=1)


def add_arguments(parser: argparse.ArgumentParser, script_name: str) -> None:
    if script_name == 'main.py':
        add_main_arguments(parser)
    elif script_name == 'train.py':
        add_training_arguments(parser)
    elif script_name == 'extract.py':
        add_feature_extr_arguments(parser)
    elif script_name == 'quantify.py':
        add_quant_arguments(parser)
    elif script_name == 'calibrate.py':
        add_calib_arguments(parser)
    else:
        raise ValueError


def parse_argument(script_name: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_arguments(parser, script_name)
    args = parser.parse_args()
    return args, parser