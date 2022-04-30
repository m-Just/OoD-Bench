# OoD-Bench
**OoD-Bench** is a benchmark for both datasets and algorithms of out-of-distribution generalization.
It positions datasets along two dimensions of distribution shift: *diversity shift* and *correlation shift*, unifying the disjoint threads of research from the perspective of data distribution.
OoD algorithms are then evaluated and compared on two groups of datasets, each dominanted by one kind of the distribution shift.
See [our paper](https://arxiv.org/abs/2106.03721) (to appear in CVPR 2022 for **oral** presentation) for more details.

This repository contains the code to produce the benchmark, which has two main components:
- a framework for quantifying distribution shift that benchmarks the datasets, and
- a modified version of [DomainBed](https://github.com/facebookresearch/DomainBed) that benchmarks the algorithms.

## Environment requirements
- Python 3.6 or above
- The packages listed in `requirements.txt`. You can install them via `pip install -r requirements.txt`. Package `torch_scatter` may require a [manual installation](https://github.com/rusty1s/pytorch_scatter#installation)
- Submodules are added to the path:
```bash
export PYTHONPATH="$PYTHONPATH:$(pwd)/external/DomainBed/"
export PYTHONPATH="$PYTHONPATH:$(pwd)/external/wilds/"
```

## Data preparation
Please follow [this instruction](data/README.md).

## Quantifying diversity and correlation shift
The quantification process consists of three main steps:
(1) training an environment classifier,
(2) extracting features from the trained classifier, and
(3) measuring the shifts with the extracted features.
The module `ood_bench.scripts.main` will handle the whole process for you.
For example, to quantify the distribution shift between the training environments (indexed by 0 and 1) and the test environment (indexed by 2) of [Colored MNIST](https://github.com/facebookresearch/InvariantRiskMinimization/blob/fc185d0f828a98f57030ba3647efc7394d1be95a/code/colored_mnist/main.py#L34) with 16 trials, you can simply run:
```bash
python -m ood_bench.scripts.main\
       --n_trials 16\
       --data_dir /path/to/my/data\
       --dataset ColoredMNIST_IRM\
       --envs_p 0 1\
       --envs_q 2\
       --backbone mlp\
       --output_dir /path/to/store/outputs
```
In other cases where pretrained models are used, `--pretrained_model_path` must be specified.
For models in [torchvision model zoo](https://pytorch.org/vision/stable/models.html), you can pass `auto` to the argument and the pretrained model will be downloaded automatically.

These two optional arguments are also useful:
- `--parallel`: utilize multiple GPUs to conduct the trials in parallel. The maximum number of parallel trials is the number of visible GPUs which can be controlled by setting `CUDA_VISIBLE_DEVICES`.
- `--calibrate`: calibrate the thresholds `eps_div` and `eps_cor` so that the estimated diversity and correlation shift are ensured to be within a range close to 0 under i.i.d. condition.

For more quantification examples on other datasets, see [`ood_bench/examples`](ood_bench/examples).
Note that there will be some difference between the results produced by this implementation and those reported in our paper because we reworked the original implementation to ease public use and to improve quantification stability.
One of the main improvements is the use of calibration whereas previously the same thresholds that are empirically sound are used across all the datasets studied in our paper (but this may not hold for other datasets).

### Extend OoD-Bench

#### Experiment with other datasets
New datasets must first be added to `external/DomainBed/domainbed/datasets.py` as a subclass of `MultipleDomainDataset`, for example:
```python
class MyDataset(MultipleDomainDataset):
    ENVIRONMENTS = ['env0', 'env1']        # at least two environments
    def __init__(self, root, test_envs, hparams):
        super().__init__()

        # you may change the transformations below
        transform = get_transform()
        augment_scheme = hparams.get('data_augmentation_scheme', 'default')
        augment_transform = get_augment_transform(augment_scheme)

        self.datasets = []                 # required
        for i, env_name in enumerate(self.ENVIRONMENTS):
            if hparams['data_augmentation'] and (i not in test_envs):
                env_transform = augment_transform
            else:
                env_transform = transform
            # load the environments, not necessarily as ImageFolders;
            # you may write a specialized class to load them; the class
            # must possess an attribute named `samples`, a sequence of
            # 2-tuples where the second elements are the labels
            dataset = ImageFolder(Path(root, env_name), transform=env_transform)
            self.datasets.append(dataset)

        self.input_shape = (3, 224, 224,)  # required
        self.num_classes = 2               # required
```

#### Experiment with other backbones
New network backbones must be first added to `ood_bench/networks.py` as a subclass of `Backbone`, for example:
```python
class MyBackbone(Backbone):
    def __init__(self, hdim, pretrained_model_path=None):
        self._hdim = hdim
        super(MyBackbone, self).__init__(pretrained_model_path)

    @property
    def hdim(self):
        return self._hdim

    def _load_modules(self):
        self.modules_ = nn.Sequential(
            nn.Linear(3 * 14 * 14, self.hdim),
            nn.ReLU(True),
        )

    def forward(self, x):
        return self.modules_(x)
```

## Benchmarking OoD algorithms
Please refer to [this repository](https://github.com/m-Just/DomainBed?organization=m-Just&organization=m-Just).

## Citing
If you find the code useful or find our paper relevant to your research, please consider citing:
```
@inproceedings{ye2022ood,
    title={OoD-Bench: Quantifying and Understanding Two Dimensions of Out-of-Distribution Generalization},
    author={Ye, Nanyang and Li, Kaican and Bai, Haoyue and Yu, Runpeng and Hong, Lanqing and Zhou, Fengwei and Li, Zhenguo and Zhu, Jun},
    booktitle={CVPR},
    year={2022}
}
```
