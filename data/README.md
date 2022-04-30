# Instructions for preparing the datasets
## Download
Most of the datasets, including MNIST (the base of ColoredMNIST), PACS, OfficeHome, Terra Incognita, and WILDSCamelyon, can be downloaded with [this script](https://github.com/m-Just/DomainBed/blob/main/domainbed/scripts/download.py).
Other datasets are also publicly available but need to be downloaded manually.

## Directory structure
Make sure that the directory structure of each dataset is arranged as follows:
#### MNIST
```
MNIST
└── processed
    ├── training.pt
    └── test.pt
```
#### PACS
```
PACS
├── art_painting
├── cartoon
├── photo
└── sketch
```
#### OfficeHome
```
office_home
├── Art
├── Clipart
├── Product
└── Real World
```
#### Terra Incognita
```
terra_incognita
├── location_38
├── location_43
├── location_46
└── location_100
```
#### Camelyon17-WILDS
```
camelyon17_v1.0
├── patches
└── metadata.csv
```
#### DomainNet
```
domain_net
├── clipart
├── infograph
├── painting
├── quickdraw
├── real
└── sketch
```
#### CelebA
```
celeba
├── img_align_celeba
└── blond_split
    ├── tr_env1_df.csv
    ├── tr_env2_df.csv
    └── te_env_df.csv
```
#### NICO
```
NICO
├── animal
├── vehicle
└── mixed_split_corrected
    ├── env_train1.csv
    ├── env_train2.csv
    ├── env_val.csv
    └── env_test.csv
```
#### ImageNet-A
```
imagenet-a
├── n01498041
└── ...
```
#### ImageNet-R
```
imagenet-r
├── n01443537
└── ...
```
#### ImageNet-V2
```
imagenetv2-matched-frequency-format-val
├── n01440764
└── ...
```

For the experiments on the ImageNet variants, we will also need the original ImageNet:
```
ILSVRC
└── Data
    └── CLS-LOC
        └── train
            ├── n01440764
            └── ...
```
Moreover, since ImageNet-A and ImageNet-R contains only 200 classes of the original ImageNet, we also need separate directories for holding the 200-class subsets of ImageNet (symbolic links should also work).
```
imagenet-subset-a200
└── train
    ├── n01498041
    └── ...
```
```
imagenet-subset-r200
└── train
    ├── n01443537
    └── ...
```