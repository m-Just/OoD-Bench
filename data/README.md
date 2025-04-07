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

With ImageNet-A, ImageNet-R, and the original ImageNet (ILSVRC) in place,
you can run the following script at `OoD-Bench/data` to create the subset folders:
```python
from pathlib import Path
import subprocess
import os

def create_imagenet_subset(x):
    class_dirs = [d.name for d in Path(f'imagenet-{x}').glob('n*')]
    subset_dir = Path(f'imagenet-subset-{x}200/train')
    subset_dir.mkdir(parents=True, exist_ok=True)
    for class_dir in class_dirs:
        cmd = f'ln -s $(pwd)/ILSVRC/Data/CLS-LOC/train/{class_dir} {subset_dir}/{class_dir}'
        print(f'running: {cmd}')
        subprocess.run(cmd, shell=True, check=True)
    print(f'{len(class_dirs)} created under {subset_dir}')

create_imagenet_subset('a')
create_imagenet_subset('r')
```

#### ImageNet-V2(-Super400)
```
imagenetv2-matched-frequency-format-val
├── n01440764
└── ...
```
**Important**: the imagenetv2 dataset may be initially organized as
```
0/ 1/ 2/ ..... 999/
```
They need to be converted into
```
n01440764/ n01443537/ ...  n15075141/
```
i.e., the original ImageNet indices, as mentioned [here](https://github.com/modestyachts/ImageNetV2/issues/4).

With ImageNet-V2 and the original ImageNet (ILSVRC) in place, you can run the following script at `OoD-Bench/data` to rename the folders:
```python
from pathlib import Path

class_names = [d.name for d in Path('ILSVRC/Data/CLS-LOC/train').glob('*')]
class_names.sort()

v2_class_dirs = {int(d.name): d for d in Path('imagenetv2-matched-frequency-format-val').glob('*')}
for i in range(len(class_names)):
    old_dir = v2_class_dirs[i]
    new_dir = Path.joinpath(v2_class_dirs[i].parent, class_names[i])
    print(old_dir, '->', new_dir)
    old_dir.rename(new_dir)
```