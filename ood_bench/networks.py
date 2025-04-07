from typing import Tuple

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.feature_extraction import create_feature_extractor
from torch import Tensor


BACKBONES = [
    'mlp',
    'resnet-18',
    'resnet-34',
    'resnet-50',
    'resnet-101',
    'resnet-152',
]


def get_backbone(name: str, input_shape: Tuple[int, int, int],
                pretrained_model_path: str) -> nn.Module:
    if name == 'mlp':
        backbone = MNIST_MLP(input_shape, pretrained_model_path=pretrained_model_path)
    elif name.startswith('resnet'):
        _, depth = name.split('-')
        backbone = ResNet(depth, pretrained_model_path=pretrained_model_path)
    else:
        raise NotImplementedError
    return backbone


class Backbone(nn.Module):
    def __init__(self, pretrained_model_path: str = None) -> None:
        super(Backbone, self).__init__()
        self.pretrained_model_path = pretrained_model_path
        
        self._load_modules()
        if pretrained_model_path and pretrained_model_path != 'auto':
            state_dict = torch.load(pretrained_model_path)
            self.modules_.load_state_dict(state_dict, strict=False)
            
    @property
    def hdim(self) -> int:
        raise NotImplementedError
        
    def _load_modules(self) -> nn.Module:
        raise NotImplementedError
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
        
        
class MNIST_MLP(Backbone):
    def __init__(self, input_shape: Tuple[int, int, int], hdim: int = 390,
                 pretrained_model_path: str = None) -> None:
        self.input_shape = input_shape
        self._hdim = hdim
        super(MNIST_MLP, self).__init__(pretrained_model_path)
            
    @property
    def hdim(self) -> int:
        return self._hdim
    
    def _load_modules(self) -> None:
        if self.pretrained_model_path == 'auto':
            raise ValueError('no available pretrained models of MNIST_MLP')
        input_dim = self.input_shape[0] * self.input_shape[1] * self.input_shape[2]
        self.modules_ = nn.Sequential(
            nn.Linear(input_dim, self.hdim),
            nn.ReLU(True),
            nn.Linear(self.hdim, self.hdim),
            nn.ReLU(True)
        )
        for m in self.modules_:
            if isinstance(m, nn.Linear):
                gain = nn.init.calculate_gain('relu')
                nn.init.xavier_uniform_(m.weight, gain=gain)
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        return self.modules_(x)


class ResNet(Backbone):
    def __init__(self, depth: int, pretrained_model_path: str = None) -> None:
        self.depth = depth
        super(ResNet, self).__init__(pretrained_model_path)
            
    @property
    def hdim(self) -> int:
        return self._hdim
    
    def _load_modules(self) -> None:
        if self.pretrained_model_path == 'auto':
            network = getattr(models, f'resnet{self.depth}')(pretrained=True)
        else:
            network = getattr(models, f'resnet{self.depth}')(pretrained=False)
        self._hdim = network.fc.in_features
        return_nodes = {'avgpool': 'avgpool'}
        self.modules_ = create_feature_extractor(network, return_nodes=return_nodes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.modules_(x)['avgpool'].flatten(1)
    
    
class EnvClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, class_dim: int, logits_dim: int,
                 output_dim: int, freeze_backbone: bool = False) -> None:
        super(EnvClassifier, self).__init__()
        self.class_dim = class_dim
        self.logits_dim = logits_dim
        self.output_dim = output_dim
        self.backbone = backbone
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)
        
        self.fc = nn.Linear(backbone.hdim, logits_dim)
        self.g = nn.Sequential(self.backbone, self.fc)
        
        y_lin = nn.Linear(class_dim, logits_dim)
        self.y_map = nn.Sequential(y_lin)
        self.h = nn.Linear(logits_dim, output_dim)
        
        for m in (self.fc, y_lin, self.h):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
            
    def forward(self, x: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
        z = self.g(x)
        w = self.y_map(y)
        return z, self.h(z * w)