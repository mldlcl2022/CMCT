import torch.nn as nn
import timm
import torch

class ImageModel(nn.Module):
    def __init__(
        self,
        model_name: str,
        pretrained: bool,
        num_classes: int,
        dropout: float= 0.2
    ):
        super().__init__()
        
        # image model #
        if model_name == 'convnext':
            model_name = 'convnext_base'
        elif model_name == 'resnet':
            model_name = 'resnet18'
        elif model_name == 'efficientnet':
            model_name = 'efficientnet_b0'
        elif model_name == 'inception':
            model_name = 'inception_v4'
        elif model_name == 'vit':
            model_name = 'vit_base_patch16_224'
        
        if 'inception' in model_name:
            self.model = timm.create_model(
                model_name,
                pretrained= pretrained,
                num_classes= num_classes
            )
        else:
            self.model = timm.create_model(
                model_name,
                pretrained= pretrained,
                num_classes= num_classes,
                drop_path_rate= dropout
            )
    
    def forward(self, x: torch.Tensor):
        task = self.model(x)
        return task