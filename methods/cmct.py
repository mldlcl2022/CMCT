import torch.nn as nn
import timm
import torch

class CMCT(nn.Module):
    def __init__(
        self,
        encoder_name: str,
        num_concepts: int,
        num_labels: int,
        dropout: float= 0.2
    ):
        super().__init__()
        
        # backbone encoder
        if encoder_name == 'convnext':
            encoder_name = 'convnext_base'
        elif encoder_name == 'inception':
            encoder_name = 'inception_v4'
        elif encoder_name == 'resnet':
            encoder_name = 'resnet18'
        elif encoder_name == 'efficientnet':
            encoder_name = 'efficientnet_b0'
        elif encoder_name == 'vit':
            encoder_name = 'vit_base_patch16_224'
        
        if 'inception' in encoder_name:
            self.encoder = timm.create_model(
                encoder_name,
                pretrained= True,
                num_classes= 0
            )
        else:
            self.encoder = timm.create_model(
                encoder_name,
                pretrained= True,
                num_classes= 0,
                drop_path_rate= dropout
            )
        
        # concept layer
        self.concept_layer = nn.Sequential(
            nn.Linear(getattr(self.encoder, 'num_features'), num_concepts)
        )
        
        # task layer
        self.task_layer = nn.Sequential(
            nn.LayerNorm(num_concepts),
            nn.Dropout(dropout),
            nn.Linear(num_concepts, num_labels)
        )
        
        # dropout
        self.dropout = nn.Dropout(dropout)
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.concept_layer.weight, gain= 0.5)
        nn.init.zeros_(self.concept_layer.bias)
        nn.init.xavier_uniform_(self.task_layer.weight, gain= 0.5)
        nn.init.zeros_(self.task_layer.bias)
    
    def forward(self, x: torch.Tensor):
        x = self.encoder(x)
        x = self.dropout(x)
        
        concept_pred = self.concept_layer(x)
        task_pred = self.task_layer(concept_pred)
        
        return concept_pred, task_pred