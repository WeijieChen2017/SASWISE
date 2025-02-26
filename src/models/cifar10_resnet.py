import torch
import torch.nn as nn
from torchvision.models import resnet50

class ResNet50ForCIFAR10(nn.Module):
    """
    ResNet50 model adapted for CIFAR-10 dataset.
    This wrapper handles the mismatch between ImageNet and CIFAR-10 classes.
    """
    def __init__(self):
        super(ResNet50ForCIFAR10, self).__init__()
        # Load the base ResNet50 model
        self.base_model = resnet50(pretrained=False)
        
        # Replace the final fully connected layer for 10 classes
        in_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(in_features, 10)
        
    def forward(self, x):
        return self.base_model(x)
    
    def load_state_dict(self, state_dict, strict=False):
        """
        Custom state_dict loader that handles the mismatch between
        ImageNet pretrained weights (1000 classes) and CIFAR-10 (10 classes).
        """
        # Filter out the fc layer weights from the state dict
        filtered_dict = {k: v for k, v in state_dict.items() if not k.startswith('fc.')}
        
        # Load the filtered state dict
        result = self.base_model.load_state_dict(filtered_dict, strict=False)
        
        return result 