import torch
from typing import Optional, Dict, Any


def load_pretrained_model(
    model_path: str,
    model_type: str,
    device: Optional[str] = None
) -> torch.nn.Module:
    """
    Load a pretrained model from the given path.
    
    Args:
        model_path: Path to the pretrained model weights
        model_type: Type of model to load (e.g., 'transformer', 'resnet', etc.)
        device: Device to load the model on ('cuda' or 'cpu')
        
    Returns:
        Loaded model instance
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load state dict
    state_dict = torch.load(model_path, map_location=device)
    
    # Initialize appropriate model based on type
    model = _get_model_class(model_type)
    
    # Load weights
    model.load_state_dict(state_dict)
    model.to(device)
    
    return model


def _get_model_class(model_type: str) -> torch.nn.Module:
    """
    Get the appropriate model class based on the model type.
    
    Args:
        model_type: Type of model to instantiate
        
    Returns:
        Model class
    """
    # Add more model types as needed
    model_classes = {
        'transformer': torch.nn.Transformer,
        # Add other model types here
    }
    
    if model_type not in model_classes:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    return model_classes[model_type]()


def get_model_info(model: torch.nn.Module) -> Dict[str, Any]:
    """
    Get information about the model's architecture and parameters.
    
    Args:
        model: The model to analyze
        
    Returns:
        Dictionary containing model information
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'layers': [name for name, _ in model.named_modules()],
        'device': next(model.parameters()).device
    } 