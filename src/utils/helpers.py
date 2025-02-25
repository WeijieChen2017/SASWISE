from typing import Dict, Any
import yaml
from pathlib import Path
import torch


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Dictionary containing configuration parameters
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict[str, Any], save_path: str):
    """
    Save configuration to a YAML file.
    
    Args:
        config: Configuration dictionary
        save_path: Path to save the configuration
    """
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def get_device(device_str: str = None) -> torch.device:
    """
    Get the appropriate device for training.
    
    Args:
        device_str: Optional device specification ('cuda' or 'cpu')
        
    Returns:
        torch.device instance
    """
    if device_str is None:
        device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    return torch.device(device_str)


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def ensure_dir(path: str):
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        path: Directory path
    """
    Path(path).mkdir(parents=True, exist_ok=True)


def format_time(seconds: float) -> str:
    """
    Format time in seconds to a human-readable string.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    
    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0 or hours > 0:
        parts.append(f"{minutes}m")
    parts.append(f"{seconds}s")
    
    return " ".join(parts)


def get_lr(optimizer: torch.optim.Optimizer) -> float:
    """
    Get the current learning rate from an optimizer.
    
    Args:
        optimizer: PyTorch optimizer
        
    Returns:
        Current learning rate
    """
    for param_group in optimizer.param_groups:
        return param_group['lr'] 