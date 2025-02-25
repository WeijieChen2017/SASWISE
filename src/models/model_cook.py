from typing import Dict, Any, List, Optional
import torch
import copy

from .kitchen_setup import ModelHierarchy


class ModelCook:
    """Class for modifying models according to a menu of instructions."""
    
    def __init__(self, kitchen: Dict[str, Any]):
        self.original_model = kitchen['model']
        self.hierarchy = kitchen['hierarchy']
        self.model = copy.deepcopy(self.original_model)
    
    def apply_menu(self, menu: Dict[str, Any]) -> torch.nn.Module:
        """
        Apply a menu of modifications to the model.
        
        Args:
            menu: Dictionary of modifications to apply, such as:
                {
                    'freeze_layers': ['encoder.layer.0', 'encoder.layer.1'],
                    'dropout_rates': {'decoder': 0.2},
                    'activation_functions': {'encoder': 'gelu'},
                    'ensemble_config': {...}
                }
        
        Returns:
            Modified model
        """
        if 'freeze_layers' in menu:
            self._freeze_layers(menu['freeze_layers'])
        
        if 'dropout_rates' in menu:
            self._set_dropout_rates(menu['dropout_rates'])
        
        if 'activation_functions' in menu:
            self._set_activation_functions(menu['activation_functions'])
        
        if 'ensemble_config' in menu:
            self._setup_ensemble(menu['ensemble_config'])
        
        return self.model
    
    def _freeze_layers(self, layer_names: List[str]):
        """Freeze specified layers in the model."""
        for name, param in self.model.named_parameters():
            should_freeze = any(name.startswith(layer) for layer in layer_names)
            if should_freeze:
                param.requires_grad = False
    
    def _set_dropout_rates(self, dropout_config: Dict[str, float]):
        """Set dropout rates for specified parts of the model."""
        for module_path, rate in dropout_config.items():
            module = self._get_module(module_path)
            for m in module.modules():
                if isinstance(m, torch.nn.Dropout):
                    m.p = rate
    
    def _set_activation_functions(self, activation_config: Dict[str, str]):
        """Change activation functions in specified parts of the model."""
        activation_map = {
            'relu': torch.nn.ReLU(),
            'gelu': torch.nn.GELU(),
            'tanh': torch.nn.Tanh(),
            'sigmoid': torch.nn.Sigmoid()
        }
        
        for module_path, activation_name in activation_config.items():
            if activation_name not in activation_map:
                raise ValueError(f"Unsupported activation function: {activation_name}")
            
            module = self._get_module(module_path)
            self._replace_activations(module, activation_map[activation_name])
    
    def _setup_ensemble(self, ensemble_config: Dict[str, Any]):
        """Setup model ensembling according to configuration."""
        # Implementation depends on specific ensembling requirements
        raise NotImplementedError("Ensemble setup not implemented yet")
    
    def _get_module(self, module_path: str) -> torch.nn.Module:
        """Get a module by its path in the model."""
        if not module_path:
            return self.model
            
        current = self.model
        for part in module_path.split('.'):
            current = getattr(current, part)
        return current
    
    def _replace_activations(self, module: torch.nn.Module, new_activation: torch.nn.Module):
        """Replace activation functions in a module."""
        for name, child in module.named_children():
            if isinstance(child, (torch.nn.ReLU, torch.nn.GELU, torch.nn.Tanh, torch.nn.Sigmoid)):
                setattr(module, name, new_activation)
            else:
                self._replace_activations(child, new_activation)


def model_cook(menu: Dict[str, Any], kitchen: Dict[str, Any]) -> torch.nn.Module:
    """
    Create a cooked model based on the menu and kitchen.
    
    Args:
        menu: Dictionary of modifications to apply
        kitchen: Dictionary containing model and hierarchy information
    
    Returns:
        Modified model ready for fine-tuning
    """
    cook = ModelCook(kitchen)
    return cook.apply_menu(menu) 