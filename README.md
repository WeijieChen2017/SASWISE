# SASWISE: Serving-Aware Swappable Weights for Intelligent Systems Engineering

SASWISE is a framework for training and testing models with swappable components (servings) organized into courses. This approach allows for modular model development and evaluation.

## Project Structure

```
SASWISE/
├── experiment/
│   └── templates/           # Templates for new experiments
│       ├── models/          # Directory for model weights
│       ├── logs/            # Directory for training logs
│       ├── servings/        # Directory for serving weights
│       ├── data/            # Directory for dataset
│       └── train_config.yaml # Configuration template
├── src/
│   └── models/
│       ├── cook_model.py    # Utility to combine servings into a model
│       ├── kitchen_setup.py # Setup for model decomposition
│       ├── tasting_menu.py  # Core class for menu-based model composition
│       ├── train_diversification.py # Training script for model diversification
│       ├── michelyn_inspector.py # Testing script for evaluating different menus
│       └── generate_kitchen.py # Utility to generate kitchen setup
```

## Core Concepts

- **Course**: A logical grouping of model components (e.g., a layer or set of layers)
- **Serving**: A specific implementation/variant of a course
- **Menu**: A selection of servings, one from each course, that together form a complete model
- **Kitchen**: The environment where models are decomposed and recomposed

## Key Components

### TastingMenu Class

The core class that handles the composition of models from different servings according to a menu.

### Kitchen Setup

Utilities for decomposing a model into courses and servings, and setting up the kitchen environment.

### Cook Model

Utility for combining servings into a complete model according to a specified menu.

### Training Diversification

Script for training models with diversification, using consistency loss between different menu configurations.

### Michelyn Inspector

Testing utility for evaluating different menu combinations and analyzing their performance.

## Usage

### Setting Up a New Experiment

1. Create a new experiment directory:
   ```
   mkdir -p experiment/YOUR_EXPERIMENT/{models,logs,servings,data}
   ```

2. Copy and modify the template files:
   ```
   cp experiment/templates/train_config.yaml experiment/YOUR_EXPERIMENT/
   cp experiment/templates/servings/serving_info.json experiment/YOUR_EXPERIMENT/servings/
   ```

3. Update the paths and parameters in these files for your specific experiment.

### Training with Diversification

```
python src/models/train_diversification.py
```

This script trains models with diversification, using consistency loss between different menu configurations.

### Testing Different Menus

```
python src/models/michelyn_inspector.py --menu_spec 10
```

This will test 10 different menu combinations and save the results.

Options for `menu_spec`:
- A number (e.g., `10`): Test that many random menus
- A percentage (e.g., `10%`): Test that percentage of all possible menus
- `full`: Test all possible menu combinations

## Requirements

- Python 3.8+
- PyTorch 1.8+
- torchvision
- numpy
- tqdm
- pyyaml

## License

[Your License Information] 