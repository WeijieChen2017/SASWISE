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

## Practical Example: Using SASWISE with ResNet for CIFAR-10

This example demonstrates how to use SASWISE with a pretrained ResNet model for CIFAR-10 classification.

### 1. Install Required Packages

```bash
pip install torch torchvision datasets
```

### 2. Run the CIFAR-10 ResNet Test Script

We provide a test script that automates the setup process:

```bash
python tests/test_CIFAR10_ResNet.py
```

This script will:
- Create the experiment folder structure for CIFAR10_ResNet
- Download the CIFAR-10 dataset using torchvision
- Load a pretrained ResNet-50 model from torchvision
- Save the model state dict to the experiment folder
- Print the next steps to run

### 3. Run the Kitchen Setup Commands

After running the test script, you'll need to execute the following commands in order:

```bash
# 1. Generate model hierarchy
python -m src.models.kitchen_setup.generate_hierarchy --state_dict CIFAR10_ResNet/models/base_model.pth --out CIFAR10_ResNet/models/hierarchy

# 2. Analyze course parameters
# Replace <output from Command 1> with the actual output file from the previous command
python -m src.models.kitchen_setup.course_analysis --model_hierarchy <output from Command 1> --out CIFAR10_ResNet/models/course_analysis

# 3. Create serving structure
# Replace <output from Command 2> with the actual output file from the previous command
python -m src.models.kitchen_setup.create_serving --course_analysis <output from Command 2> --out CIFAR10_ResNet --state_dict CIFAR10_ResNet/models/base_model.pth
```

### 4. Resulting Directory Structure

After completing these steps, you'll have a fully set up experiment with the following structure:

```
CIFAR10_ResNet/
├── models/
│   ├── base_model.pth           # The saved ResNet model state dict
│   ├── hierarchy/               # Model hierarchy information
│   └── course_analysis/         # Course analysis results
├── servings/
│   └── servings/                # Generated servings for each course
│       ├── course_1/
│       │   ├── serving_1_state_dict.pt
│       │   ├── serving_2_state_dict.pt
│       │   └── ...
│       ├── course_2/
│       │   ├── serving_1_state_dict.pt
│       │   └── ...
│       └── ...
├── logs/                        # Directory for training logs
└── data/                        # CIFAR-10 dataset
```

### 5. Next Steps

With this setup complete, you can now:
- Cook models with different serving combinations
- Train with diversification
- Evaluate different menu combinations

See the previous sections for details on these operations.

## Requirements

- Python 3.8+
- PyTorch 1.8+
- torchvision
- numpy
- tqdm
- pyyaml

## License

[Your License Information] 