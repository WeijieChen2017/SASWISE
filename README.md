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
pip install torch torchvision
pip install git+https://github.com/huyvnphan/PyTorch_CIFAR10.git
```

### 2. Download and Set Up the Pretrained Model

```python
import torch
import os
from pathlib import Path
from cifar10_models import resnet

# Create experiment directories
experiment_dir = Path("experiment/CIFAR10")
models_dir = experiment_dir / "models"
servings_dir = experiment_dir / "servings"
os.makedirs(models_dir, exist_ok=True)
os.makedirs(servings_dir, exist_ok=True)

# Load pretrained ResNet18 for CIFAR-10
model = resnet.resnet18(pretrained=True)

# Save the base model
torch.save(model.state_dict(), models_dir / "resnet18_cifar10_base.pth")
```

### 3. Set Up the Kitchen (Decompose the Model)

```python
from src.models.kitchen_setup import setup_kitchen

# Analyze model and create hierarchy
model_hierarchy = setup_kitchen.analyze_model(model, "resnet18_cifar10")

# Define courses based on model layers
courses = [
    {"name": "course_1", "nodes": ["conv1", "bn1", "layer1"]},
    {"name": "course_2", "nodes": ["layer2"]},
    {"name": "course_3", "nodes": ["layer3"]},
    {"name": "course_4", "nodes": ["layer4"]},
    {"name": "course_5", "nodes": ["fc"]}
]

# Create servings for each course
for i, course in enumerate(courses, 1):
    setup_kitchen.create_servings(
        model, 
        course["nodes"], 
        servings_dir, 
        f"course_{i}", 
        num_servings=4
    )

# Generate serving_info.json
setup_kitchen.generate_serving_info(
    experiment_dir / "servings" / "serving_info.json",
    "CIFAR10",
    "resnet18",
    courses
)
```

### 4. Cook a Model with a Specific Menu

```python
from src.models.cook_model import cook_model

# Define a menu (one serving from each course)
menu = [1, 2, 3, 2, 1]  # Using serving 1, 2, 3, 2, 1 for courses 1-5

# Cook the model
cooked_model_path = models_dir / "cooked_model.pth"
cook_model(
    str(models_dir / "resnet18_cifar10_base.pth"),
    menu,
    str(cooked_model_path)
)

# Load the cooked model
model = resnet.resnet18(pretrained=False)
model.load_state_dict(torch.load(cooked_model_path))
model.eval()
```

### 5. Use the Cooked Model for Inference

```python
import torchvision.transforms as transforms
from PIL import Image

# Define transforms for CIFAR-10
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], 
                         std=[0.2471, 0.2435, 0.2616])
])

# Load and preprocess an image
img = Image.open('path_to_your_image.jpg')
img = transform(img).unsqueeze(0)  # Add batch dimension

# Move to appropriate device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
img = img.to(device)

# Get predictions
with torch.no_grad():
    outputs = model(img)
    _, predicted = outputs.max(1)

print(f'Predicted class: {predicted.item()}')
```

### 6. Train with Diversification

```python
# Create a train_config.yaml file in your experiment directory
# Then run:
python src/models/train_diversification.py --config experiment/CIFAR10/train_config.yaml
```

### 7. Evaluate Different Menu Combinations

```python
# Test 10 different menu combinations
python src/models/michelyn_inspector.py --menu_spec 10 --config experiment/CIFAR10/train_config.yaml
```

This example demonstrates the complete workflow of using SASWISE with a pretrained ResNet model for CIFAR-10, from setting up the kitchen to cooking models with different menus and evaluating their performance.

## Requirements

- Python 3.8+
- PyTorch 1.8+
- torchvision
- numpy
- tqdm
- pyyaml

## License

[Your License Information] 