import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path
import yaml
from datetime import datetime
from cook_model import cook_model
from torchvision.models import resnet18
import numpy as np
from tqdm import tqdm
import random
import itertools
import warnings


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_device(config):
    if config['training']['device'] == 'cuda' and torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def load_test_data(config):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = datasets.MNIST(
        config['paths']['data_dir'],
        train=False,
        transform=transform
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['data']['val_batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory']
    )
    
    return test_loader


def calculate_max_possible_menus(serving_info):
    """Calculate the maximum number of possible menu combinations."""
    num_servings_per_course = []
    for course_num in range(1, len(serving_info['courses']) + 1):
        course = serving_info['courses'][str(course_num)]
        num_servings_per_course.append(course['num_servings'])
    
    # Total possible combinations is the product of number of servings per course
    return np.prod(num_servings_per_course)


def generate_all_possible_menus(serving_info):
    """Generate all possible menu combinations."""
    serving_options = []
    for course_num in range(1, len(serving_info['courses']) + 1):
        course = serving_info['courses'][str(course_num)]
        serving_options.append(list(range(1, course['num_servings'] + 1)))
    
    # Generate all combinations
    return list(itertools.product(*serving_options))


def calculate_menu_similarity(menu1, menu2):
    """Calculate similarity between two menus."""
    return sum(1 for a, b in zip(menu1, menu2) if a == b) / len(menu1)


def generate_diverse_menu(serving_info, previous_menus=None, max_attempts=100):
    """Generate a menu that is sufficiently different from previous menus."""
    if previous_menus is None:
        previous_menus = []
    
    num_courses = len(serving_info['courses'])
    best_menu = None
    lowest_similarity = float('inf')
    
    for _ in range(max_attempts):
        menu = []
        for course_num in range(1, num_courses + 1):
            course = serving_info['courses'][str(course_num)]
            num_servings = course['num_servings']
            menu.append(random.randint(1, num_servings))
        
        if not previous_menus:
            return menu
        
        max_similarity = max(calculate_menu_similarity(menu, prev_menu) 
                           for prev_menu in previous_menus)
        
        if max_similarity < lowest_similarity:
            lowest_similarity = max_similarity
            best_menu = menu.copy()
        
        if lowest_similarity < 0.6:
            return best_menu
    
    return best_menu


def generate_menus(serving_info, menu_spec):
    """Generate menus based on the specification.
    
    Args:
        serving_info: Dictionary containing serving information
        menu_spec: Can be "full", a percentage like "10%", or a specific number
        
    Returns:
        List of menus to test
    """
    max_possible = calculate_max_possible_menus(serving_info)
    print(f"Maximum possible menu combinations: {max_possible}")
    
    # Determine how many menus to generate
    if menu_spec == "full":
        num_menus = max_possible
        print(f"Generating all {num_menus} possible menu combinations")
    elif isinstance(menu_spec, str) and menu_spec.endswith("%"):
        percentage = float(menu_spec.rstrip("%")) / 100
        num_menus = int(max_possible * percentage)
        print(f"Generating {num_menus} menus ({menu_spec} of all possible combinations)")
    else:
        try:
            num_menus = int(menu_spec)
            print(f"Generating {num_menus} menus")
        except ValueError:
            raise ValueError(f"Invalid menu specification: {menu_spec}. Use 'full', a percentage like '10%', or a number.")
    
    # Check if requested number exceeds maximum
    if num_menus > max_possible:
        warnings.warn(f"Requested {num_menus} menus, but only {max_possible} are possible. Using all possible menus.")
        num_menus = max_possible
    
    # Strategy depends on how many menus we need relative to the maximum
    if num_menus == max_possible:
        # Generate all possible combinations
        return generate_all_possible_menus(serving_info)
    elif num_menus > max_possible * 0.5:
        # If we need more than 50% of all possibilities, generate all and sample
        warnings.warn(f"Generating {num_menus} menus (>50% of all possible). This may take some time.")
        all_menus = generate_all_possible_menus(serving_info)
        random.shuffle(all_menus)
        return all_menus[:num_menus]
    else:
        # For a smaller number, generate diverse menus one by one
        menus = []
        for _ in tqdm(range(num_menus), desc="Generating diverse menus"):
            menu = generate_diverse_menu(serving_info, menus)
            menus.append(menu)
        return menus


def create_model(config, device, menu):
    """Create and load a model with the specified menu configuration."""
    model = resnet18()
    
    # Create temporary path for cooked model
    output_path = f"experiment/MNIST/models/temp_test_model.pth"
    cook_model(config['paths']['model_path'], menu, output_path)
    state_dict = torch.load(output_path, map_location=device)
    
    # Modify first conv layer for grayscale input
    model.conv1 = nn.Conv2d(
        config['model']['in_channels'],
        64,
        kernel_size=7,
        stride=2,
        padding=3,
        bias=False
    )
    
    if 'conv1.weight' in state_dict:
        rgb_weights = state_dict['conv1.weight']
        grayscale_weights = rgb_weights.mean(dim=1, keepdim=True)
        state_dict['conv1.weight'] = grayscale_weights
    
    model.fc = nn.Linear(512, config['model']['num_classes'])
    model.load_state_dict(state_dict, strict=False)
    
    # Clean up temporary file
    if os.path.exists(output_path):
        os.remove(output_path)
    
    return model.to(device)


def test_model(model, test_loader, device):
    """Test the model and return predictions and accuracy."""
    model.eval()
    all_preds = []
    all_labels = []
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc='Testing'):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())
            
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    accuracy = 100. * correct / total
    return np.array(all_preds), np.array(all_labels), accuracy


def main(menu_spec="10"):
    """Test multiple cooked meals and save their predictions.
    
    Args:
        menu_spec: Can be "full", a percentage like "10%", or a specific number
    """
    # Load configuration
    config = load_config('experiment/MNIST/train_config.yaml')
    device = get_device(config)
    print(f"Using device: {device}")
    
    # Load test data
    test_loader = load_test_data(config)
    print("Test data loaded successfully")
    
    # Load serving info
    servings_dir = Path(config['paths']['model_path']).parent.parent / "servings"
    with open(servings_dir / "serving_info.json", 'r') as f:
        serving_info = json.load(f)
    
    # Create results directory
    results_dir = Path(config['paths']['log_dir']) / "test_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate menus based on specification
    menus = generate_menus(serving_info, menu_spec)
    
    # Test each menu
    all_results = []
    
    for meal_idx, menu in enumerate(menus):
        print(f"\nTesting meal {meal_idx + 1}/{len(menus)}")
        print(f"Menu: {menu}")
        
        # Create and test model
        model = create_model(config, device, menu)
        predictions, labels, accuracy = test_model(model, test_loader, device)
        print(f"Test Accuracy: {accuracy:.2f}%")
        
        # Save results
        result = {
            'meal_id': meal_idx + 1,
            'menu': menu,
            'accuracy': accuracy,
            'predictions': predictions.tolist(),
            'true_labels': labels.tolist(),
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
        }
        all_results.append(result)
        
        # Save individual meal results
        meal_file = results_dir / f"meal_{meal_idx + 1}_results.json"
        with open(meal_file, 'w') as f:
            json.dump(result, f, indent=2)
    
    # Save summary of all meals
    summary_file = results_dir / "test_summary.json"
    summary = {
        'num_meals': len(menus),
        'menu_spec': menu_spec,
        'average_accuracy': np.mean([r['accuracy'] for r in all_results]),
        'meals': all_results
    }
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nTesting completed. Results saved in {results_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Test multiple cooked meals')
    parser.add_argument('--menu_spec', type=str, default="10",
                      help='Menu specification: "full", percentage like "10%", or a number (default: 10)')
    args = parser.parse_args()
    main(args.menu_spec) 