import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import torch.nn.functional as F
import yaml
from datetime import datetime
from collections import defaultdict
import random
import numpy as np
import argparse
import sys
import importlib

# Add the project root to the path so we can import modules properly
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.models.cook_model import cook_model


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_device(config):
    if config['training']['device'] == 'cuda' and torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def load_data(config):
    """Load dataset from disk based on config settings."""
    data_dir = config['paths']['data_dir']
    
    # Load dataset using the method specified in config
    dataset_module = importlib.import_module(config['data']['dataset_module'])
    dataset_loader = getattr(dataset_module, config['data']['dataset_loader'])
    
    # Load train and validation datasets
    train_dataset, val_dataset = dataset_loader(data_dir, **config['data'].get('dataset_args', {}))
    
    # Create subsets if specified (optional)
    if 'train_subset_fraction' in config['data'] and config['data']['train_subset_fraction'] < 1.0:
        train_size = len(train_dataset)
        val_size = len(val_dataset)
        
        train_subset_size = int(train_size * config['data']['train_subset_fraction'])
        val_subset_size = int(val_size * config['data'].get('val_subset_fraction', 1.0))
        
        train_indices = torch.randperm(train_size)[:train_subset_size]
        val_indices = torch.randperm(val_size)[:val_subset_size]
        
        train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
        val_dataset = torch.utils.data.Subset(val_dataset, val_indices)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['data']['train_batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['data']['val_batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory']
    )
    
    print(f"Using {len(train_dataset)} training samples and {len(val_dataset)} validation samples")
    return train_loader, val_loader


def create_model(config, device, menu, model_num):
    """Create a model with specified menu configuration."""
    # Load model architecture using the method specified in config
    model_module = importlib.import_module(config['model']['model_module'])
    model_class = getattr(model_module, config['model']['model_class'])
    
    # Create model instance
    model = model_class(**config['model'].get('model_args', {}))
    
    # Load cooked model state dict
    model_path = config['paths']['model_path']
    output_path = f"{os.path.dirname(model_path)}/current_meal_model_{model_num}.pth"
    
    # Use cook_model to create a model with the specified menu
    cook_model(model_path, menu, output_path)
    state_dict = torch.load(output_path, map_location=device)
    
    # Load the state dict
    model.load_state_dict(state_dict, strict=False)
    
    return model.to(device)


def train_epoch(model1, model2, train_loader, criterion, optimizer, device, alpha):
    model1.train()
    model2.eval()  # model2 is only used for consistency loss, no updates
    
    running_acc_loss = 0.0
    running_cons_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass through both models
        outputs1 = model1(inputs)
        with torch.no_grad():
            outputs2 = model2(inputs)
        
        # Calculate accuracy loss (cross entropy with ground truth)
        accuracy_loss = criterion(outputs1, targets)
        
        # Calculate consistency loss (MSE between model outputs)
        consistency_loss = F.mse_loss(F.softmax(outputs1, dim=1), 
                                    F.softmax(outputs2, dim=1))
        
        # Combined loss with alpha weight
        loss = accuracy_loss + alpha * consistency_loss
        
        # Backward pass (only for model1)
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_acc_loss += accuracy_loss.item()
        running_cons_loss += consistency_loss.item()
        _, predicted = outputs1.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        pbar.set_postfix({
            'acc_loss': running_acc_loss/total,
            'cons_loss': running_cons_loss/total,
            'acc': 100.*correct/total
        })
    
    return (running_acc_loss/len(train_loader), 
            running_cons_loss/len(train_loader), 
            100.*correct/total)


def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    return running_loss/len(val_loader), 100.*correct/total


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
        
        # If no previous menus, accept first generated menu
        if not previous_menus:
            return menu
        
        # Calculate maximum similarity to any previous menu
        max_similarity = max(calculate_menu_similarity(menu, prev_menu) 
                           for prev_menu in previous_menus)
        
        # Update best menu if this one is more diverse
        if max_similarity < lowest_similarity:
            lowest_similarity = max_similarity
            best_menu = menu.copy()
        
        # Accept menu if it's diverse enough
        if lowest_similarity < 0.6:
            return best_menu
    
    # Return best found menu if we couldn't find a perfectly diverse one
    return best_menu


def update_serving_weights(model, menu, serving_info, device):
    """Update weights for each serving used in the menu."""
    state_dict = model.state_dict()
    
    # For each course in the menu
    for course_num, serving_num in enumerate(menu, start=1):
        course = serving_info['courses'][str(course_num)]
        serving = course['servings'][serving_num - 1]
        
        # Get the nodes for this course
        course_nodes = course['nodes']
        
        # Create a state dict for this serving containing only its nodes
        serving_state_dict = {
            key: value for key, value in state_dict.items()
            if any(key.startswith(node) or key == node for node in course_nodes)
        }
        
        # Save the updated weights
        torch.save(serving_state_dict, serving['state_dict'])


def update_serving_history(serving_info, menu, epochs, log_dir):
    """Update training history for each serving."""
    history_file = log_dir / "serving_training_history.json"
    
    # Load existing history or create new
    if history_file.exists():
        with open(history_file, 'r') as f:
            history = json.load(f)
    else:
        history = defaultdict(lambda: defaultdict(int))
        history = dict(history)  # Convert to regular dict for JSON serialization
    
    # Update epoch counts for each serving in the menu
    for course_num, serving_num in enumerate(menu, start=1):
        course_key = f"course_{course_num}"
        serving_key = f"serving_{serving_num}"
        
        if course_key not in history:
            history[course_key] = {}
        if serving_key not in history[course_key]:
            history[course_key][serving_key] = 0
            
        history[course_key][serving_key] += epochs
    
    # Save updated history
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=2)


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train model with diversification')
    parser.add_argument('--config', type=str, required=True, 
                        help='Path to the configuration YAML file')
    args = parser.parse_args()
    
    # Load configuration from the provided path
    config = load_config(args.config)
    
    # Set up device
    device = get_device(config)
    print(f"Using device: {device}")
    
    # Create log directory if it doesn't exist
    log_dir = Path(config['paths']['log_dir'])
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    train_loader, val_loader = load_data(config)
    print("Data loaded successfully")
    
    # Load serving info for menu generation
    servings_dir = Path(config['paths']['model_path']).parent.parent / "servings"
    with open(servings_dir / "serving_info.json", 'r') as f:
        serving_info = json.load(f)
    
    # Training loop
    print("\nStarting training...")
    log_file = log_dir / "training_log.json"
    
    # Initialize menu history
    menu_history = []
    
    for round_idx in range(1, config['training']['num_rounds'] + 1):
        print(f"\nRound {round_idx}/{config['training']['num_rounds']}")
        
        # Generate diverse menus for this round
        menu1 = generate_diverse_menu(serving_info, menu_history)
        menu_history.append(menu1)
        
        # Generate second menu diverse from both menu history and menu1
        menu2 = generate_diverse_menu(serving_info, menu_history)
        menu_history.append(menu2)
        
        # Keep menu history manageable (keep last 5 rounds = 10 menus)
        if len(menu_history) > 10:
            menu_history = menu_history[-10:]
        
        similarity = calculate_menu_similarity(menu1, menu2)
        print(f"Menu 1: {menu1}")
        print(f"Menu 2: {menu2}")
        print(f"Menu similarity: {similarity:.2f}")
        
        # Create models with different menus
        model1 = create_model(config, device, menu1, 1)
        model2 = create_model(config, device, menu2, 2)
        print("Models created and loaded successfully")
        
        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(
            model1.parameters(),
            lr=config['training']['learning_rate'],
            momentum=config['training']['momentum'],
            weight_decay=config['training']['weight_decay']
        )
        
        # Train for specified number of epochs
        for epoch in range(1, config['training']['epochs_per_round'] + 1):
            print(f"\nEpoch {epoch}/{config['training']['epochs_per_round']}")
            
            # Train
            train_acc_loss, train_cons_loss, train_acc = train_epoch(
                model1, model2, train_loader, criterion, optimizer, device,
                config['training']['alpha']
            )
            
            # Evaluate periodically
            if epoch % config['training']['eval_per_epoch'] == 0:
                val_loss, val_acc = validate(model1, val_loader, criterion, device)
                print(f"Epoch {epoch} - Train Acc Loss: {train_acc_loss:.4f}, "
                      f"Cons Loss: {train_cons_loss:.4f}, Accuracy: {train_acc:.2f}%")
                print(f"Validation - Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%")
                
                # Log results with menu similarity
                log_entry = {
                    'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
                    'round': round_idx,
                    'menu1': menu1,
                    'menu2': menu2,
                    'menu_similarity': similarity,
                    'epochs': epoch,
                    'train_accuracy_loss': train_acc_loss,
                    'train_consistency_loss': train_cons_loss,
                    'train_accuracy': train_acc,
                    'val_loss': val_loss,
                    'val_accuracy': val_acc
                }
                with open(log_file, 'a') as f:
                    json.dump(log_entry, f)
                    f.write('\n')
        
        # Update serving weights and history
        update_serving_weights(model1, menu1, serving_info, device)
        update_serving_history(serving_info, menu1, config['training']['epochs_per_round'], log_dir)
        print(f"Updated weights and history for menu 1 servings")
    
    print("\nTraining completed. Check serving_training_history.json for serving usage details.")


if __name__ == "__main__":
    main() 