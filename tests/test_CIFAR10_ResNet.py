# Run pip install datasets transformers already
import os
import torch
import ssl
import requests
from datasets import load_dataset
from transformers import AutoFeatureExtractor, ResNetForImageClassification

# This sets the default SSL context to an unverified one
ssl._create_default_https_context = ssl._create_unverified_context

def setup_experiment_folders():
    experiment_name = "CIFAR10_ResNet"
    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)
        os.makedirs(os.path.join(experiment_name, "models"))
        os.makedirs(os.path.join(experiment_name, "logs"))
        os.makedirs(os.path.join(experiment_name, "servings"))
        os.makedirs(os.path.join(experiment_name, "data"))
        # Create train_config.yaml file
        with open(os.path.join(experiment_name, "train_config.yaml"), 'w') as f:
            pass  # Creates an empty config file
    return experiment_name

def prepare_dataset(experiment_name):
    # Load CIFAR10 dataset
    dataset = load_dataset("cifar10")
    # Save dataset to experiment folder
    dataset.save_to_disk(os.path.join(experiment_name, "data"))
    print(dataset)
    return dataset

def prepare_model(experiment_name):
    # 1. Load the feature extractor
    feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/resnet-50")
    
    # 2. Load the pretrained ResNet model
    model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")
    
    # Save model state dict to experiment folder
    model_save_path = os.path.join(experiment_name, "models", "base_model.pth")
    model_state_dict = model.state_dict()
    torch.save(model_state_dict, model_save_path)
    print(f"Saved model state dict to {model_save_path}")
    return model, feature_extractor

def print_next_steps():
    """Print the commands that need to be run next"""
    commands = [
        "python -m src.models.kitchen_setup.generate_hierarchy --state_dict_path experiment/CIFAR10_ResNet/models/base_model.pth --out experiment/CIFAR10_ResNet/models/hierarchy.json",
        "python -m src.models.kitchen_setup.analyze_course_parameters --hierarchy_path experiment/CIFAR10_ResNet/models/hierarchy.json --course_dict experiment/CIFAR10_ResNet/models/course_dict.json --output_file experiment/CIFAR10_ResNet/models/course_analysis.json",
        "python -m src.models.kitchen_setup.generate_serving --hierarchy_path experiment/CIFAR10_ResNet/models/hierarchy.json --output_file experiment/CIFAR10_ResNet/servings/serving_info.json"
    ]
    
    print("\nNext steps - run these commands in order:")
    for i, cmd in enumerate(commands, 1):
        print(f"\n{i}. {cmd}")

def main():
    experiment_name = setup_experiment_folders()
    dataset = prepare_dataset(experiment_name)
    model, feature_extractor = prepare_model(experiment_name)
    print_next_steps()

if __name__ == "__main__":
    main()