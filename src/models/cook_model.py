import os
import json
import torch
from pathlib import Path
from typing import List, Dict, Any


def cook_model(path_to_saved_model: str, menu: List[int], output_path: str) -> None:
    """Cook a model by combining different servings according to the menu.
    
    Args:
        path_to_saved_model: Path to the original saved model or pretrained weights
        menu: List of serving numbers, with length matching number of courses
        output_path: Path where to save the cooked model
        
    The menu list should have the same length as the number of courses in serving_info.json,
    and each number should be within the valid range of servings for that course.
    """
    # Load the original model state dict
    original_state_dict = torch.load(path_to_saved_model, map_location='cpu')
    
    # Get the experiment root directory (two levels up from the model file)
    experiment_dir = Path(path_to_saved_model).parent.parent
    servings_dir = experiment_dir / "servings"
    
    # Load serving info
    with open(servings_dir / "serving_info.json", 'r') as f:
        serving_info = json.load(f)
    
    # Validate menu length
    num_courses = len(serving_info['courses'])
    if len(menu) != num_courses:
        raise ValueError(f"Menu length ({len(menu)}) does not match number of courses ({num_courses})")
    
    # Create new state dict for the cooked model
    cooked_state_dict = {}
    
    # For each course, load the specified serving and update the state dict
    for course_num, serving_num in enumerate(menu, start=1):
        course_info = serving_info['courses'][str(course_num)]
        
        # Validate serving number
        if serving_num < 1 or serving_num > course_info['num_servings']:
            raise ValueError(f"Invalid serving number {serving_num} for course {course_num}. "
                           f"Must be between 1 and {course_info['num_servings']}")
        
        # Get the serving state dict path
        serving_path = course_info['servings'][serving_num - 1]['state_dict']
        
        # Load the serving state dict
        serving_state_dict = torch.load(serving_path, map_location='cpu')
        
        # Update the cooked state dict with this serving's parameters
        cooked_state_dict.update(serving_state_dict)
    
    # Save the cooked model
    torch.save(cooked_state_dict, output_path)


def test_cook_model():
    """Test the cook_model function using actual MNIST experiment files."""
    # Define paths
    path_to_saved_model = "experiment/MNIST/models/resnet18_mnist_init.pth"
    output_path = "experiment/MNIST/models/current_meal_model.pth"
    
    # Create menu with all 1's (first serving for each course)
    # From the serving_info.json we know there are 5 courses
    menu = [1, 1, 1, 1, 1]
    
    try:
        print("Testing cook_model with actual MNIST experiment files...")
        cook_model(path_to_saved_model, menu, output_path)
        print(f"✓ Model cooked successfully")
        print(f"✓ Cooked model saved to: {output_path}")
    except Exception as e:
        print(f"✗ Test failed: {str(e)}")


if __name__ == "__main__":
    test_cook_model() 