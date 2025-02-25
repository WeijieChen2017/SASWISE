import os
import torch
import json
from pathlib import Path
from monai.networks.nets import ViT
from typing import Dict, List, Any
import re


def create_kitchen_structure(course_analysis_file: str, base_dir: str = "project"):
    """Create kitchen folder structure and save course servings."""
    # Create base directory structure
    project_dir = Path(base_dir)
    kitchen_dir = project_dir / "kitchen"
    kitchen_dir.mkdir(parents=True, exist_ok=True)

    # Create ViT model
    model = ViT(
        in_channels=1,
        img_size=(96, 96, 96),
        patch_size=(16, 16, 16),
        hidden_size=768,
        mlp_dim=3072,
        num_layers=12,
        num_heads=12,
        num_classes=2
    )

    # Parse course analysis file
    courses = {}
    current_course = None
    current_servings = 0

    with open(course_analysis_file, 'r') as f:
        for line in f:
            # Match course header with serving count
            course_match = re.match(r"Course (\d+)<(\d+)>:", line)
            if course_match:
                current_course = int(course_match.group(1))
                current_servings = int(course_match.group(2))
                courses[current_course] = {
                    'servings': current_servings,
                    'nodes': [],
                }
                continue

            # Match nodes
            if current_course and line.strip().startswith("- "):
                node = line.strip()[2:].strip()
                courses[current_course]['nodes'].append(node)

    # Create course directories and save servings
    state_dict = model.state_dict()

    for course_num, course_info in courses.items():
        # Create course directory
        course_dir = kitchen_dir / f"course_{course_num}"
        course_dir.mkdir(exist_ok=True)

        # Extract parameters for this course
        course_state = {}
        total_params = 0
        for node in course_info['nodes']:
            # Find all parameters belonging to this node
            node_params = {}
            for param_name, param in state_dict.items():
                if param_name.startswith(node):
                    param_shape = list(param.shape)
                    param_size = param.numel()
                    node_params[param_name] = {
                        'shape': param_shape,
                        'size': param_size
                    }
                    total_params += param_size

        # Save course info with parameter shapes and sizes
        info = {
            'num_servings': course_info['servings'],
            'nodes': course_info['nodes'],
            'parameters': total_params,
            'parameter_info': node_params
        }
        with open(course_dir / "info.json", 'w') as f:
            json.dump(info, f, indent=2)

        # Save serving metadata
        for serving in range(1, course_info['servings'] + 1):
            serving_info = {
                'course': course_num,
                'serving_num': serving,
                'nodes': course_info['nodes'],
                'total_parameters': total_params
            }
            with open(course_dir / f"serving_{serving}.json", 'w') as f:
                json.dump(serving_info, f, indent=2)

    # Save kitchen info
    kitchen_info = {
        'total_courses': len(courses),
        'courses': courses,
        'total_parameters': sum(p.numel() for p in state_dict.values()),
        'model_config': {
            'in_channels': 1,
            'img_size': [96, 96, 96],
            'patch_size': [16, 16, 16],
            'hidden_size': 768,
            'mlp_dim': 3072,
            'num_layers': 12,
            'num_heads': 12,
            'num_classes': 2
        }
    }
    with open(kitchen_dir / "kitchen_info.json", 'w') as f:
        json.dump(kitchen_info, f, indent=2)

    return project_dir


if __name__ == "__main__":
    course_analysis_file = "course_analysis_ViT_20250221_155726.txt"
    project_dir = create_kitchen_structure(course_analysis_file)
    print(f"Kitchen structure created at: {project_dir}")
    print("\nStructure created:")
    print("project/")
    print("└── kitchen/")
    for course in sorted(os.listdir(project_dir / "kitchen")):
        if course.startswith("course_"):
            print(f"    ├── {course}/")
            course_path = project_dir / "kitchen" / course
            for file in sorted(os.listdir(course_path)):
                print(f"    │   ├── {file}")
    print("    └── kitchen_info.json") 