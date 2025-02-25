"""Creator module for creating serving folders."""

import re
import json
import torch
from pathlib import Path
from typing import Dict, Any, List, Set


def get_course_nodes(course_analysis_path: str) -> Dict[int, Dict[str, Any]]:
    """Extract course information from course analysis file.
    
    Args:
        course_analysis_path: Path to the course analysis file
        
    Returns:
        dict: Course information with nodes and servings
    """
    courses = {}
    current_course = None
    
    with open(course_analysis_path, 'r') as f:
        for line in f:
            if line.startswith("Course "):
                # Extract course number and servings
                course_match = re.match(r"Course (\d+)<(\d+)>:", line)
                if course_match:
                    current_course = int(course_match.group(1))
                    num_servings = int(course_match.group(2))
                    courses[current_course] = {
                        'num_servings': num_servings,
                        'nodes': set()
                    }
            elif current_course is not None and line.strip().startswith("- "):
                # Extract node paths
                node = line.strip("- \n")
                courses[current_course]['nodes'].add(node)
                
    return courses


def create_serving_folders(course_analysis_path: str, root_dir: str, state_dict_path: str) -> Dict[str, Any]:
    """Create serving folders and state dicts based on course analysis.
    
    Args:
        course_analysis_path: Path to the course analysis file
        root_dir: Root directory where servings will be created
        state_dict_path: Path to the original state dict file
        
    Returns:
        dict: Information about created serving structure
    """
    # Create serving directories
    servings_dir = Path(root_dir) / "servings"
    servings_dir.mkdir(exist_ok=True)
    
    # Load original state dict
    state_dict = torch.load(state_dict_path, map_location='cpu')
    
    # Get course information
    courses = get_course_nodes(course_analysis_path)
    
    serving_info = {
        'root_dir': str(servings_dir),
        'courses': {}
    }
    
    # Create course directories and state dicts
    for course_num, course_info in courses.items():
        course_dir = servings_dir / f"course_{course_num}"
        course_dir.mkdir(exist_ok=True)
        
        serving_info['courses'][course_num] = {
            'num_servings': course_info['num_servings'],
            'nodes': list(course_info['nodes']),
            'servings': []
        }
        
        # Create state dict for each serving
        for serving in range(1, course_info['num_servings'] + 1):
            # Create a copy of state dict with only the nodes for this course
            course_state_dict = {
                key: value for key, value in state_dict.items()
                if any(key.startswith(node + '.') or key == node for node in course_info['nodes'])
            }
            
            # Save state dict
            state_dict_file = course_dir / f"serving_{serving}_state_dict.pt"
            torch.save(course_state_dict, state_dict_file)
            
            serving_info['courses'][course_num]['servings'].append({
                'serving': serving,
                'state_dict': str(state_dict_file),
                'status': "initialized"
            })
    
    # Save serving info to a single JSON file
    info_file = servings_dir / "serving_info.json"
    with open(info_file, 'w') as f:
        json.dump(serving_info, f, indent=2)
    
    return serving_info 