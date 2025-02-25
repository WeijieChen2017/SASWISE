import os
import re
import torch
from pathlib import Path
from typing import Dict, List, Any, Optional
import shutil
import json


class TastingMenu:
    """Class for managing course servings and building meals."""
    
    def __init__(self, model: torch.nn.Module, course_analysis_file: str, servings_dir: str = "course_servings"):
        """
        Initialize TastingMenu.
        
        Args:
            model: The base model to create servings from
            course_analysis_file: Path to the course analysis file
            servings_dir: Directory to store course servings
        """
        self.model = model
        self.course_analysis_file = course_analysis_file
        self.servings_dir = Path(servings_dir)
        self.servings_dir.mkdir(parents=True, exist_ok=True)
        
        # Load course analysis
        self.courses = self._load_course_analysis()
        
        # Create serving management system
        self.serving_registry = {}
        self._initialize_serving_registry()
    
    def _load_course_analysis(self) -> Dict[int, Dict[str, Any]]:
        """Load and parse the course analysis file."""
        courses = {}
        current_course = None
        
        with open(self.course_analysis_file, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            # Match course header with serving count
            course_match = re.match(r"Course (\d+)<(\d+)>:", line)
            if course_match:
                current_course = int(course_match.group(1))
                servings = int(course_match.group(2))
                courses[current_course] = {
                    'servings': servings,
                    'nodes': [],
                    'parameters': 0
                }
                continue
            
            # Match parameter count
            if current_course and "Total Parameters:" in line:
                param_count = int(line.split(":")[1].strip().replace(",", ""))
                courses[current_course]['parameters'] = param_count
                continue
            
            # Match nodes
            if current_course and line.strip().startswith("- "):
                node = line.strip()[2:].strip()
                courses[current_course]['nodes'].append(node)
        
        return courses
    
    def _initialize_serving_registry(self):
        """Initialize the serving registry with default entries."""
        for course_num, course_info in self.courses.items():
            self.serving_registry[course_num] = {
                'total_servings': course_info['servings'],
                'available_servings': []
            }
    
    def _get_course_state_dict(self, course_num: int) -> Dict[str, torch.Tensor]:
        """Extract state dict for a specific course."""
        course_state = {}
        course_nodes = self.courses[course_num]['nodes']
        
        full_state = self.model.state_dict()
        for node in course_nodes:
            # Find all parameters belonging to this node
            for param_name, param in full_state.items():
                if param_name.startswith(node):
                    course_state[param_name] = param.clone()
        
        return course_state
    
    def create_serving(self, course_num: int, serving_num: int):
        """
        Create a serving (copy) of a course.
        
        Args:
            course_num: Course number
            serving_num: Serving number to create
        """
        if course_num not in self.courses:
            raise ValueError(f"Course {course_num} not found")
            
        if serving_num > self.courses[course_num]['servings']:
            raise ValueError(f"Serving {serving_num} exceeds maximum servings for course {course_num}")
        
        # Extract course state
        course_state = self._get_course_state_dict(course_num)
        
        # Save serving
        serving_path = self.servings_dir / f"course_{course_num}_serving_{serving_num}.pt"
        torch.save(course_state, serving_path)
        
        # Update registry
        if serving_num not in self.serving_registry[course_num]['available_servings']:
            self.serving_registry[course_num]['available_servings'].append(serving_num)
    
    def create_all_servings(self):
        """Create all possible servings for all courses."""
        for course_num, course_info in self.courses.items():
            for serving_num in range(1, course_info['servings'] + 1):
                self.create_serving(course_num, serving_num)
    
    def build_meal(self, menu: Dict[int, int]) -> Dict[str, torch.Tensor]:
        """
        Build a meal from specified servings.
        
        Args:
            menu: Dictionary mapping course numbers to serving numbers
            
        Returns:
            Combined state dict for the specified meal
        """
        meal_state = {}
        
        # Validate menu
        for course_num, serving_num in menu.items():
            if course_num not in self.courses:
                raise ValueError(f"Course {course_num} not found")
            if serving_num not in self.serving_registry[course_num]['available_servings']:
                raise ValueError(f"Serving {serving_num} not available for course {course_num}")
        
        # Combine servings
        for course_num, serving_num in menu.items():
            serving_path = self.servings_dir / f"course_{course_num}_serving_{serving_num}.pt"
            course_state = torch.load(serving_path)
            meal_state.update(course_state)
        
        return meal_state
    
    def save_meal(self, menu: Dict[int, int], output_path: str):
        """
        Save a complete meal to a file.
        
        Args:
            menu: Dictionary mapping course numbers to serving numbers
            output_path: Path to save the meal state dict
        """
        meal_state = self.build_meal(menu)
        torch.save(meal_state, output_path)
    
    def list_available_servings(self) -> Dict[int, List[int]]:
        """
        List all available servings for each course.
        
        Returns:
            Dictionary mapping course numbers to lists of available serving numbers
        """
        return {
            course_num: info['available_servings']
            for course_num, info in self.serving_registry.items()
        }
    
    def get_serving_info(self, course_num: int, serving_num: int) -> Dict[str, Any]:
        """
        Get information about a specific serving.
        
        Args:
            course_num: Course number
            serving_num: Serving number
            
        Returns:
            Dictionary containing serving information
        """
        if course_num not in self.courses:
            raise ValueError(f"Course {course_num} not found")
            
        serving_path = self.servings_dir / f"course_{course_num}_serving_{serving_num}.pt"
        if not serving_path.exists():
            raise ValueError(f"Serving {serving_num} not found for course {course_num}")
            
        return {
            'course': course_num,
            'serving': serving_num,
            'path': str(serving_path),
            'parameters': self.courses[course_num]['parameters'],
            'nodes': self.courses[course_num]['nodes']
        }
    
    def cleanup(self):
        """Remove all serving files and the servings directory."""
        if self.servings_dir.exists():
            shutil.rmtree(self.servings_dir)
        self._initialize_serving_registry()
    
    def save_division_info(self, output_file):
        """Save course division and serving information to a text file."""
        # First analyze the courses
        self.analyze_courses()
        
        with open(output_file, 'w') as f:
            f.write("Course Division and Serving Information:\n")
            f.write("=====================================\n\n")
            
            # Write total parameters
            f.write(f"Total Model Parameters: {self.total_params:,}\n\n")
            
            # Write course information
            f.write("Course Information:\n")
            f.write("=================\n")
            for course_num in sorted(self.course_info.keys()):
                info = self.course_info[course_num]
                f.write(f"\nCourse {course_num}:\n")
                f.write(f"  Parameters: {info['params']:,}\n")
                f.write(f"  Nodes: {len(info['nodes'])}\n")
                f.write("  Components:\n")
                for node in sorted(info['nodes']):
                    f.write(f"    - {node}\n")
            
            f.write("\nServing Files Location:\n")
            f.write(f"  {self.servings_dir}\n\n")
            
            # Write example menu format
            f.write("Example Menu Format:\n")
            f.write("  menu = {\n")
            for course_num in sorted(self.course_info.keys()):
                f.write(f"    {course_num}: 1,  # Select serving number for course {course_num}\n")
            f.write("  }\n")

    def analyze_courses(self):
        """Analyze courses from the hierarchy file and store information."""
        self.course_info = {}
        self.total_params = 0
        
        # Read and parse the hierarchy file
        with open(self.course_analysis_file, 'r') as f:
            lines = f.readlines()
        
        current_course = None
        for line in lines:
            if '[' in line and ']' in line:
                # Extract course number if present
                course_match = line.split('[')[1].split(']')[0].strip()
                if course_match and course_match.isdigit():
                    current_course = int(course_match)
                    if current_course not in self.course_info:
                        self.course_info[current_course] = {
                            'params': 0,
                            'nodes': set()
                        }
                
                # Extract node name and parameters
                node = line.split('[')[0].strip('- ').strip()
                if '(params:' in line:
                    params = int(line.split('params:')[1].split(')')[0].strip().replace(',', ''))
                    if current_course is not None:
                        self.course_info[current_course]['params'] += params
                        self.course_info[current_course]['nodes'].add(node)
                        self.total_params += params 