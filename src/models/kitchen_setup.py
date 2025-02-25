from typing import Dict, Any, Optional, Tuple, List
import torch
from collections import defaultdict
import argparse
from datetime import datetime
from pathlib import Path
import re
import json

from src.models.model_loader import load_pretrained_model, get_model_info


class ModelHierarchy:
    """Class representing the hierarchical structure of a model."""
    
    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.hierarchy = self._build_hierarchy()
        self.parameter_counts = self._count_parameters()
    
    def _build_hierarchy(self) -> Dict[str, Any]:
        """Build a tree structure representing the model hierarchy."""
        hierarchy = {}
        
        for name, module in self.model.named_modules():
            if not name:  # Skip root
                continue
                
            parts = name.split('.')
            current = hierarchy
            
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            
            current[parts[-1]] = {
                'type': type(module).__name__,
                'children': {}
            }
        
        return hierarchy
    
    def _count_parameters(self) -> Dict[str, int]:
        """Count parameters for each node in the hierarchy."""
        counts = defaultdict(int)
        
        for name, param in self.model.named_parameters():
            module_path = '.'.join(name.split('.')[:-1])
            counts[module_path] += param.numel()
            
            # Add to parent counts
            while '.' in module_path:
                module_path = '.'.join(module_path.split('.')[:-1])
                counts[module_path] += param.numel()
        
        return dict(counts)
    
    def get_node_info(self, node_path: str) -> Dict[str, Any]:
        """Get information about a specific node in the hierarchy."""
        parts = node_path.split('.')
        current = self.hierarchy
        
        for part in parts:
            if part not in current:
                raise ValueError(f"Invalid node path: {node_path}")
            current = current[part]
        
        return {
            'type': current['type'],
            'parameters': self.parameter_counts.get(node_path, 0),
            'children': list(current['children'].keys())
        }


def kitchen_setup(
    model_path: str,
    model_type: str,
    device: Optional[str] = None
) -> Dict[str, Any]:
    """
    Load pretrained weights and output the model hierarchy.
    
    Args:
        model_path: Path to pretrained model weights
        model_type: Type of model to load
        device: Device to load model on
    
    Returns:
        Dictionary containing:
        - model: The loaded model
        - hierarchy: ModelHierarchy instance
        - info: General model information
    """
    # Load the model
    model = load_pretrained_model(model_path, model_type, device)
    
    # Create hierarchy
    hierarchy = ModelHierarchy(model)
    
    # Get general model info
    info = get_model_info(model)
    
    return {
        'model': model,
        'hierarchy': hierarchy,
        'info': info
    }


def build_model_hierarchy_from_state_dict(state_dict_path: str) -> Tuple[Dict[str, Any], Dict[str, int]]:
    """
    Build a hierarchical representation of model structure from a state dictionary file.
    
    Args:
        state_dict_path: Path to the PyTorch state dictionary file
        
    Returns:
        Tuple containing:
        - hierarchy: Dictionary representing the model's hierarchical structure
        - param_counts: Dictionary containing parameter counts for each node
    """
    # Load state dict
    state_dict = torch.load(state_dict_path, map_location='cpu')
    
    # Initialize hierarchy and parameter counts
    hierarchy = {}
    param_counts = defaultdict(int)
    
    # First pass: Count parameters for each node
    for key, param in state_dict.items():
        parts = key.split('.')
        current_path = []
        
        for part in parts[:-1]:  # Exclude the parameter name
            current_path.append(part)
            path_str = '.'.join(current_path)
            param_counts[path_str] += param.numel()
    
    # Second pass: Build hierarchy
    for key in state_dict.keys():
        parts = key.split('.')
        current_dict = hierarchy
        current_path = []
        
        for part in parts[:-1]:  # Exclude the parameter name
            current_path.append(part)
            path_str = '.'.join(current_path)
            
            if path_str not in current_dict:
                current_dict[path_str] = {
                    'params': [],
                    'children': {}
                }
            
            # Add parameter to the immediate parent's param list
            if part == parts[-2]:  # If this is the immediate parent
                current_dict[path_str]['params'].append(key)
            
            current_dict = current_dict[path_str]['children']
    
    return hierarchy, dict(param_counts)


def save_model_hierarchy(hierarchy: Dict[str, Any], param_counts: Dict[str, int], output_path: str, root_dir: Optional[str] = None):
    """
    Save the model hierarchy to a text file with brackets for course indexing.
    
    Args:
        hierarchy: Dictionary representing the model's hierarchical structure
        param_counts: Dictionary containing parameter counts for each node
        output_path: Path where to save the hierarchy text file
        root_dir: Optional root directory to prepend to output_path
    """
    if root_dir:
        output_path = str(Path(root_dir) / output_path)
        Path(root_dir).mkdir(parents=True, exist_ok=True)
        
    with open(output_path, 'w') as f:
        f.write("Model Hierarchy:\n")
        f.write("Instructions: Add course indices in the brackets [ ] for nodes you want to group.\n")
        f.write("Rules:\n")
        f.write("1. If a node has an index, all its children belong to that course\n")
        f.write("2. If multiple siblings have the same index, they form one course\n")
        f.write("3. Empty brackets [ ] mean this node will inherit its course index from its parent node\n\n")
        
        def write_hierarchy(d, level=0):
            def sort_key(k):
                # Split the key into parts
                parts = k.split('.')
                # Convert numerical parts to integers for sorting
                return [int(p) if p.isdigit() else p for p in parts]
            
            for k, v in sorted(d.items(), key=lambda x: sort_key(x[0])):
                indent = '  ' * level
                f.write(f"{indent}- {k} [ ] (params: {param_counts.get(k, 0):,})\n")
                write_hierarchy(v['children'], level + 1)
        
        write_hierarchy(hierarchy)


def parse_user_course_indexing(hierarchy_file):
    """Parse the user's course indexing from the hierarchy file.
    
    Args:
        hierarchy_file (str): Path to the hierarchy file with user's course indexing
        
    Returns:
        tuple: (course_dict, void_nodes, overlap_nodes, empty_box_nodes)
            - course_dict: Dictionary mapping course indices to lists of node paths
            - void_nodes: List of nodes with no course assignment and not all descendants indexed
            - overlap_nodes: Dictionary mapping nodes to their multiple course assignments
            - empty_box_nodes: List of nodes that are empty boxes (all descendants are assigned or empty boxes)
    """
    course_dict = {}  # Maps course index to list of nodes
    node_courses = {}  # Maps node to list of courses it belongs to
    void_nodes = []
    node_children = {}  # Maps node to its children
    node_descendants = defaultdict(set)  # Maps node to all its descendants
    empty_box_nodes = []  # Nodes that are empty boxes
    
    def get_parent_course(node_path):
        # Split the path into components
        parts = node_path.split('.')
        # Try increasingly shorter paths until we find a parent with a course
        while len(parts) > 1:
            parts.pop()
            parent = '.'.join(parts)
            if parent in node_courses and node_courses[parent]:
                return parent, node_courses[parent][0]  # Return parent path and its first course
        return None, None

    # First pass: collect node relationships and explicit course assignments
    with open(hierarchy_file, 'r') as f:
        for line in f:
            if '[' not in line or ']' not in line:
                continue
            
            # Skip header lines
            if line.strip().startswith(('Instructions', 'Rules', 'Model')):
                continue
            
            # Extract node path and course index
            node = line.split('[')[0].strip('- \t')
            course_str = line.split('[')[1].split(']')[0].strip()
            
            # Record parent-child relationship and build descendant tree
            if '.' in node:
                parts = node.split('.')
                # Add node as descendant to all its ancestors
                for i in range(1, len(parts)):
                    ancestor = '.'.join(parts[:i])
                    node_descendants[ancestor].add(node)
                
                # Record immediate parent-child relationship
                parent = '.'.join(parts[:-1])
                if parent not in node_children:
                    node_children[parent] = []
                node_children[parent].append(node)
            
            # Record course assignment
            if course_str:  # Has explicit course index
                try:
                    course = int(course_str)
                    if course not in course_dict:
                        course_dict[course] = []
                    course_dict[course].append(node)
                    if node not in node_courses:
                        node_courses[node] = []
                    node_courses[node].append(course)
                except ValueError:
                    continue
            else:  # Empty brackets
                node_courses[node] = []
                void_nodes.append(node)
    
    # Second pass: handle inheritance and check for overlaps
    overlap_nodes = {}
    
    # First check explicit course assignments for overlaps with parents
    for node, courses in node_courses.items():
        if courses:  # Node has explicit course assignment
            parent_node, parent_course = get_parent_course(node)
            if parent_course is not None and parent_course not in courses:
                # Node has different course than parent - mark as overlap
                if node not in overlap_nodes:
                    overlap_nodes[node] = []
                overlap_nodes[node] = courses + [parent_course]
                # Add to both courses in course_dict
                if parent_course not in course_dict:
                    course_dict[parent_course] = []
                if node not in course_dict[parent_course]:
                    course_dict[parent_course].append(node)
    
    # Third pass: identify empty box nodes
    def is_empty_box(node):
        descendants = node_descendants.get(node, set())
        if not descendants:
            # If this is a leaf node with empty brackets, it's not an empty box
            return False
        
        # Get all descendants that have course assignments
        assigned_descendants = set()
        for course_nodes in course_dict.values():
            assigned_descendants.update(set(course_nodes))
        
        # Check if all immediate children are either:
        # 1. Assigned to a course
        # 2. Already marked as empty boxes
        # 3. Have all their descendants assigned
        children = node_children.get(node, [])
        for child in children:
            if child in assigned_descendants:
                continue
            if child in empty_box_nodes:
                continue
            # Check if all descendants of this child are assigned
            child_descendants = node_descendants.get(child, set())
            if not child_descendants or not all(desc in assigned_descendants for desc in child_descendants):
                return False
        return True
    
    # Iterate through nodes multiple times to handle nested empty boxes
    max_iterations = 10  # Prevent infinite loops
    for _ in range(max_iterations):
        found_new_empty_box = False
        for node in list(void_nodes):  # Use list to avoid modifying during iteration
            if is_empty_box(node):
                empty_box_nodes.append(node)
                void_nodes.remove(node)
                found_new_empty_box = True
        if not found_new_empty_box:
            break
    
    # Handle inheritance for remaining void nodes
    for node in list(void_nodes):  # Use list to avoid modifying during iteration
        parent_node, parent_course = get_parent_course(node)
        if parent_course is not None:
            if parent_course not in course_dict:
                course_dict[parent_course] = []
            course_dict[parent_course].append(node)
            node_courses[node] = [parent_course]
            void_nodes.remove(node)
    
    return course_dict, void_nodes, overlap_nodes, empty_box_nodes


def analyze_course_parameters(hierarchy_file, course_dict, void_nodes, overlap_nodes, empty_box_nodes):
    """Analyze parameters for each course based on the hierarchy file and course assignments.
    
    Args:
        hierarchy_file (str): Path to the hierarchy file
        course_dict (dict): Dictionary mapping course indices to lists of node paths
        void_nodes (list): List of nodes with no course assignment
        overlap_nodes (dict): Dictionary mapping nodes to their multiple course assignments
        empty_box_nodes (list): List of nodes that are empty boxes
        
    Returns:
        dict: Analysis results containing:
            - course_info: Dictionary mapping course indices to their parameter counts and nodes
            - void_info: Information about void nodes and their parameters
            - overlap_info: Information about overlapping nodes
            - empty_box_info: Information about empty box nodes (no parameter counting)
            - total_params: Total parameters across all courses (leaf nodes only)
            - validation_info: Comparison between course parameters and actual model parameters
    """
    analysis = {
        'course_info': {},
        'void_info': {'total_params': 0, 'nodes': []},
        'overlap_info': {},
        'empty_box_info': {'nodes': []},  # No parameter counting for empty boxes
        'total_params': 0,
        'validation_info': {'leaf_params': 0, 'all_leaf_nodes': set()}
    }
    
    # Extract parameter counts and build node relationships
    node_params = {}
    node_children = defaultdict(list)
    all_nodes = set()
    
    with open(hierarchy_file, 'r') as f:
        for line in f:
            if '[' not in line or ']' not in line:
                continue
            if line.strip().startswith(('Instructions', 'Rules', 'Model')):
                continue
                
            # Extract node and parameters
            node = line.split('[')[0].strip('- ')
            if '(params:' in line:
                params = int(line.split('(params:')[1].split(')')[0].strip().replace(',', ''))
                node_params[node] = params
            
            # Build parent-child relationships
            all_nodes.add(node)
            if '.' in node:
                parent = '.'.join(node.split('.')[:-1])
                node_children[parent].append(node)
    
    # Identify leaf nodes (nodes with no children)
    leaf_nodes = {node for node in all_nodes if node not in node_children}
    analysis['validation_info']['all_leaf_nodes'] = leaf_nodes
    
    # Analyze parameters for each course (counting only leaf nodes)
    for course, nodes in course_dict.items():
        course_leaf_params = 0
        course_leaf_nodes = []
        
        for node in nodes:
            # If node is a leaf node, count its parameters
            if node in leaf_nodes and node in node_params:
                course_leaf_params += node_params[node]
                course_leaf_nodes.append(node)
                analysis['validation_info']['leaf_params'] += node_params[node]
        
        analysis['course_info'][course] = {
            'total_params': course_leaf_params,
            'nodes': nodes,
            'leaf_nodes': course_leaf_nodes
        }
        analysis['total_params'] += course_leaf_params
    
    # Analyze void nodes (only leaf nodes)
    for node in void_nodes:
        if node in leaf_nodes and node in node_params:
            analysis['void_info']['total_params'] += node_params[node]
            analysis['void_info']['nodes'].append(node)
    
    # Record empty box nodes (without parameter counting)
    analysis['empty_box_info']['nodes'] = empty_box_nodes
    
    # Analyze overlapping nodes (only leaf nodes)
    for node, courses in overlap_nodes.items():
        if node in leaf_nodes and node in node_params:
            analysis['overlap_info'][node] = {
                'courses': courses,
                'params': node_params[node]
            }
    
    return analysis


def save_course_analysis(analysis, output_file):
    """Save the course analysis results to a file.
    
    Args:
        analysis (dict): Analysis results from analyze_course_parameters
        output_file (str): Path to save the analysis results
    """
    with open(output_file, 'w') as f:
        f.write("Course Analysis:\n\n")

        # Write course information with serving counts
        for course, info in sorted(analysis['course_info'].items()):
            # Default to 1 serving if not specified
            servings = info.get('servings', 1)
            f.write(f"Course {course}<{servings}>:\n")
            f.write(f"Total Parameters: {info['total_params']:,}\n")
            f.write("Nodes:\n")
            for node in sorted(info['nodes']):
                f.write(f"  - {node}\n")
            f.write("\n")

        # Write empty box information
        if analysis['empty_box_info']['nodes']:
            f.write("\nEmpty Box Nodes (all descendants are assigned or empty boxes):\n")
            for node in sorted(analysis['empty_box_info']['nodes']):
                f.write(f"  - {node}\n")
            f.write("\n")

        # Write void node information
        if analysis['void_info']['nodes']:
            f.write("\nNodes with no course assignment:\n")
            f.write(f"Total Parameters: {analysis['void_info']['total_params']:,}\n")
            for node in sorted(analysis['void_info']['nodes']):
                f.write(f"  - {node}\n")
            f.write("\n")

        # Write overlap information
        if analysis['overlap_info']:
            f.write("\nNodes assigned to multiple courses:\n")
            for node, info in sorted(analysis['overlap_info'].items()):
                f.write(f"  - {node} (courses: {info['courses']}, params: {info['params']:,})\n")
            f.write("\n")

        # Write validation information
        f.write("\nParameter Validation:\n")
        f.write(f"Total leaf node parameters across all courses: {analysis['validation_info']['leaf_params']:,}\n")
        f.write(f"Total number of leaf nodes: {len(analysis['validation_info']['all_leaf_nodes'])}\n")
        f.write(f"Total parameters across all courses: {analysis['total_params']:,}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Kitchen setup with multiple modes')
    
    # Add mutually exclusive arguments for different modes
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--state_dict_path', type=str,
                         help='Path to the model state dict file to generate hierarchy')
    mode_group.add_argument('--model_hierarchy_path', type=str,
                         help='Path to the annotated model hierarchy file to generate course analysis')
    mode_group.add_argument('--course_plating_path', type=str,
                         help='Path to the course analysis file to create serving folders')
    
    # Add root directory argument
    parser.add_argument('--root_dir', type=str, default='experiment/MNIST',
                      help='Root directory for all outputs')
    
    args = parser.parse_args()
    root_dir = Path(args.root_dir)
    root_dir.mkdir(parents=True, exist_ok=True)

    try:
        if args.state_dict_path:
            # Mode 1: Generate hierarchy from state dict
            print(f"Building hierarchy from state dict: {args.state_dict_path}")
            hierarchy, param_counts = build_model_hierarchy_from_state_dict(args.state_dict_path)
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"model_hierarchy_ResNet18_{timestamp}.txt"
            
            # Save hierarchy in the root directory
            save_model_hierarchy(hierarchy, param_counts, output_file, args.root_dir)
            
            print(f"\nHierarchy has been saved to: {root_dir / output_file}")
            print("\nPlease annotate the hierarchy file by:")
            print("1. Adding course indices in the brackets [ ]")
            print("2. Use empty brackets [ ] for nodes that should inherit from their parent")
            print("3. Group components logically (e.g., conv layers, batch norms, etc.)")
            print(f"\nTotal model parameters: {sum(count for count in param_counts.values()):,}")

        elif args.model_hierarchy_path:
            # Mode 2: Generate course analysis from annotated hierarchy
            print(f"Analyzing course assignments in: {args.model_hierarchy_path}")
            
            # Parse course indexing
            course_dict, void_nodes, overlap_nodes, empty_box_nodes = parse_user_course_indexing(args.model_hierarchy_path)
            
            # Analyze course parameters
            analysis = analyze_course_parameters(args.model_hierarchy_path, course_dict, void_nodes, overlap_nodes, empty_box_nodes)
            
            # Generate output filename based on input
            output_file = Path(args.model_hierarchy_path).name.replace("model_hierarchy", "course_analysis")
            output_path = root_dir / output_file
            
            # Save course analysis
            save_course_analysis(analysis, str(output_path))
            print(f"\nCourse analysis has been saved to: {output_path}")

        else:  # args.course_plating_path
            # Mode 3: Create course serving folders
            print(f"Creating course serving structure from: {args.course_plating_path}")
            
            # Create serving directories
            servings_dir = root_dir / "servings"
            servings_dir.mkdir(exist_ok=True)
            
            # Parse the course analysis file and create serving structure
            with open(args.course_plating_path, 'r') as f:
                for line in f:
                    if line.startswith("Course "):
                        # Extract course number and servings
                        course_match = re.match(r"Course (\d+)<(\d+)>:", line)
                        if course_match:
                            course_num = course_match.group(1)
                            num_servings = int(course_match.group(2))
                            
                            # Create course directory
                            course_dir = servings_dir / f"course_{course_num}"
                            course_dir.mkdir(exist_ok=True)
                            
                            # Create serving placeholders
                            for serving in range(1, num_servings + 1):
                                serving_file = course_dir / f"serving_{serving}.json"
                                with open(serving_file, 'w') as sf:
                                    json.dump({
                                        "course": int(course_num),
                                        "serving": serving,
                                        "status": "initialized"
                                    }, sf, indent=2)
            
            print(f"\nCourse serving structure created at: {servings_dir}")
            print("\nDirectory structure:")
            print(f"{servings_dir}/")
            for course_dir in sorted(servings_dir.glob("course_*")):
                print(f"├── {course_dir.name}/")
                for serving_file in sorted(course_dir.glob("serving_*.json")):
                    print(f"│   ├── {serving_file.name}")
            print("└── (end)")

    except Exception as e:
        print(f"Error: {str(e)}") 