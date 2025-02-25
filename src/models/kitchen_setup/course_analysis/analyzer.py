"""Analyzer module for course analysis."""

from collections import defaultdict
from typing import Dict, List, Any


def analyze_course_parameters(hierarchy_file: str, course_dict: Dict[int, List[str]], 
                           void_nodes: List[str], overlap_nodes: Dict[str, List[int]], 
                           empty_box_nodes: List[str]) -> Dict[str, Any]:
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