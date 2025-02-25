"""Parser module for course analysis."""

from collections import defaultdict
from typing import Dict, List, Tuple, Set


def parse_user_course_indexing(hierarchy_file: str) -> Tuple[Dict[int, List[str]], List[str], Dict[str, List[int]], List[str]]:
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