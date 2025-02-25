"""Kitchen Setup Package for Model Analysis and Course Management.

This package provides tools for analyzing neural network models and organizing them into courses:

1. Generate Hierarchy:
   python -m src.models.kitchen_setup.generate_hierarchy --state_dict <path> --out <dir>
   - Creates a hierarchical view of model layers for annotation

2. Course Analysis:
   python -m src.models.kitchen_setup.course_analysis --model_hierarchy <path> --out <dir>
   - Analyzes annotated hierarchy and generates course statistics

3. Create Servings:
   python -m src.models.kitchen_setup.create_serving --course_analysis <path> --state_dict <path> --out <dir>
   - Creates serving folders with course-specific state dicts
"""

from .hierarchy.build_hierarchy import build_model_hierarchy_from_state_dict
from .hierarchy.save_hierarchy import save_model_hierarchy

__all__ = [
    'build_model_hierarchy_from_state_dict',
    'save_model_hierarchy',
] 