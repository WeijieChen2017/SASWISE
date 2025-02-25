"""Main entry point for course analysis."""

import argparse
from pathlib import Path
from .parser import parse_user_course_indexing
from .analyzer import analyze_course_parameters
from .saver import save_course_analysis


def main():
    parser = argparse.ArgumentParser(description='Generate course analysis from annotated hierarchy')
    parser.add_argument('--model_hierarchy', type=str, required=True,
                      help='Path to the annotated model hierarchy file')
    parser.add_argument('--out', type=str, required=True,
                      help='Directory where the course analysis will be saved')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        print(f"Analyzing course assignments in: {args.model_hierarchy}")
        
        # Parse course indexing from hierarchy file
        course_dict, void_nodes, overlap_nodes, empty_box_nodes = parse_user_course_indexing(args.model_hierarchy)
        
        # Analyze course parameters
        analysis = analyze_course_parameters(args.model_hierarchy, course_dict, void_nodes, overlap_nodes, empty_box_nodes)
        
        # Generate output filename based on input
        input_name = Path(args.model_hierarchy).name
        output_file = input_name.replace("model_hierarchy", "course_analysis")
        output_path = output_dir / output_file
        
        # Save course analysis
        save_course_analysis(analysis, str(output_path))
        print(f"\nCourse analysis has been saved to: {output_path}")
        
        # Return the output path for potential chaining
        return str(output_path)
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return None


if __name__ == "__main__":
    main() 