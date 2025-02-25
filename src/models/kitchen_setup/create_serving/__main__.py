"""Main entry point for creating serving folders."""

import argparse
from pathlib import Path
from .creator import create_serving_folders


def main():
    parser = argparse.ArgumentParser(description='Create serving folders from course analysis')
    parser.add_argument('--course_analysis', type=str, required=True,
                      help='Path to the course analysis file')
    parser.add_argument('--out', type=str, required=True,
                      help='Root directory where servings will be created')
    parser.add_argument('--state_dict', type=str, required=True,
                      help='Path to the original state dict file')
    
    args = parser.parse_args()
    
    try:
        print(f"Creating course serving structure from: {args.course_analysis}")
        print(f"Using state dict from: {args.state_dict}")
        
        # Create serving folders
        serving_info = create_serving_folders(args.course_analysis, args.out, args.state_dict)
        
        # Print directory structure
        servings_dir = Path(serving_info['root_dir'])
        print(f"\nCourse serving structure created at: {servings_dir}")
        print("\nDirectory structure:")
        print(f"{servings_dir}/")
        for course_num, course_info in sorted(serving_info['courses'].items()):
            print(f"├── course_{course_num}/")
            for serving in course_info['servings']:
                state_dict_file = Path(serving['state_dict']).name
                print(f"│   ├── {state_dict_file}")
        print("├── serving_info.json")
        print("└── (end)")
        
        return serving_info
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return None


if __name__ == "__main__":
    main() 