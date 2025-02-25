import json
from pathlib import Path

def read_kitchen_info(kitchen_path: str = "project/kitchen"):
    """Read and display kitchen information."""
    info_path = Path(kitchen_path) / "kitchen_info.json"
    
    with open(info_path, 'r') as f:
        info = json.load(f)
    
    print("Kitchen Info:")
    print(f"\nTotal Courses: {info['total_courses']}")
    print(f"Total Parameters: {info['total_parameters']:,}")
    
    print("\nModel Configuration:")
    for key, value in info['model_config'].items():
        print(f"  {key}: {value}")
    
    print("\nCourses:")
    for course_num, course_info in sorted(info['courses'].items()):
        print(f"\nCourse {course_num}:")
        print(f"  Number of Servings: {course_info['servings']}")
        print(f"  Nodes:")
        for node in course_info['nodes']:
            print(f"    - {node}")
        
        # Read and display course-specific info
        course_info_path = Path(kitchen_path) / f"course_{course_num}" / "info.json"
        with open(course_info_path, 'r') as f:
            detailed_info = json.load(f)
        print(f"  Total Parameters: {detailed_info['parameters']:,}")

if __name__ == "__main__":
    read_kitchen_info() 