import pytest
import torch
import tempfile
from pathlib import Path
from monai.networks.nets import ViT

from src.models.tasting_menu import TastingMenu


@pytest.fixture
def sample_vit_model():
    """Create a sample ViT model for testing."""
    return ViT(
        in_channels=1,
        img_size=(96, 96, 96),
        patch_size=(16, 16, 16),
        hidden_size=768,
        mlp_dim=3072,
        num_layers=12,
        num_heads=12,
        proj_type="conv",
        classification=True,
        num_classes=2
    )


@pytest.fixture
def sample_course_analysis(tmp_path):
    """Create a sample course analysis file."""
    analysis_content = """Course Analysis:

Course 1<3>:
Total Parameters: 3,148,032
Nodes:
  - patch_embedding
  - patch_embedding.patch_embeddings
  - norm

Course 2<2>:
Total Parameters: 1,538
Nodes:
  - classification_head
  - classification_head.0

Course 3<4>:
Total Parameters: 21,256,704
Nodes:
  - blocks.0
  - blocks.0.attn
  - blocks.0.attn.out_proj
  - blocks.0.attn.qkv
  - blocks.0.mlp
  - blocks.0.mlp.linear1
  - blocks.0.mlp.linear2
  - blocks.0.norm1
  - blocks.0.norm2

Empty Box Nodes (all descendants are assigned or empty boxes):
  - blocks

Parameter Validation:
Total leaf node parameters across all courses: 24,406,274
Total number of leaf nodes: 12
Total parameters across all courses: 24,406,274
"""
    analysis_file = tmp_path / "course_analysis_test.txt"
    analysis_file.write_text(analysis_content)
    return str(analysis_file)


def test_tasting_menu_initialization(sample_vit_model, sample_course_analysis, tmp_path):
    """Test TastingMenu initialization and course loading."""
    servings_dir = tmp_path / "servings"
    menu = TastingMenu(sample_vit_model, sample_course_analysis, str(servings_dir))
    
    # Check courses were loaded correctly
    assert len(menu.courses) == 3
    assert menu.courses[1]['servings'] == 3
    assert menu.courses[2]['servings'] == 2
    assert menu.courses[3]['servings'] == 4
    
    # Check serving registry initialization
    assert len(menu.serving_registry) == 3
    assert menu.serving_registry[1]['total_servings'] == 3
    assert menu.serving_registry[2]['total_servings'] == 2
    assert menu.serving_registry[3]['total_servings'] == 4


def test_create_serving(sample_vit_model, sample_course_analysis, tmp_path):
    """Test creating individual servings."""
    servings_dir = tmp_path / "servings"
    menu = TastingMenu(sample_vit_model, sample_course_analysis, str(servings_dir))
    
    # Create a serving
    menu.create_serving(1, 1)
    
    # Check serving was created
    serving_path = servings_dir / "course_1_serving_1.pt"
    assert serving_path.exists()
    
    # Check serving registry was updated
    assert 1 in menu.serving_registry[1]['available_servings']
    
    # Check serving content
    serving_state = torch.load(serving_path)
    assert isinstance(serving_state, dict)
    assert all(key.startswith(('patch_embedding', 'norm')) for key in serving_state.keys())


def test_create_all_servings(sample_vit_model, sample_course_analysis, tmp_path):
    """Test creating all possible servings."""
    servings_dir = tmp_path / "servings"
    menu = TastingMenu(sample_vit_model, sample_course_analysis, str(servings_dir))
    
    # Create all servings
    menu.create_all_servings()
    
    # Check all servings were created
    expected_servings = {
        1: list(range(1, 4)),  # 3 servings
        2: list(range(1, 3)),  # 2 servings
        3: list(range(1, 5))   # 4 servings
    }
    
    for course, servings in expected_servings.items():
        for serving in servings:
            serving_path = servings_dir / f"course_{course}_serving_{serving}.pt"
            assert serving_path.exists()


def test_build_meal(sample_vit_model, sample_course_analysis, tmp_path):
    """Test building a meal from servings."""
    servings_dir = tmp_path / "servings"
    menu = TastingMenu(sample_vit_model, sample_course_analysis, str(servings_dir))
    
    # Create all servings
    menu.create_all_servings()
    
    # Build a meal
    meal_menu = {
        1: 2,  # Use serving 2 from course 1
        2: 1,  # Use serving 1 from course 2
        3: 3   # Use serving 3 from course 3
    }
    
    meal_state = menu.build_meal(meal_menu)
    
    # Check meal content
    assert isinstance(meal_state, dict)
    assert any(key.startswith('patch_embedding') for key in meal_state.keys())
    assert any(key.startswith('classification_head') for key in meal_state.keys())
    assert any(key.startswith('blocks.0') for key in meal_state.keys())


def test_save_meal(sample_vit_model, sample_course_analysis, tmp_path):
    """Test saving a complete meal."""
    servings_dir = tmp_path / "servings"
    menu = TastingMenu(sample_vit_model, sample_course_analysis, str(servings_dir))
    
    # Create all servings
    menu.create_all_servings()
    
    # Define and save a meal
    meal_menu = {1: 1, 2: 1, 3: 1}
    meal_path = tmp_path / "test_meal.pt"
    menu.save_meal(meal_menu, str(meal_path))
    
    # Check meal was saved
    assert meal_path.exists()
    
    # Load and verify meal
    meal_state = torch.load(meal_path)
    assert isinstance(meal_state, dict)
    assert len(meal_state) > 0


def test_error_handling(sample_vit_model, sample_course_analysis, tmp_path):
    """Test error handling in TastingMenu."""
    servings_dir = tmp_path / "servings"
    menu = TastingMenu(sample_vit_model, sample_course_analysis, str(servings_dir))
    
    # Test invalid course number
    with pytest.raises(ValueError, match="Course 99 not found"):
        menu.create_serving(99, 1)
    
    # Test invalid serving number
    with pytest.raises(ValueError, match="Serving 5 exceeds maximum servings"):
        menu.create_serving(1, 5)
    
    # Test building meal with unavailable serving
    with pytest.raises(ValueError, match="Serving 1 not available"):
        menu.build_meal({1: 1})  # Haven't created serving 1 yet


def test_cleanup(sample_vit_model, sample_course_analysis, tmp_path):
    """Test cleanup functionality."""
    servings_dir = tmp_path / "servings"
    menu = TastingMenu(sample_vit_model, sample_course_analysis, str(servings_dir))
    
    # Create some servings
    menu.create_all_servings()
    
    # Verify servings exist
    assert servings_dir.exists()
    assert len(list(servings_dir.glob("*.pt"))) > 0
    
    # Clean up
    menu.cleanup()
    
    # Verify cleanup
    assert not servings_dir.exists()
    assert len(menu.serving_registry[1]['available_servings']) == 0


def test_save_division_info(sample_vit_model, sample_course_analysis, tmp_path):
    """Test saving division information to a file."""
    servings_dir = tmp_path / "servings"
    menu = TastingMenu(sample_vit_model, sample_course_analysis, str(servings_dir))
    
    # Create some servings
    menu.create_serving(1, 1)
    menu.create_serving(2, 1)
    menu.create_serving(3, 1)
    
    # Save division info
    division_file = tmp_path / "division_info.txt"
    menu.save_division_info(str(division_file))
    
    # Verify file was created
    assert division_file.exists()
    
    # Check content
    content = division_file.read_text()
    assert "Course Division and Serving Information" in content
    assert "Total Model Parameters:" in content
    assert "Course 1:" in content
    assert "Course 2:" in content
    assert "Course 3:" in content
    assert "Available Servings: 1" in content
    assert "Example Menu Format:" in content 