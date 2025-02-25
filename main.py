import argparse
import yaml
from pathlib import Path

from src.models.kitchen_setup import kitchen_setup
from src.models.model_cook import model_cook
from src.training.fine_tuner import FineTuner
from src.utils.logger import setup_logging
from src.utils.helpers import load_config


def parse_args():
    parser = argparse.ArgumentParser(description='Fine-tuning workflow')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--menu', type=str, default=None,
                       help='Path to menu file for model cooking')
    return parser.parse_args()


def main():
    # Parse arguments and load configuration
    args = parse_args()
    config = load_config(args.config)
    
    # Setup logging
    logger = setup_logging(config['logging'])
    logger.info("Starting fine-tuning workflow")
    
    # Step 1: Kitchen Setup
    logger.info("Setting up kitchen (loading and analyzing model)")
    kitchen = kitchen_setup(
        model_path=config['model']['pretrained_path'],
        model_type=config['model']['model_type']
    )
    
    # Step 2: Model Cook
    logger.info("Cooking model according to menu")
    menu = load_config(args.menu) if args.menu else {}
    model_cooked = model_cook(menu=menu, kitchen=kitchen)
    
    # Step 3: Fine-tuning
    logger.info("Starting fine-tuning process")
    fine_tuner = FineTuner(
        model=model_cooked,
        config=config['training'],
        kitchen=kitchen
    )
    fine_tuner.train()
    
    logger.info("Fine-tuning workflow completed")


if __name__ == "__main__":
    main() 