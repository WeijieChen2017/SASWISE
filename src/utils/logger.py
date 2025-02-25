import logging
from pathlib import Path
import time
from typing import Optional
from torch.utils.tensorboard import SummaryWriter


class TrainingLogger:
    """Class for handling training logs and metrics."""
    
    def __init__(
        self,
        log_dir: str = 'logs',
        experiment_name: Optional[str] = None,
        use_tensorboard: bool = True
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Setup experiment name
        self.experiment_name = experiment_name or f'experiment_{int(time.time())}'
        
        # Setup file logger
        self.logger = self._setup_file_logger()
        
        # Setup tensorboard if requested
        self.writer = None
        if use_tensorboard:
            tensorboard_dir = self.log_dir / 'tensorboard' / self.experiment_name
            tensorboard_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(tensorboard_dir)
        
        self.current_epoch = 0
        self.start_time = None
        self.epoch_start_time = None
    
    def _setup_file_logger(self) -> logging.Logger:
        """Setup the file logger."""
        logger = logging.getLogger(self.experiment_name)
        logger.setLevel(logging.INFO)
        
        # Create file handler
        fh = logging.FileHandler(
            self.log_dir / f'{self.experiment_name}.log'
        )
        fh.setLevel(logging.INFO)
        
        # Create console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        # Add handlers
        logger.addHandler(fh)
        logger.addHandler(ch)
        
        return logger
    
    def start_training(self, total_epochs: int):
        """Log the start of training."""
        self.start_time = time.time()
        self.logger.info(f'Starting training for {total_epochs} epochs')
    
    def end_training(self):
        """Log the end of training."""
        duration = time.time() - self.start_time
        self.logger.info(
            f'Training completed in {duration:.2f} seconds'
        )
        
        if self.writer:
            self.writer.close()
    
    def start_epoch(self, epoch: int):
        """Log the start of an epoch."""
        self.current_epoch = epoch
        self.epoch_start_time = time.time()
        self.logger.info(f'Starting epoch {epoch}')
    
    def end_epoch(self):
        """Log the end of an epoch."""
        duration = time.time() - self.epoch_start_time
        self.logger.info(
            f'Epoch {self.current_epoch} completed in {duration:.2f} seconds'
        )
    
    def log_step(self, step: int, loss: float):
        """Log a training step."""
        self.logger.info(
            f'Epoch {self.current_epoch}, Step {step}: loss = {loss:.4f}'
        )
        
        if self.writer:
            self.writer.add_scalar(
                'training/step_loss',
                loss,
                step + self.current_epoch * step
            )
    
    def log_epoch_train(self, avg_loss: float):
        """Log training metrics for an epoch."""
        self.logger.info(
            f'Epoch {self.current_epoch} - Average training loss: {avg_loss:.4f}'
        )
        
        if self.writer:
            self.writer.add_scalar(
                'training/epoch_loss',
                avg_loss,
                self.current_epoch
            )
    
    def log_epoch_validation(self, avg_loss: float):
        """Log validation metrics for an epoch."""
        self.logger.info(
            f'Epoch {self.current_epoch} - Validation loss: {avg_loss:.4f}'
        )
        
        if self.writer:
            self.writer.add_scalar(
                'validation/epoch_loss',
                avg_loss,
                self.current_epoch
            )
    
    def log_info(self, message: str):
        """Log an informational message."""
        self.logger.info(message)
    
    def log_warning(self, message: str):
        """Log a warning message."""
        self.logger.warning(message)
    
    def log_error(self, message: str):
        """Log an error message."""
        self.logger.error(message) 