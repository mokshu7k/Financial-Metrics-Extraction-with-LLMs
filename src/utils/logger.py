import logging 
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional
import os

def setup_logging(
        level:str = "INFO",
        log_file: Optional[str] = None,
        console : bool = True,
        format_string: Optional[str] = None
) -> logging.Logger:
    """
    Setup Logging Configuration for the entire pipeline.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to the log file(optional)
        console: Whether to output to console
        format_string: Custom format string for log messages

    Returns:
        Configured root logger
    """
    # Convert string level to logging constant
    # getattr(obj, name, default)
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    formatter = logging.Formatter(format_string)

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Remove existing handlers
    root_logger.handlers = []

    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(format_string)
        root_logger.addHandler(console_handler)

    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, mode= 'a', encoding='utf-8')
        file_handler.setFormatter(format_string)
        file_handler.setLevel(numeric_level)
        root_logger.addHandler(file_handler)

    return root_logger

def get_logger(name: str, level:Optional[str] = None) -> logging.Logger:
    """
    Get a logger with a specific name.

    Args:
        name: Name of the logger(usually __name__)
        level: Optional level to set for this logger

    Returns:
        Configured logger
    """

    logger = logging.getLogger(name)

    if level:
        numeric_level = getattr(logging, level.upper(), logging.INFO)
        logger.setLevel(numeric_level)

    return logger

class ExperimentLogger:
    """Logger specifically for tracking experiment runs."""

    def __init__(self, experiment_name: str, log_dir: str = "results/logs"):
        self.experiment_name = experiment_name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Create experiment-specific log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir /f"{experiment_name}_{timestamp}.log"
        
        self.logger = logging.getLogger(f"experiment.{experiment_name}")
        self.logger.setLevel(logging.DEBUG)

        # File handler for this experiment
        handler = logging.FileHandler(log_file)
        handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        )
        self.logger.addHandler(handler)

        self.logger.info(f"Started experiment: {experiment_name}")

    def log_parameters(self, params: dict):
        """Log experiment parameters."""
        self.logger.info("Experiment Parameters:")
        for key, value, in params.items():
            self.logger.info(f"  {key}: {value}")

    def log_metrics(self, metrics: dict, step: Optional[int] = None):
        """Log experiment metrics."""
        step_str = f" (Step {step})" if step is not None else ""
        self.logger.info(f"Metrics{step_str}:")
        for key, value in metrics.items():
            self.logger.info(f"  {key}: {value}")
    
    def log_error(self, error: Exception):
        """Log an error with full traceback."""
        self.logger.error(f"Error occurred: {str(error)}", exc_info=True)