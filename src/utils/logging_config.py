# src/utils/logging_config.py
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

def setup_logging(
    log_level: str = "INFO",
    log_dir: str = "logs",
    log_to_file: bool = True,
    log_to_console: bool = True,
    log_format: Optional[str] = None
) -> logging.Logger:
    """
    Setup comprehensive logging for the ML pipeline
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory to store log files
        log_to_file: Whether to log to file
        log_to_console: Whether to log to console
        log_format: Custom log format string
    
    Returns:
        Configured logger instance
    """
    
    # Create logs directory if it doesn't exist
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Default log format
    if log_format is None:
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    
    # Create formatters
    formatter = logging.Formatter(log_format)
    
    # Get root logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()
    
    handlers = []
    
    # File handler
    if log_to_file:
        log_filename = f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_path = log_path / log_filename
        
        file_handler = logging.FileHandler(file_path, mode='a', encoding='utf-8')
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)
    
    # Console handler
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level.upper()))
        console_handler.setFormatter(formatter)
        handlers.append(console_handler)
    
    # Add handlers to logger
    for handler in handlers:
        logger.addHandler(handler)
    
    # Create pipeline-specific logger
    pipeline_logger = logging.getLogger("autonomous_ml_pipeline")
    pipeline_logger.info(f"Logging initialized. Level: {log_level}")
    
    if log_to_file:
        pipeline_logger.info(f"Log file: {file_path}")
    
    return pipeline_logger

def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name"""
    return logging.getLogger(f"autonomous_ml_pipeline.{name}")

def log_execution_time(func):
    """Decorator to log function execution time"""
    import functools
    import time
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        start_time = time.time()
        
        try:
            logger.info(f"Starting {func.__name__}")
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.info(f"Completed {func.__name__} in {execution_time:.2f} seconds")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Failed {func.__name__} after {execution_time:.2f} seconds: {str(e)}")
            raise
    
    return wrapper

def log_async_execution_time(func):
    """Decorator to log async function execution time"""
    import functools
    import time
    import asyncio
    
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        start_time = time.time()
        
        try:
            logger.info(f"Starting {func.__name__}")
            result = await func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.info(f"Completed {func.__name__} in {execution_time:.2f} seconds")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Failed {func.__name__} after {execution_time:.2f} seconds: {str(e)}")
            raise
    
    return wrapper

class PipelineLogger:
    """Context manager for pipeline step logging"""
    
    def __init__(self, step_name: str, logger: Optional[logging.Logger] = None):
        self.step_name = step_name
        self.logger = logger or get_logger("pipeline")
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        self.logger.info(f"=== Starting {self.step_name} ===")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = datetime.now() - self.start_time
        
        if exc_type is None:
            self.logger.info(f"=== Completed {self.step_name} in {duration.total_seconds():.2f} seconds ===")
        else:
            self.logger.error(f"=== Failed {self.step_name} after {duration.total_seconds():.2f} seconds ===")
            self.logger.error(f"Error: {exc_val}")
    
    def log_progress(self, message: str):
        """Log progress within the step"""
        self.logger.info(f"[{self.step_name}] {message}")
    
    def log_metric(self, name: str, value: Union[int, float, str]):
        """Log a metric within the step"""
        self.logger.info(f"[{self.step_name}] Metric - {name}: {value}")

# Example usage functions
def configure_mlflow_logging():
    """Configure MLflow logging integration"""
    try:
        import mlflow
        # Disable MLflow's own logging to avoid conflicts
        logging.getLogger("mlflow").setLevel(logging.WARNING)
        logging.getLogger("mlflow.tracking").setLevel(logging.WARNING)
        logging.getLogger("mlflow.utils").setLevel(logging.WARNING)
    except ImportError:
        pass

def configure_third_party_logging():
    """Configure third-party library logging levels"""
    # Reduce noise from third-party libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("numba").setLevel(logging.WARNING)
    
    # Configure scikit-learn logging
    import warnings
    from sklearn.utils._testing import ignore_warnings
    warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn')

# Initialize logging when module is imported
def initialize_default_logging():
    """Initialize default logging configuration"""
    setup_logging()
    configure_mlflow_logging()
    configure_third_party_logging()

# Auto-initialize when imported
if __name__ != "__main__":
    try:
        initialize_default_logging()
    except Exception as e:
        print(f"Warning: Could not initialize logging: {e}")