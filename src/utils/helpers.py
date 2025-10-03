"""General helper utilities for the RAG pipeline."""

import time
import json
import pickle
import hashlib
import functools
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# ============================================================================
# Timing and Performance
# ============================================================================

class timer:
    """Context manager and decorator for timing code execution."""
    
    def __init__(self, name: str = "Operation", log_result: bool = True):
        self.name = name
        self.log_result = log_result
        self.start_time = None
        self.end_time = None
        self.elapsed = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args):
        self.end_time = time.time()
        self.elapsed = self.end_time - self.start_time
        
        if self.log_result:
            logger.info(f"{self.name} took {self.elapsed:.2f} seconds")
    
    def __call__(self, func: Callable) -> Callable:
        """Use as decorator."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with timer(f"{func.__name__}", self.log_result):
                return func(*args, **kwargs)
        return wrapper

# ============================================================================
# Caching
# ============================================================================

def cache_result(
    cache_dir: str = ".cache",
    cache_key_func: Optional[Callable] = None,
    ttl_seconds: Optional[int] = None
):
    """
    Decorator to cache function results to disk.
    
    Args:
        cache_dir: Directory to store cache files
        cache_key_func: Function to generate cache key from args
        ttl_seconds: Time-to-live for cache in seconds
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            if cache_key_func:
                cache_key = cache_key_func(*args, **kwargs)
            else:
                # Default: hash function name and arguments
                key_data = f"{func.__name__}_{str(args)}_{str(kwargs)}"
                cache_key = hashlib.md5(key_data.encode()).hexdigest()
            
            cache_file = cache_path / f"{cache_key}.pkl"
            
            # Check if cache exists and is valid
            if cache_file.exists():
                if ttl_seconds:
                    age = time.time() - cache_file.stat().st_mtime
                    if age > ttl_seconds:
                        cache_file.unlink()
                    else:
                        with open(cache_file, 'rb') as f:
                            logger.debug(f"Loading cached result for {func.__name__}")
                            return pickle.load(f)
                else:
                    with open(cache_file, 'rb') as f:
                        logger.debug(f"Loading cached result for {func.__name__}")
                        return pickle.load(f)
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            
            with open(cache_file, 'wb') as f:
                pickle.dump(result, f)
            
            return result
        
        return wrapper
    return decorator

# ============================================================================
# Retry Logic
# ============================================================================

def retry_on_failure(
    max_attempts: int = 3,
    delay_seconds: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: tuple = (Exception,)
):
    """
    Decorator to retry function on failure with exponential backoff.
    
    Args:
        max_attempts: Maximum number of retry attempts
        delay_seconds: Initial delay between retries
        backoff_factor: Multiplier for delay after each retry
        exceptions: Tuple of exceptions to catch
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            attempt = 0
            current_delay = delay_seconds
            
            while attempt < max_attempts:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    attempt += 1
                    if attempt >= max_attempts:
                        logger.error(f"{func.__name__} failed after {max_attempts} attempts")
                        raise
                    
                    logger.warning(
                        f"{func.__name__} failed (attempt {attempt}/{max_attempts}): {e}. "
                        f"Retrying in {current_delay:.1f}s..."
                    )
                    time.sleep(current_delay)
                    current_delay *= backoff_factor
            
            return None
        
        return wrapper
    return decorator

# ============================================================================
# Token and Cost Estimation
# ============================================================================

def estimate_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    """
    Estimate token count for a given text.
    
    Args:
        text: Input text
        model: Model name for tokenization rules
    
    Returns:
        Estimated token count
    """
    # Simple estimation: ~4 characters per token for English
    # More accurate would use tiktoken library
    return len(text) // 4

def calculate_cost(
    prompt_tokens: int,
    completion_tokens: int,
    model: str = "gpt-3.5-turbo"
) -> float:
    """
    Calculate cost for API call based on token usage.
    
    Args:
        prompt_tokens: Number of prompt tokens
        completion_tokens: Number of completion tokens
        model: Model name
    
    Returns:
        Estimated cost in USD
    """
    # Pricing as of 2024 (update as needed)
    pricing = {
        'gpt-3.5-turbo': {'prompt': 0.0015 / 1000, 'completion': 0.002 / 1000},
        'gpt-4': {'prompt': 0.03 / 1000, 'completion': 0.06 / 1000},
        'gpt-4-turbo': {'prompt': 0.01 / 1000, 'completion': 0.03 / 1000},
        'claude-3-sonnet': {'prompt': 0.003 / 1000, 'completion': 0.015 / 1000},
        'claude-3-opus': {'prompt': 0.015 / 1000, 'completion': 0.075 / 1000}
    }
    
    if model not in pricing:
        logger.warning(f"Unknown model {model}, using default pricing")
        model = 'gpt-3.5-turbo'
    
    prompt_cost = prompt_tokens * pricing[model]['prompt']
    completion_cost = completion_tokens * pricing[model]['completion']
    
    return prompt_cost + completion_cost

# ============================================================================
# File I/O Utilities
# ============================================================================

def safe_json_load(file_path: Union[str, Path], default: Any = None) -> Any:
    """
    Safely load JSON file with error handling.
    
    Args:
        file_path: Path to JSON file
        default: Default value if file doesn't exist or is invalid
    
    Returns:
        Loaded data or default value
    """
    path = Path(file_path)
    
    if not path.exists():
        logger.warning(f"File not found: {path}")
        return default
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in {path}: {e}")
        return default
    except Exception as e:
        logger.error(f"Error loading {path}: {e}")
        return default

def safe_json_save(data: Any, file_path: Union[str, Path], indent: int = 2):
    """
    Safely save data to JSON file.
    
    Args:
        data: Data to save
        file_path: Path to output file
        indent: JSON indentation
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
        logger.debug(f"Saved data to {path}")
    except Exception as e:
        logger.error(f"Error saving to {path}: {e}")
        raise

def ensure_directory(path: Union[str, Path]) -> Path:
    """
    Ensure directory exists, create if it doesn't.
    
    Args:
        path: Directory path
    
    Returns:
        Path object
    """
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path

# ============================================================================
# Formatting Utilities
# ============================================================================

def get_timestamp(format: str = "%Y%m%d_%H%M%S") -> str:
    """
    Get current timestamp as formatted string.
    
    Args:
        format: Timestamp format string
    
    Returns:
        Formatted timestamp
    """
    return datetime.now().strftime(format)

def format_metric_name(name: str) -> str:
    """
    Format metric name for display (convert snake_case to Title Case).
    
    Args:
        name: Metric name in snake_case
    
    Returns:
        Formatted name
    """
    return name.replace('_', ' ').title()

def format_number(value: float, precision: int = 2, suffix: str = "") -> str:
    """
    Format number with optional precision and suffix.
    
    Args:
        value: Number to format
        precision: Decimal places
        suffix: Optional suffix (e.g., '%', 'M')
    
    Returns:
        Formatted string
    """
    if isinstance(value, int):
        return f"{value:,}{suffix}"
    return f"{value:,.{precision}f}{suffix}"

# ============================================================================
# Data Structure Utilities
# ============================================================================

def flatten_dict(d: Dict, parent_key: str = '', sep: str = '.') -> Dict:
    """
    Flatten nested dictionary with dot notation keys.
    
    Args:
        d: Dictionary to flatten
        parent_key: Parent key for recursion
        sep: Separator for nested keys
    
    Returns:
        Flattened dictionary
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def unflatten_dict(d: Dict, sep: str = '.') -> Dict:
    """
    Unflatten dictionary with dot notation keys.
    
    Args:
        d: Flattened dictionary
        sep: Separator used in keys
    
    Returns:
        Nested dictionary
    """
    result = {}
    for key, value in d.items():
        parts = key.split(sep)
        current = result
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[parts[-1]] = value
    return result