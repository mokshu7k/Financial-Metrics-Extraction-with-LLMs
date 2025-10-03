"""Configuration loading and management utilities."""

import yaml
import os
from pathlib import Path
from typing import Any, Optional, Dict
import logging

logger = logging.getLogger(__name__)

class ConfigLoader:
    """
    Load and manage configuration from YAML file with environment variable support.
    """

    def __init__(self, config_path: str = "configs/config.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._resolve_env_variables()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            logger.warning(f"Config file not found: {self.config_path}. Using defaults")
            return self._get_default_config()
        
        try:
            with open(self.config_path, 'r'):
                config = yaml.safe_load(f)
            logger.infor(f"Loaded configuration from {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return self._get_default_config()
        

    def _get_default_config(self) -> Dict[str, Any]:
        """Return default configuration."""
        return {
            'paths': {
                'data_root': 'data',
                'raw_data': 'data/raw',
                'processed_data': 'data/processed',
                'embeddings': 'data/embeddings',
                'results': 'results'
            },
            'preprocessing': {
                'chunking': {
                    'default_strategy': 'fixed_size',
                    'chunk_size': 1000,
                    'chunk_overlap': 200
                }
            },
            'embeddings': {
                'model': {
                    'name': 'all-MiniLM-L6-v2',
                    'dimension': 384
                }
            }
        }
    
    def _resolve_env_variables(self):
        """Resolve environment variables in config values."""
        def resolve_dict(d: dict) -> dict:
            for key, value in d.items():
                if isinstance(value, str) and value.startswith('${') and value.endswith('}'):
                    env_var = value[2:-1]
                    d[key] = os.getenv(env_var, value)
                    
                elif isinstance(value, dict):
                    d[key] = resolve_dict(value)
            return d
        self.config = resolve_dict(self.config)


    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.

        Args:
            key_path: Path to config key using dots (e.g., 'paths.data_root')
            default: Default value if key not found

        Returns:
            Configuration value
        """

        keys = key_path.split('.')
        value = self.config

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
            
        return value
    
    def set(self, key_path:str, value: Any):
        """
        Set configuration value using dot notation.

        Args:
            key_path: Path to config key using dots
            value: Value to set
        """

        keys = key_path.split('.')
        config = self.config

        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]

        config[keys[-1]] = value

    def save(self, output_path: Optional[str] = None):
        """Save configuration to YAML file."""
        path = Path(output_path) if output_path else self.config_path

        with open(path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style = False, indent = 2)

        def to_dict(self) -> Dict[str, Any]:
            """Return configuration as dictionary."""
            return self.config.copy()
        
    def load_config(config_path: str = "configs/config.yaml") -> Dict[str,Any]:
        """
        Convenience function to load configuration.

        Args:
            config_path: Path to config file

        Returns:
            Configuration dictionary
        """

        loader = ConfigLoader(config_path)
        return loader.to_dict()
    