"""
Configuration management for the Quantum Signal Emulator.

This module provides tools for loading, validating, and managing configuration settings
for the Quantum Signal Emulator. It supports multiple configuration formats and includes
schema validation to ensure configuration correctness.
"""

import os
import json
import logging
import copy
from typing import Dict, Any, Optional, List, Set, Union
import yaml

from ..constants import ANALYSIS_MODES, DEFAULT_NUM_QUBITS
from ..system_configs import SYSTEM_CONFIGS

logger = logging.getLogger("QuantumSignalEmulator.ConfigManager")

class ConfigManager:
    """
    Configuration management for the Quantum Signal Emulator.
    
    This class handles loading, validating, and providing access to configuration
    settings for the emulator. It supports JSON and YAML formats, and includes
    schema validation to ensure configuration correctness.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Path to configuration file (None for default values)
        """
        # Default configuration
        self.defaults = {
            "system": "nes",
            "analysis_mode": "hybrid",
            "num_qubits": DEFAULT_NUM_QUBITS,
            "gpu_enabled": True,
            "visualization": {
                "enabled": True,
                "use_3d": True,
                "dark_mode": True,
                "auto_save": False,
                "output_dir": "./output"
            },
            "logging": {
                "level": "INFO",
                "file": None,
                "console": True
            },
            "performance": {
                "multi_threading": True,
                "thread_count": 4,
                "batch_size": 100
            },
            "analysis": {
                "cycle_precision": 0.1,
                "enable_quantum": True,
                "enable_machine_learning": True,
                "save_results": True,
                "results_format": "json"
            }
        }
        
        # Current configuration (copy of defaults initially)
        self.config = copy.deepcopy(self.defaults)
        
        # Set of keys that have been modified from defaults
        self.modified_keys = set()
        
        # Load configuration file if provided
        if config_path:
            self.load_config(config_path)
            
        logger.debug("ConfigManager initialized")
    
    def load_config(self, config_path: str) -> bool:
        """
        Load configuration from file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            True if configuration loaded successfully, False otherwise
        """
        try:
            if not os.path.exists(config_path):
                logger.error(f"Configuration file not found: {config_path}")
                return False
                
            # Determine file format based on extension
            _, ext = os.path.splitext(config_path)
            ext = ext.lower()
            
            if ext == '.json':
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
            elif ext in ['.yaml', '.yml']:
                with open(config_path, 'r') as f:
                    user_config = yaml.safe_load(f)
            else:
                logger.error(f"Unsupported configuration format: {ext}")
                return False
                
            # Validate configuration
            validation_errors = self.validate_config(user_config)
            if validation_errors:
                for error in validation_errors:
                    logger.error(f"Configuration validation error: {error}")
                return False
                
            # Merge with defaults (deep merge)
            self._merge_config(user_config)
            
            logger.info(f"Configuration loaded from {config_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            return False
    
    def _merge_config(self, user_config: Dict[str, Any], path: str = "") -> None:
        """
        Merge user configuration with defaults, tracking modified keys.
        
        Args:
            user_config: User configuration dictionary
            path: Current key path for tracking (internal use)
        """
        for key, value in user_config.items():
            current_path = f"{path}.{key}" if path else key
            
            if isinstance(value, dict) and key in self.config and isinstance(self.config[key], dict):
                # Recursively merge nested dictionaries
                self._merge_config(value, current_path)
            else:
                # Set value and track modification
                self.config[key] = value
                self.modified_keys.add(current_path)
    
    def validate_config(self, config: Dict[str, Any]) -> List[str]:
        """
        Validate configuration against schema.
        
        Args:
            config: Configuration dictionary to validate
            
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        # Check system type
        if "system" in config and config["system"] not in SYSTEM_CONFIGS:
            valid_systems = ", ".join(SYSTEM_CONFIGS.keys())
            errors.append(f"Invalid system type: {config['system']}. Valid options: {valid_systems}")
            
        # Check analysis mode
        if "analysis_mode" in config and config["analysis_mode"] not in ANALYSIS_MODES:
            valid_modes = ", ".join(ANALYSIS_MODES)
            errors.append(f"Invalid analysis mode: {config['analysis_mode']}. Valid options: {valid_modes}")
            
        # Check numeric values
        if "num_qubits" in config:
            if not isinstance(config["num_qubits"], int) or config["num_qubits"] < 2 or config["num_qubits"] > 20:
                errors.append(f"Invalid num_qubits: {config['num_qubits']}. Must be an integer between 2 and 20")
                
        # Check boolean values
        for key in ["gpu_enabled"]:
            if key in config and not isinstance(config[key], bool):
                errors.append(f"Invalid {key}: {config[key]}. Must be a boolean")
                
        # Check nested configuration
        if "visualization" in config:
            vis_config = config["visualization"]
            
            if "enabled" in vis_config and not isinstance(vis_config["enabled"], bool):
                errors.append(f"Invalid visualization.enabled: {vis_config['enabled']}. Must be a boolean")
                
            if "use_3d" in vis_config and not isinstance(vis_config["use_3d"], bool):
                errors.append(f"Invalid visualization.use_3d: {vis_config['use_3d']}. Must be a boolean")
                
        # Check logging configuration
        if "logging" in config:
            log_config = config["logging"]
            
            if "level" in log_config and log_config["level"] not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
                valid_levels = "DEBUG, INFO, WARNING, ERROR, CRITICAL"
                errors.append(f"Invalid logging.level: {log_config['level']}. Valid options: {valid_levels}")
                
        # Check performance configuration
        if "performance" in config:
            perf_config = config["performance"]
            
            if "thread_count" in perf_config:
                if not isinstance(perf_config["thread_count"], int) or perf_config["thread_count"] < 1:
                    errors.append(f"Invalid performance.thread_count: {perf_config['thread_count']}. "
                               f"Must be a positive integer")
                    
        # Check analysis configuration
        if "analysis" in config:
            analysis_config = config["analysis"]
            
            if "results_format" in analysis_config and analysis_config["results_format"] not in ["json", "csv", "binary"]:
                valid_formats = "json, csv, binary"
                errors.append(f"Invalid analysis.results_format: {analysis_config['results_format']}. "
                           f"Valid options: {valid_formats}")
                
        return errors
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key path.
        
        Args:
            key: Configuration key path (e.g., 'visualization.use_3d')
            default: Default value if key not found
            
        Returns:
            Configuration value or default if not found
        """
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value by key path.
        
        Args:
            key: Configuration key path (e.g., 'visualization.use_3d')
            value: Value to set
        """
        keys = key.split('.')
        config = self.config
        
        # Navigate to the nested dictionary
        for k in keys[:-1]:
            if k not in config or not isinstance(config[k], dict):
                config[k] = {}
            config = config[k]
            
        # Set the value
        config[keys[-1]] = value
        
        # Track modification
        self.modified_keys.add(key)
        
        logger.debug(f"Configuration updated: {key} = {value}")
    
    def reset(self, key: Optional[str] = None) -> None:
        """
        Reset configuration to defaults.
        
        Args:
            key: Key path to reset (None for all)
        """
        if key is None:
            # Reset all
            self.config = copy.deepcopy(self.defaults)
            self.modified_keys.clear()
            logger.info("Configuration reset to defaults")
        else:
            # Reset specific key
            keys = key.split('.')
            default_value = self._get_default_value(keys)
            
            config = self.config
            for k in keys[:-1]:
                if k not in config:
                    return
                config = config[k]
                
            config[keys[-1]] = default_value
            
            # Remove from modified keys
            self.modified_keys.discard(key)
            logger.info(f"Configuration key reset to default: {key}")
    
    def _get_default_value(self, keys: List[str]) -> Any:
        """
        Get default value for a key path.
        
        Args:
            keys: List of keys in the path
            
        Returns:
            Default value
        """
        value = self.defaults
        for k in keys:
            if k not in value:
                return None
            value = value[k]
        return copy.deepcopy(value)
    
    def save_config(self, config_path: str, format: str = 'json') -> bool:
        """
        Save current configuration to file.
        
        Args:
            config_path: Path to output file
            format: Output format ('json' or 'yaml')
            
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            # Create directory if it doesn't exist
            directory = os.path.dirname(config_path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)
                
            if format.lower() == 'json':
                with open(config_path, 'w') as f:
                    json.dump(self.config, f, indent=2)
            elif format.lower() in ['yaml', 'yml']:
                with open(config_path, 'w') as f:
                    yaml.dump(self.config, f, default_flow_style=False)
            else:
                logger.error(f"Unsupported configuration format: {format}")
                return False
                
            logger.info(f"Configuration saved to {config_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            return False
    
    def get_modified_config(self) -> Dict[str, Any]:
        """
        Get a dictionary containing only modified configuration values.
        
        Returns:
            Dictionary with modified values
        """
        modified_config = {}
        
        for key in self.modified_keys:
            value = self.get(key)
            
            # Build nested dictionaries
            keys = key.split('.')
            current = modified_config
            
            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]
                
            current[keys[-1]] = value
            
        return modified_config
    
    def load_from_dict(self, config_dict: Dict[str, Any]) -> bool:
        """
        Load configuration from dictionary.
        
        Args:
            config_dict: Configuration dictionary
            
        Returns:
            True if loaded successfully, False otherwise
        """
        # Validate configuration
        validation_errors = self.validate_config(config_dict)
        if validation_errors:
            for error in validation_errors:
                logger.error(f"Configuration validation error: {error}")
            return False
            
        # Merge with defaults
        self._merge_config(config_dict)
        
        logger.debug("Configuration loaded from dictionary")
        return True
    
    def get_system_config(self) -> Dict[str, Any]:
        """
        Get system-specific configuration.
        
        Returns:
            System configuration dictionary
        """
        system_type = self.get("system", "nes")
        return SYSTEM_CONFIGS.get(system_type, {})
    
    def as_dict(self) -> Dict[str, Any]:
        """
        Get full configuration as dictionary.
        
        Returns:
            Configuration dictionary
        """
        return copy.deepcopy(self.config)