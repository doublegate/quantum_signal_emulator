"""
Error handling and logging utilities for the Quantum Signal Emulator.

This module provides a standardized approach for handling errors, logging,
and reporting issues across the Quantum Signal Emulator codebase.
"""

import logging
import sys
import os
import traceback
import json
import datetime
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
from enum import Enum, auto
import threading
from functools import wraps
import inspect

# Configure base logger
logger = logging.getLogger("QuantumSignalEmulator")

class ErrorLevel(Enum):
    """Error severity levels."""
    DEBUG = auto()
    INFO = auto()
    WARNING = auto()
    ERROR = auto()
    CRITICAL = auto()

class ErrorCategory(Enum):
    """Categories of errors."""
    SYSTEM = auto()
    CONFIGURATION = auto()
    INPUT = auto()
    PROCESSING = auto()
    ANALYSIS = auto()
    HARDWARE = auto()
    INTEGRATION = auto()
    PERFORMANCE = auto()
    UNKNOWN = auto()

class ErrorHandler:
    """
    Centralized error handling and logging for the Quantum Signal Emulator.
    
    This class provides tools for capturing, processing, and responding to
    errors across the system, with support for different error levels,
    categories, and handling strategies.
    """
    
    def __init__(self, 
                log_file: Optional[str] = None,
                console_level: int = logging.INFO,
                file_level: int = logging.DEBUG,
                report_errors: bool = True,
                max_error_history: int = 100):
        """
        Initialize the error handler.
        
        Args:
            log_file: Path to log file (None for no file logging)
            console_level: Logging level for console output
            file_level: Logging level for file output
            report_errors: Whether to collect error reports
            max_error_history: Maximum number of errors to keep in history
        """
        self.log_file = log_file
        self.console_level = console_level
        self.file_level = file_level
        self.report_errors = report_errors
        self.max_error_history = max_error_history
        
        # Error history
        self.error_history = []
        self.error_history_lock = threading.Lock()
        
        # Error handlers by category
        self.error_handlers = {}
        
        # Configure logging
        self._configure_logging()
        
        logger.info("Error handler initialized")
    
    def _configure_logging(self) -> None:
        """Configure logging system."""
        # Reset handlers
        logger.handlers = []
        
        # Set global log level
        logger.setLevel(logging.DEBUG)
        
        # Create console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.console_level)
        console_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_format)
        logger.addHandler(console_handler)
        
        # Create file handler if specified
        if self.log_file:
            # Ensure directory exists
            log_dir = os.path.dirname(self.log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir)
                
            file_handler = logging.FileHandler(self.log_file)
            file_handler.setLevel(self.file_level)
            file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(file_format)
            logger.addHandler(file_handler)
    
    def handle_error(self, 
                  exception: Optional[Exception] = None,
                  message: Optional[str] = None,
                  level: ErrorLevel = ErrorLevel.ERROR,
                  category: ErrorCategory = ErrorCategory.UNKNOWN,
                  context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Handle an error.
        
        Args:
            exception: Exception object
            message: Error message
            level: Error severity level
            category: Error category
            context: Additional context
            
        Returns:
            Error information dictionary
        """
        # Build error info
        caller_info = self._get_caller_info()
        
        error_info = {
            "timestamp": datetime.datetime.now().isoformat(),
            "level": level.name,
            "category": category.name,
            "message": message or str(exception) if exception else "Unknown error",
            "exception_type": exception.__class__.__name__ if exception else None,
            "exception_args": exception.args if exception else None,
            "traceback": traceback.format_exc() if exception else None,
            "context": context or {},
            "caller": caller_info
        }
        
        # Log error
        log_level = getattr(logging, level.name)
        logger.log(log_level, f"{error_info['message']} ({category.name})")
        
        if exception and log_level >= logging.ERROR:
            logger.log(log_level, f"Exception: {exception.__class__.__name__}: {exception}")
            if log_level >= logging.DEBUG:
                logger.debug(f"Traceback: {error_info['traceback']}")
                
        # Add to history
        if self.report_errors:
            with self.error_history_lock:
                self.error_history.append(error_info)
                
                # Trim history if needed
                if len(self.error_history) > self.max_error_history:
                    self.error_history = self.error_history[-self.max_error_history:]
        
        # Call category handler if available
        handler = self.error_handlers.get(category)
        if handler:
            try:
                handler(error_info)
            except Exception as e:
                logger.error(f"Error in error handler: {e}")
        
        return error_info
    
    def _get_caller_info(self) -> Dict[str, Any]:
        """
        Get information about the caller.
        
        Returns:
            Dictionary with caller information
        """
        caller_info = {
            "file": None,
            "function": None,
            "line": None,
            "module": None
        }
        
        try:
            # Get call stack
            frame = inspect.currentframe()
            
            # Look for caller outside of error_handler.py
            while frame:
                frame_info = inspect.getframeinfo(frame)
                if 'error_handler.py' not in frame_info.filename:
                    caller_info["file"] = frame_info.filename
                    caller_info["function"] = frame_info.function
                    caller_info["line"] = frame_info.lineno
                    caller_info["module"] = inspect.getmodule(frame).__name__ if inspect.getmodule(frame) else None
                    break
                frame = frame.f_back
        except:
            # Ignore errors in getting caller info
            pass
            
        return caller_info
    
    def register_handler(self, category: ErrorCategory, handler: Callable[[Dict[str, Any]], None]) -> None:
        """
        Register a handler for a specific error category.
        
        Args:
            category: Error category
            handler: Handler function
        """
        self.error_handlers[category] = handler
        logger.debug(f"Registered handler for {category.name} errors")
    
    def unregister_handler(self, category: ErrorCategory) -> bool:
        """
        Unregister a handler for a specific error category.
        
        Args:
            category: Error category
            
        Returns:
            True if handler was removed, False if not found
        """
        if category in self.error_handlers:
            del self.error_handlers[category]
            logger.debug(f"Unregistered handler for {category.name} errors")
            return True
        return False
    
    def clear_error_history(self) -> None:
        """Clear the error history."""
        with self.error_history_lock:
            self.error_history = []
        logger.debug("Cleared error history")
    
    def get_error_history(self, 
                         level: Optional[ErrorLevel] = None,
                         category: Optional[ErrorCategory] = None,
                         max_errors: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get error history, optionally filtered.
        
        Args:
            level: Filter by error level
            category: Filter by error category
            max_errors: Maximum number of errors to return
            
        Returns:
            List of error dictionaries
        """
        with self.error_history_lock:
            # Start with full history
            errors = self.error_history.copy()
            
        # Apply filters
        if level:
            errors = [e for e in errors if e["level"] == level.name]
            
        if category:
            errors = [e for e in errors if e["category"] == category.name]
            
        # Apply limit
        if max_errors and max_errors < len(errors):
            errors = errors[-max_errors:]
            
        return errors
    
    def get_error_summary(self) -> Dict[str, Any]:
        """
        Get a summary of errors by category and level.
        
        Returns:
            Dictionary with error summary
        """
        with self.error_history_lock:
            errors = self.error_history.copy()
            
        # Count errors by category
        categories = {}
        for e in errors:
            category = e["category"]
            categories[category] = categories.get(category, 0) + 1
            
        # Count errors by level
        levels = {}
        for e in errors:
            level = e["level"]
            levels[level] = levels.get(level, 0) + 1
            
        # Count errors by exception type
        exceptions = {}
        for e in errors:
            exception_type = e.get("exception_type")
            if exception_type:
                exceptions[exception_type] = exceptions.get(exception_type, 0) + 1
                
        return {
            "total": len(errors),
            "by_category": categories,
            "by_level": levels,
            "by_exception": exceptions,
            "latest": errors[-1] if errors else None
        }
    
    def export_error_report(self, filename: str) -> bool:
        """
        Export error history to a file.
        
        Args:
            filename: Output filename
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create directory if needed
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            
            # Get error history
            with self.error_history_lock:
                errors = self.error_history.copy()
                
            # Create report
            report = {
                "timestamp": datetime.datetime.now().isoformat(),
                "summary": self.get_error_summary(),
                "errors": errors
            }
            
            # Export to JSON
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2)
                
            logger.info(f"Exported error report to {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting error report: {e}")
            return False
    
    def set_log_levels(self, console_level: int, file_level: Optional[int] = None) -> None:
        """
        Set logging levels.
        
        Args:
            console_level: Logging level for console output
            file_level: Logging level for file output (None to keep current)
        """
        self.console_level = console_level
        if file_level is not None:
            self.file_level = file_level
            
        # Update handlers
        for handler in logger.handlers:
            if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
                handler.setLevel(console_level)
            elif isinstance(handler, logging.FileHandler) and file_level is not None:
                handler.setLevel(file_level)
                
        logger.debug(f"Updated log levels: console={console_level}, file={file_level}")
    
    def set_log_file(self, log_file: Optional[str]) -> None:
        """
        Set log file.
        
        Args:
            log_file: Path to log file (None to disable file logging)
        """
        # Remove existing file handler
        for handler in list(logger.handlers):
            if isinstance(handler, logging.FileHandler):
                logger.removeHandler(handler)
                
        self.log_file = log_file
        
        # Add new file handler if specified
        if log_file:
            # Ensure directory exists
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir)
                
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(self.file_level)
            file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(file_format)
            logger.addHandler(file_handler)
            
            logger.debug(f"Set log file to {log_file}")
        else:
            logger.debug("Disabled file logging")
    
    def log_exception(self, exception: Exception, 
                    message: Optional[str] = None,
                    category: ErrorCategory = ErrorCategory.UNKNOWN,
                    context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Log an exception.
        
        Args:
            exception: Exception object
            message: Error message
            category: Error category
            context: Additional context
            
        Returns:
            Error information dictionary
        """
        return self.handle_error(
            exception=exception,
            message=message,
            level=ErrorLevel.ERROR,
            category=category,
            context=context
        )
    
    def log_error(self, message: str,
                category: ErrorCategory = ErrorCategory.UNKNOWN,
                context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Log an error message.
        
        Args:
            message: Error message
            category: Error category
            context: Additional context
            
        Returns:
            Error information dictionary
        """
        return self.handle_error(
            message=message,
            level=ErrorLevel.ERROR,
            category=category,
            context=context
        )
    
    def log_warning(self, message: str,
                  category: ErrorCategory = ErrorCategory.UNKNOWN,
                  context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Log a warning message.
        
        Args:
            message: Warning message
            category: Error category
            context: Additional context
            
        Returns:
            Error information dictionary
        """
        return self.handle_error(
            message=message,
            level=ErrorLevel.WARNING,
            category=category,
            context=context
        )
    
    def log_info(self, message: str,
               category: ErrorCategory = ErrorCategory.UNKNOWN,
               context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Log an info message.
        
        Args:
            message: Info message
            category: Error category
            context: Additional context
            
        Returns:
            Error information dictionary
        """
        return self.handle_error(
            message=message,
            level=ErrorLevel.INFO,
            category=category,
            context=context
        )
    
    def log_debug(self, message: str,
                category: ErrorCategory = ErrorCategory.UNKNOWN,
                context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Log a debug message.
        
        Args:
            message: Debug message
            category: Error category
            context: Additional context
            
        Returns:
            Error information dictionary
        """
        return self.handle_error(
            message=message,
            level=ErrorLevel.DEBUG,
            category=category,
            context=context
        )


# Global error handler instance
error_handler = ErrorHandler()

# Decorators

def error_boundary(category: ErrorCategory = ErrorCategory.UNKNOWN):
    """
    Decorator for catching and handling exceptions.
    
    Args:
        category: Error category
        
    Returns:
        Decorator function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Get function info
                context = {
                    "function": func.__name__,
                    "module": func.__module__,
                    "args": str(args),
                    "kwargs": str(kwargs)
                }
                
                # Handle error
                error_handler.log_exception(
                    exception=e,
                    message=f"Error in {func.__name__}: {e}",
                    category=category,
                    context=context
                )
                
                # Re-raise if critical
                if category == ErrorCategory.SYSTEM:
                    raise
                    
                # Return None for other categories
                return None
                
        return wrapper
    return decorator

def log_execution(level: ErrorLevel = ErrorLevel.DEBUG):
    """
    Decorator for logging function execution.
    
    Args:
        level: Log level
        
    Returns:
        Decorator function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Log function call
            log_level = getattr(logging, level.name)
            logger.log(log_level, f"Calling {func.__name__}")
            
            # Call function
            result = func(*args, **kwargs)
            
            # Log completion
            logger.log(log_level, f"Completed {func.__name__}")
            
            return result
            
        return wrapper
    return decorator

def performance_log(threshold_ms: int = 100):
    """
    Decorator for logging function performance.
    
    Args:
        threshold_ms: Log if execution takes longer than this
        
    Returns:
        Decorator function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Record start time
            start_time = datetime.datetime.now()
            
            # Call function
            result = func(*args, **kwargs)
            
            # Calculate execution time
            end_time = datetime.datetime.now()
            execution_time = (end_time - start_time).total_seconds() * 1000  # ms
            
            # Log if above threshold
            if execution_time > threshold_ms:
                logger.warning(f"Performance: {func.__name__} took {execution_time:.2f}ms "
                              f"(threshold: {threshold_ms}ms)")
                
            return result
            
        return wrapper
    return decorator