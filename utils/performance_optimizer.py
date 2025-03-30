"""
Performance optimization utilities for the Quantum Signal Emulator.

This module provides tools for optimizing performance of the emulator,
including multithreading, GPU acceleration, and memory management.
"""

import logging
import os
import time
import threading
from typing import Dict, List, Tuple, Any, Callable, Optional, Union
import multiprocessing
from contextlib import contextmanager
from functools import wraps, lru_cache
import sys
import gc
import psutil
import numpy as np

# Try to import GPU acceleration libraries
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

logger = logging.getLogger("QuantumSignalEmulator.PerformanceOptimizer")

class PerformanceOptimizer:
    """
    Performance optimization tools for the Quantum Signal Emulator.
    
    This class provides utilities for optimizing emulator performance,
    including parallel processing, memory management, and GPU acceleration.
    """
    
    def __init__(self, gpu_enabled: bool = True, 
                multithreading_enabled: bool = True,
                memory_limit_mb: Optional[int] = None,
                cache_size: int = 128):
        """
        Initialize the performance optimizer.
        
        Args:
            gpu_enabled: Whether to use GPU acceleration if available
            multithreading_enabled: Whether to use multithreading
            memory_limit_mb: Memory limit in MB (None for no limit)
            cache_size: Size of LRU cache for memoization
        """
        self.gpu_enabled = gpu_enabled and CUPY_AVAILABLE
        self.multithreading_enabled = multithreading_enabled
        self.memory_limit_mb = memory_limit_mb
        self.cache_size = cache_size
        
        # Initialize state
        self.thread_pool = None
        self.process_pool = None
        self.memory_tracker = None
        self.current_device = None
        
        # Performance metrics
        self.metrics = {
            "execution_times": {},
            "memory_usage": [],
            "gpu_usage": [],
            "thread_counts": []
        }
        
        # Initialize subsystems
        self._initialize_threading()
        self._initialize_gpu()
        self._initialize_memory_tracking()
        
        logger.info(f"Performance optimizer initialized (GPU: {self.gpu_enabled}, "
                  f"Multithreading: {self.multithreading_enabled})")
    
    def _initialize_threading(self) -> None:
        """Initialize threading and multiprocessing pools."""
        if self.multithreading_enabled:
            # Determine optimal thread count
            cpu_count = os.cpu_count() or 4
            self.optimal_thread_count = max(1, cpu_count - 1)  # Leave one CPU for system
            
            # Create thread pool
            self.thread_pool = ThreadPool(self.optimal_thread_count)
            
            # Create process pool for CPU-bound tasks
            try:
                self.process_pool = multiprocessing.Pool(processes=self.optimal_thread_count // 2 or 1)
            except:
                logger.warning("Failed to create multiprocessing pool")
                self.process_pool = None
                
            logger.debug(f"Initialized thread pool with {self.optimal_thread_count} threads")
    
    def _initialize_gpu(self) -> None:
        """Initialize GPU acceleration if available."""
        if self.gpu_enabled and CUPY_AVAILABLE:
            try:
                # Get GPU device count
                device_count = cp.cuda.runtime.getDeviceCount()
                
                if device_count > 0:
                    # Use the first device by default
                    cp.cuda.runtime.setDevice(0)
                    self.current_device = 0
                    
                    # Get device properties
                    device_props = cp.cuda.runtime.getDeviceProperties(0)
                    logger.info(f"Using GPU: {device_props['name'].decode()}")
                    logger.info(f"GPU Memory: {device_props['totalGlobalMem'] / (1024**3):.2f} GB")
                else:
                    logger.warning("No CUDA devices found, disabling GPU acceleration")
                    self.gpu_enabled = False
            except Exception as e:
                logger.warning(f"Error initializing GPU: {e}")
                self.gpu_enabled = False
        elif not CUPY_AVAILABLE and self.gpu_enabled:
            logger.warning("CuPy not available, disabling GPU acceleration")
            self.gpu_enabled = False
    
    def _initialize_memory_tracking(self) -> None:
        """Initialize memory tracking."""
        self.memory_tracker = MemoryTracker(limit_mb=self.memory_limit_mb)
        
        # Record initial memory usage
        self.metrics["memory_usage"].append({
            "timestamp": time.time(),
            "usage_mb": self.memory_tracker.get_current_memory_usage()
        })
    
    def parallel_map(self, func: Callable, items: List[Any], 
                   chunksize: Optional[int] = None) -> List[Any]:
        """
        Apply a function to items in parallel.
        
        Args:
            func: Function to apply
            items: List of items
            chunksize: Chunk size for parallel processing
            
        Returns:
            List of results
        """
        if not items:
            return []
            
        if not self.multithreading_enabled or len(items) <= 1:
            # Sequential processing
            return list(map(func, items))
            
        # Determine if the function is CPU-bound or IO-bound
        # This is a heuristic - more sophisticated detection could be implemented
        cpu_bound = self._is_cpu_bound(func)
        
        if cpu_bound and self.process_pool:
            # Use process pool for CPU-bound tasks
            return self.process_pool.map(func, items, chunksize=chunksize or 1)
        elif self.thread_pool:
            # Use thread pool for IO-bound tasks
            return self.thread_pool.map(func, items)
        else:
            # Fallback to sequential processing
            return list(map(func, items))
    
    def _is_cpu_bound(self, func: Callable) -> bool:
        """
        Estimate if a function is CPU-bound or IO-bound.
        
        Args:
            func: Function to analyze
            
        Returns:
            True if likely CPU-bound, False otherwise
        """
        # Check function name for hints
        func_name = func.__name__.lower()
        cpu_keywords = ['calculate', 'compute', 'process', 'analyze', 'transform', 'quantum']
        io_keywords = ['read', 'write', 'load', 'save', 'fetch', 'download']
        
        # Check if function name contains CPU-bound keywords
        if any(keyword in func_name for keyword in cpu_keywords):
            return True
            
        # Check if function name contains IO-bound keywords
        if any(keyword in func_name for keyword in io_keywords):
            return False
            
        # Default to CPU-bound for safety
        return True
    
    def gpu_accelerate(self, array: Union[np.ndarray, List]) -> Union[np.ndarray, "cp.ndarray"]:
        """
        Move an array to GPU if GPU acceleration is enabled.
        
        Args:
            array: NumPy array or list to potentially move to GPU
            
        Returns:
            CuPy array if GPU enabled, original array otherwise
        """
        if not self.gpu_enabled or not CUPY_AVAILABLE:
            # Convert to NumPy array if it's a list
            if isinstance(array, list):
                return np.array(array)
            return array
            
        try:
            # Convert to NumPy array if it's a list
            if isinstance(array, list):
                array = np.array(array)
                
            # Check if already on GPU
            if isinstance(array, cp.ndarray):
                return array
                
            # Move to GPU
            return cp.asarray(array)
        except Exception as e:
            logger.warning(f"Error moving array to GPU: {e}")
            return array
    
    def cpu_return(self, array: Union[np.ndarray, "cp.ndarray"]) -> np.ndarray:
        """
        Move an array to CPU if it's on GPU.
        
        Args:
            array: Array to potentially move to CPU
            
        Returns:
            NumPy array
        """
        if not CUPY_AVAILABLE:
            return array
            
        try:
            # Check if on GPU
            if isinstance(array, cp.ndarray):
                return array.get()
            return array
        except Exception as e:
            logger.warning(f"Error moving array to CPU: {e}")
            return array
    
    @contextmanager
    def timer(self, name: str) -> None:
        """
        Context manager for timing code execution.
        
        Args:
            name: Name to identify the timed section
        """
        start_time = time.time()
        yield
        execution_time = time.time() - start_time
        
        # Record execution time
        if name in self.metrics["execution_times"]:
            self.metrics["execution_times"][name].append(execution_time)
        else:
            self.metrics["execution_times"][name] = [execution_time]
            
        logger.debug(f"Execution time for {name}: {execution_time:.4f} seconds")
    
    def optimize_array(self, array: Union[np.ndarray, List]) -> np.ndarray:
        """
        Optimize a NumPy array for performance.
        
        Args:
            array: Array to optimize
            
        Returns:
            Optimized array
        """
        if isinstance(array, list):
            array = np.array(array)
            
        if not isinstance(array, np.ndarray):
            return array
            
        # Optimize memory layout
        if not array.flags.c_contiguous:
            array = np.ascontiguousarray(array)
            
        # Convert to appropriate dtype for better performance
        if array.dtype == np.float64 and array.size > 1000:
            # Use float32 for large arrays to save memory and improve performance
            array = array.astype(np.float32)
            
        return array
    
    def memoize(self, func: Callable) -> Callable:
        """
        Decorator for memoizing function results.
        
        Args:
            func: Function to memoize
            
        Returns:
            Memoized function
        """
        @lru_cache(maxsize=self.cache_size)
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
            
        return wrapper
    
    @contextmanager
    def memory_usage_tracking(self) -> None:
        """
        Context manager for tracking memory usage during execution.
        """
        # Record memory before
        memory_before = self.memory_tracker.get_current_memory_usage()
        
        yield
        
        # Record memory after
        memory_after = self.memory_tracker.get_current_memory_usage()
        memory_delta = memory_after - memory_before
        
        # Record memory usage
        self.metrics["memory_usage"].append({
            "timestamp": time.time(),
            "usage_mb": memory_after,
            "delta_mb": memory_delta
        })
        
        logger.debug(f"Memory usage delta: {memory_delta:.2f} MB")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics.
        
        Returns:
            Dictionary with performance metrics
        """
        # Add current memory usage
        current_memory = self.memory_tracker.get_current_memory_usage()
        
        # Calculate average execution times
        avg_execution_times = {}
        for name, times in self.metrics["execution_times"].items():
            avg_execution_times[name] = sum(times) / len(times) if times else 0
        
        return {
            "current_memory_mb": current_memory,
            "peak_memory_mb": max([m["usage_mb"] for m in self.metrics["memory_usage"]]) if self.metrics["memory_usage"] else 0,
            "avg_execution_times": avg_execution_times,
            "gpu_enabled": self.gpu_enabled,
            "multithreading_enabled": self.multithreading_enabled,
            "thread_count": self.optimal_thread_count if self.multithreading_enabled else 1
        }
    
    def clear_cache(self) -> None:
        """Clear all caches and release memory."""
        # Clear NumPy cache
        np.clear_nexttolast_op_caches()
        
        # Clear GPU memory if available
        if self.gpu_enabled and CUPY_AVAILABLE:
            cp.clear_memo()
            cp.get_default_memory_pool().free_all_blocks()
            
        # Clear LRU caches
        gc.collect()
        
        logger.debug("Cleared caches and released memory")
    
    def optimize_batch_size(self, data_size: int) -> int:
        """
        Calculate optimal batch size for processing.
        
        Args:
            data_size: Size of data to process
            
        Returns:
            Optimal batch size
        """
        if not self.multithreading_enabled:
            return data_size
            
        # Heuristic for optimal batch size
        # Balance between parallelism and overhead
        if data_size < 100:
            return data_size
            
        thread_count = self.optimal_thread_count
        
        # Target at least 4 items per thread, but no more than 1000
        items_per_thread = max(4, min(1000, data_size // thread_count))
        
        # Calculate batch size
        batch_size = items_per_thread * thread_count
        
        # Ensure batch size is not larger than data size
        return min(batch_size, data_size)
    
    def check_memory_pressure(self) -> bool:
        """
        Check if system is under memory pressure.
        
        Returns:
            True if memory pressure is high, False otherwise
        """
        return self.memory_tracker.check_high_memory_pressure()
    
    def cleanup(self) -> None:
        """Clean up resources."""
        # Shut down thread pool
        if self.thread_pool:
            self.thread_pool.shutdown()
            
        # Shut down process pool
        if self.process_pool:
            self.process_pool.close()
            self.process_pool.join()
            
        # Clear caches
        self.clear_cache()
        
        logger.debug("Performance optimizer resources cleaned up")


class ThreadPool:
    """
    Simple thread pool implementation for parallel processing.
    """
    
    def __init__(self, num_threads: int):
        """
        Initialize the thread pool.
        
        Args:
            num_threads: Number of threads in the pool
        """
        self.num_threads = num_threads
        self.results = {}
        self.lock = threading.Lock()
        self.active = True
    
    def map(self, func: Callable, items: List[Any]) -> List[Any]:
        """
        Apply a function to each item in parallel.
        
        Args:
            func: Function to apply
            items: List of items
            
        Returns:
            List of results
        """
        if not items:
            return []
            
        # Create result list
        results = [None] * len(items)
        
        # Define worker function
        def worker(idx, item):
            try:
                result = func(item)
                results[idx] = result
            except Exception as e:
                logger.error(f"Error in thread worker: {e}")
                results[idx] = None
        
        # Create and start threads
        threads = []
        for i, item in enumerate(items):
            thread = threading.Thread(target=worker, args=(i, item))
            threads.append(thread)
            thread.start()
            
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
            
        return results
    
    def shutdown(self) -> None:
        """Shutdown the thread pool."""
        self.active = False


class MemoryTracker:
    """
    Track and manage memory usage.
    """
    
    def __init__(self, limit_mb: Optional[int] = None):
        """
        Initialize the memory tracker.
        
        Args:
            limit_mb: Memory limit in MB (None for no limit)
        """
        self.limit_mb = limit_mb
        self.process = psutil.Process(os.getpid())
        
        # Initialize memory usage
        self.initial_memory_mb = self.get_current_memory_usage()
        self.warnings_issued = 0
        
        # Log initial memory usage
        logger.debug(f"Initial memory usage: {self.initial_memory_mb:.2f} MB")
        
        # Set memory limit if specified
        if self.limit_mb:
            logger.info(f"Memory limit set to {self.limit_mb} MB")
    
    def get_current_memory_usage(self) -> float:
        """
        Get current memory usage in MB.
        
        Returns:
            Memory usage in MB
        """
        try:
            # Get resident set size (actual memory used)
            memory_info = self.process.memory_info()
            return memory_info.rss / (1024 * 1024)  # Convert to MB
        except:
            # Fallback
            return 0.0
    
    def check_memory_limit(self) -> bool:
        """
        Check if memory usage is within limit.
        
        Returns:
            True if within limit or no limit set, False otherwise
        """
        if not self.limit_mb:
            return True
            
        current_memory = self.get_current_memory_usage()
        
        if current_memory > self.limit_mb:
            # Memory limit exceeded
            self.warnings_issued += 1
            
            # Log warning (limit frequency of warnings)
            if self.warnings_issued <= 3 or self.warnings_issued % 10 == 0:
                logger.warning(f"Memory limit exceeded: {current_memory:.2f} MB / {self.limit_mb} MB")
                
            return False
            
        return True
    
    def check_high_memory_pressure(self) -> bool:
        """
        Check if system is under high memory pressure.
        
        Returns:
            True if high memory pressure, False otherwise
        """
        try:
            # Get system memory info
            system_memory = psutil.virtual_memory()
            
            # High pressure if available memory is less than 15% of total
            return system_memory.available / system_memory.total < 0.15
        except:
            # Conservative approach - assume no pressure
            return False
    
    def force_garbage_collection(self) -> None:
        """Force garbage collection to free memory."""
        collected = gc.collect()
        logger.debug(f"Forced garbage collection: {collected} objects collected")


# Utility functions

def optimize_thread_count() -> int:
    """
    Determine optimal thread count based on system.
    
    Returns:
        Optimal thread count
    """
    cpu_count = os.cpu_count() or 4
    
    # Check if running in a container with limited CPUs
    try:
        # Read cgroup CPU quota and period
        cpu_quota = -1
        cpu_period = 100000  # Default period
        
        cgroup_path = '/sys/fs/cgroup/cpu/cpu.cfs_quota_us'
        if os.path.exists(cgroup_path):
            with open(cgroup_path, 'r') as f:
                cpu_quota = int(f.read().strip())
                
        cgroup_path = '/sys/fs/cgroup/cpu/cpu.cfs_period_us'
        if os.path.exists(cgroup_path):
            with open(cgroup_path, 'r') as f:
                cpu_period = int(f.read().strip())
        
        if cpu_quota > 0:
            # Calculate CPUs allocated to container
            container_cpus = cpu_quota / cpu_period
            cpu_count = min(cpu_count, int(container_cpus))
    except:
        # Ignore errors and use default calculation
        pass
    
    # Leave one CPU for system processes
    return max(1, cpu_count - 1)

def gpu_available() -> bool:
    """
    Check if GPU acceleration is available.
    
    Returns:
        True if GPU acceleration is available, False otherwise
    """
    if not CUPY_AVAILABLE:
        return False
        
    try:
        # Check if any CUDA devices are available
        device_count = cp.cuda.runtime.getDeviceCount()
        return device_count > 0
    except:
        return False

def memory_stats() -> Dict[str, float]:
    """
    Get memory statistics.
    
    Returns:
        Dictionary with memory statistics
    """
    stats = {}
    
    try:
        # Get system memory info
        system_memory = psutil.virtual_memory()
        stats["total_gb"] = system_memory.total / (1024**3)
        stats["available_gb"] = system_memory.available / (1024**3)
        stats["used_gb"] = system_memory.used / (1024**3)
        stats["percent_used"] = system_memory.percent
        
        # Get process memory info
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        stats["process_mb"] = memory_info.rss / (1024**2)
        
        return stats
    except:
        return {"error": "Failed to get memory statistics"}

# Decorators

def gpu_accelerated(func):
    """
    Decorator to accelerate function with GPU if available.
    
    Args:
        func: Function to accelerate
        
    Returns:
        Accelerated function
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Check if GPU is available
        if not CUPY_AVAILABLE:
            return func(*args, **kwargs)
            
        # Convert NumPy arrays to CuPy arrays
        gpu_args = []
        for arg in args:
            if isinstance(arg, np.ndarray):
                gpu_args.append(cp.asarray(arg))
            else:
                gpu_args.append(arg)
                
        gpu_kwargs = {}
        for key, value in kwargs.items():
            if isinstance(value, np.ndarray):
                gpu_kwargs[key] = cp.asarray(value)
            else:
                gpu_kwargs[key] = value
                
        # Call function with GPU arrays
        result = func(*gpu_args, **gpu_kwargs)
        
        # Convert result back to NumPy if needed
        if isinstance(result, cp.ndarray):
            return result.get()
        elif isinstance(result, tuple) and any(isinstance(x, cp.ndarray) for x in result):
            return tuple(x.get() if isinstance(x, cp.ndarray) else x for x in result)
        elif isinstance(result, list) and any(isinstance(x, cp.ndarray) for x in result):
            return [x.get() if isinstance(x, cp.ndarray) else x for x in result]
        else:
            return result
            
    return wrapper

def timeit(func):
    """
    Decorator to time function execution.
    
    Args:
        func: Function to time
        
    Returns:
        Timed function
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time
        
        logger.debug(f"{func.__name__} executed in {execution_time:.4f} seconds")
        return result
        
    return wrapper

def check_memory(func):
    """
    Decorator to check memory usage before and after function execution.
    
    Args:
        func: Function to check
        
    Returns:
        Memory-checked function
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / (1024 * 1024)  # MB
        
        result = func(*args, **kwargs)
        
        memory_after = process.memory_info().rss / (1024 * 1024)  # MB
        memory_delta = memory_after - memory_before
        
        logger.debug(f"{func.__name__} memory delta: {memory_delta:.2f} MB")
        return result
        
    return wrapper