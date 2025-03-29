"""
State recording module for capturing and managing hardware emulation state.
"""

import numpy as np
import logging
import json
import os
import time
from typing import Dict, List, Tuple, Optional, Any, Union
from collections import defaultdict, deque
import pickle

logger = logging.getLogger("QuantumSignalEmulator.StateRecorder")

class StateRecorder:
    """
    Records, manages, and manipulates system state data during emulation.
    Supports efficiently storing and retrieving hardware state snapshots.
    """
    
    def __init__(self, max_history: int = 100000, 
                compression_ratio: int = 10,
                record_filter: Optional[List[str]] = None):
        """
        Initialize the state recorder.
        
        Args:
            max_history: Maximum number of states to keep in memory
            compression_ratio: Ratio for compressed storage (1:N)
            record_filter: List of register names to include (None for all)
        """
        self.max_history = max_history
        self.compression_ratio = max(1, compression_ratio)
        self.record_filter = record_filter
        
        # State storage (circular buffer)
        self.state_history = deque(maxlen=max_history)
        
        # Compression storage (keeps 1 out of N states)
        self.compressed_history = []
        
        # Index by cycle for fast lookup
        self.cycle_index = {}
        
        # Statistics
        self.stats = {
            "total_records": 0,
            "start_time": time.time(),
            "start_cycle": None,
            "current_cycle": None,
            "unique_registers": set(),
            "compressed_ratio": compression_ratio
        }
        
        logger.info(f"Initialized state recorder with max history {max_history}, " +
                   f"compression ratio 1:{compression_ratio}")
    
    def record_state(self, state: Dict[str, Any]) -> None:
        """
        Record a system state snapshot.
        
        Args:
            state: System state dictionary
        """
        # Filter registers if needed
        if self.record_filter is not None and "registers" in state:
            filtered_registers = {}
            for reg_name in self.record_filter:
                if reg_name in state["registers"]:
                    filtered_registers[reg_name] = state["registers"][reg_name]
            
            # Replace with filtered version
            state = state.copy()
            state["registers"] = filtered_registers
        
        # Update stats
        self.stats["total_records"] += 1
        
        if "cycle" in state:
            cycle = state["cycle"]
            
            if self.stats["start_cycle"] is None:
                self.stats["start_cycle"] = cycle
                
            self.stats["current_cycle"] = cycle
            
            # Index by cycle
            self.cycle_index[cycle] = len(self.state_history)
            
        # Update register stats
        if "registers" in state:
            self.stats["unique_registers"].update(state["registers"].keys())
            
        # Add to history
        self.state_history.append(state)
        
        # Add to compressed history if needed
        if self.stats["total_records"] % self.compression_ratio == 0:
            self.compressed_history.append(state)
    
    def get_state_history(self, start_idx: Optional[int] = None, 
                       end_idx: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get a slice of the state history.
        
        Args:
            start_idx: Starting index (None for beginning)
            end_idx: Ending index (None for end)
            
        Returns:
            List of state snapshots
        """
        # Convert to list and slice
        history_list = list(self.state_history)
        
        if start_idx is not None or end_idx is not None:
            return history_list[start_idx:end_idx]
        else:
            return history_list
    
    def get_state_by_cycle(self, cycle: int) -> Optional[Dict[str, Any]]:
        """
        Get state snapshot for a specific cycle.
        
        Args:
            cycle: Cycle number to retrieve
            
        Returns:
            State snapshot if found, None otherwise
        """
        if cycle in self.cycle_index:
            idx = self.cycle_index[cycle]
            history_list = list(self.state_history)
            
            if 0 <= idx < len(history_list):
                return history_list[idx]
        
        # Try to find nearest cycle
        nearest_cycle = self._find_nearest_cycle(cycle)
        if nearest_cycle is not None:
            logger.info(f"Exact cycle {cycle} not found, returning nearest cycle {nearest_cycle}")
            return self.get_state_by_cycle(nearest_cycle)
        
        return None
    
    def _find_nearest_cycle(self, cycle: int) -> Optional[int]:
        """
        Find the nearest recorded cycle to the requested one.
        
        Args:
            cycle: Target cycle
            
        Returns:
            Nearest recorded cycle or None if no cycles recorded
        """
        if not self.cycle_index:
            return None
            
        cycles = list(self.cycle_index.keys())
        cycles.sort(key=lambda x: abs(x - cycle))
        return cycles[0]
    
    def get_register_history(self, register_name: str) -> Dict[str, List[Any]]:
        """
        Get history for a specific register.
        
        Args:
            register_name: Name of register to retrieve
            
        Returns:
            Dictionary with cycle numbers and register values
        """
        cycles = []
        values = []
        
        for state in self.state_history:
            if "registers" in state and register_name in state["registers"]:
                if "cycle" in state:
                    cycles.append(state["cycle"])
                else:
                    cycles.append(len(cycles))
                    
                values.append(state["registers"][register_name])
        
        return {
            "cycles": cycles,
            "values": values
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get recorder statistics.
        
        Returns:
            Dictionary with statistics
        """
        # Update stats
        elapsed_time = time.time() - self.stats["start_time"]
        
        if "start_cycle" in self.stats and self.stats["start_cycle"] is not None and \
           "current_cycle" in self.stats and self.stats["current_cycle"] is not None:
            total_cycles = self.stats["current_cycle"] - self.stats["start_cycle"]
        else:
            total_cycles = 0
            
        # Calculate cycles per second
        if elapsed_time > 0:
            cycles_per_second = total_cycles / elapsed_time
        else:
            cycles_per_second = 0
        
        # Calculate memory usage (approximate)
        current_size = len(self.state_history)
        memory_usage = self._estimate_memory_usage()
        
        return {
            "total_records": self.stats["total_records"],
            "elapsed_time": elapsed_time,
            "total_cycles": total_cycles,
            "cycles_per_second": cycles_per_second,
            "current_history_size": current_size,
            "max_history_size": self.max_history,
            "compression_ratio": self.compression_ratio,
            "compressed_history_size": len(self.compressed_history),
            "unique_registers": list(self.stats["unique_registers"]),
            "estimated_memory_mb": memory_usage
        }
    
    def _estimate_memory_usage(self) -> float:
        """
        Estimate memory usage in MB.
        
        Returns:
            Estimated memory usage in MB
        """
        # Sample a few states to estimate size
        if not self.state_history:
            return 0.0
            
        sample_size = min(10, len(self.state_history))
        sample_indices = np.linspace(0, len(self.state_history)-1, sample_size, dtype=int)
        
        total_bytes = 0
        for idx in sample_indices:
            state = list(self.state_history)[idx]
            # Use pickle to estimate size
            state_bytes = len(pickle.dumps(state))
            total_bytes += state_bytes
            
        # Calculate average size per state
        avg_bytes_per_state = total_bytes / sample_size
        
        # Estimate total memory usage
        total_states = len(self.state_history) + len(self.compressed_history)
        estimated_bytes = avg_bytes_per_state * total_states
        
        # Convert to MB
        return estimated_bytes / (1024 * 1024)
    
    def save_history(self, filename: str, format: str = 'pickle') -> bool:
        """
        Save state history to a file.
        
        Args:
            filename: Output filename
            format: File format ('pickle', 'json', or 'csv')
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create output directory if it doesn't exist
            output_dir = os.path.dirname(filename)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
                
            # Package data
            data = {
                "history": list(self.state_history),
                "compressed": self.compressed_history,
                "statistics": self.get_statistics()
            }
            
            # Save in requested format
            if format == 'pickle':
                with open(filename, 'wb') as f:
                    pickle.dump(data, f)
                    
            elif format == 'json':
                # Convert numpy arrays to lists for JSON serialization
                self._convert_numpy_arrays(data)
                
                with open(filename, 'w') as f:
                    json.dump(data, f, indent=2)
                    
            elif format == 'csv':
                # Create CSV from history
                with open(filename, 'w') as f:
                    # Write header
                    if self.state_history:
                        sample_state = self.state_history[0]
                        headers = ["record_idx", "cycle", "scanline", "dot"]
                        
                        if "registers" in sample_state:
                            register_names = list(self.stats["unique_registers"])
                            register_headers = [f"reg_{name}" for name in register_names]
                            headers.extend(register_headers)
                            
                        f.write(",".join(headers) + "\n")
                        
                        # Write data
                        for i, state in enumerate(self.state_history):
                            row = [
                                str(i),
                                str(state.get("cycle", "")),
                                str(state.get("scanline", "")),
                                str(state.get("dot", ""))
                            ]
                            
                            if "registers" in state:
                                for reg_name in register_names:
                                    row.append(str(state["registers"].get(reg_name, "")))
                                    
                            f.write(",".join(row) + "\n")
                    else:
                        f.write("No state history available\n")
            else:
                logger.error(f"Unsupported format: {format}")
                return False
                
            logger.info(f"Saved state history to {filename} in {format} format")
            return True
            
        except Exception as e:
            logger.error(f"Error saving history: {e}")
            return False
    
    def _convert_numpy_arrays(self, data: Dict[str, Any]) -> None:
        """
        Convert numpy arrays to lists for JSON serialization (in-place).
        
        Args:
            data: Data dictionary to convert
        """
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, np.ndarray):
                    data[key] = value.tolist()
                elif isinstance(value, (dict, list)):
                    self._convert_numpy_arrays(value)
        elif isinstance(data, list):
            for i, item in enumerate(data):
                if isinstance(item, np.ndarray):
                    data[i] = item.tolist()
                elif isinstance(item, (dict, list)):
                    self._convert_numpy_arrays(item)
    
    def load_history(self, filename: str) -> bool:
        """
        Load state history from a file.
        
        Args:
            filename: Input filename
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check file format based on extension
            _, ext = os.path.splitext(filename)
            
            if ext.lower() == '.pkl' or ext.lower() == '.pickle':
                with open(filename, 'rb') as f:
                    data = pickle.load(f)
            elif ext.lower() == '.json':
                with open(filename, 'r') as f:
                    data = json.load(f)
            else:
                logger.error(f"Unsupported file format: {ext}")
                return False
                
            # Load data
            if "history" in data:
                self.state_history = deque(data["history"], maxlen=self.max_history)
                
            if "compressed" in data:
                self.compressed_history = data["compressed"]
                
            # Rebuild cycle index
            self.cycle_index = {}
            for i, state in enumerate(self.state_history):
                if "cycle" in state:
                    self.cycle_index[state["cycle"]] = i
                    
            # Update statistics
            if "statistics" in data:
                # Only update certain statistics
                stats_to_update = ["total_records", "start_cycle", "current_cycle"]
                for stat in stats_to_update:
                    if stat in data["statistics"]:
                        self.stats[stat] = data["statistics"][stat]
                        
                # Update unique registers
                if "unique_registers" in data["statistics"]:
                    self.stats["unique_registers"] = set(data["statistics"]["unique_registers"])
                else:
                    # Rebuild from history
                    unique_regs = set()
                    for state in self.state_history:
                        if "registers" in state:
                            unique_regs.update(state["registers"].keys())
                    self.stats["unique_registers"] = unique_regs
                    
            logger.info(f"Loaded state history from {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading history: {e}")
            return False
    
    def find_register_value_changes(self, register_name: str, 
                                 start_cycle: Optional[int] = None,
                                 end_cycle: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Find all instances where a register changes value.
        
        Args:
            register_name: Register name to track
            start_cycle: Starting cycle (None for beginning)
            end_cycle: Ending cycle (None for end)
            
        Returns:
            List of change events with cycle and value information
        """
        changes = []
        last_value = None
        
        for state in self.state_history:
            if "registers" in state and register_name in state["registers"]:
                current_value = state["registers"][register_name]
                current_cycle = state.get("cycle", None)
                
                # Check if we're within cycle range
                if start_cycle is not None and current_cycle is not None and current_cycle < start_cycle:
                    last_value = current_value
                    continue
                    
                if end_cycle is not None and current_cycle is not None and current_cycle > end_cycle:
                    break
                
                # Record change if value is different
                if last_value is not None and current_value != last_value:
                    changes.append({
                        "cycle": current_cycle,
                        "scanline": state.get("scanline", None),
                        "dot": state.get("dot", None),
                        "old_value": last_value,
                        "new_value": current_value
                    })
                
                last_value = current_value
        
        return changes
    
    def find_register_patterns(self, register_name: str, 
                           pattern_length: int = 5) -> List[Dict[str, Any]]:
        """
        Find repeating patterns in register values.
        
        Args:
            register_name: Register name to analyze
            pattern_length: Length of patterns to search for
            
        Returns:
            List of detected patterns with occurrence information
        """
        # Get register history
        reg_history = self.get_register_history(register_name)
        values = reg_history["values"]
        cycles = reg_history["cycles"]
        
        if len(values) < pattern_length * 2:
            logger.warning(f"Not enough data to find patterns of length {pattern_length}")
            return []
            
        # Find repeating patterns
        pattern_counts = defaultdict(list)
        
        for i in range(len(values) - pattern_length + 1):
            # Extract pattern
            pattern = tuple(values[i:i+pattern_length])
            
            # Record occurrence
            pattern_counts[pattern].append(i)
        
        # Filter to patterns that repeat at least 3 times
        repeating_patterns = []
        
        for pattern, occurrences in pattern_counts.items():
            if len(occurrences) >= 3:
                # Calculate average gap between occurrences
                gaps = [occurrences[i+1] - occurrences[i] for i in range(len(occurrences)-1)]
                avg_gap = sum(gaps) / len(gaps) if gaps else 0
                
                # Convert to cycles
                pattern_cycles = [cycles[i] for i in occurrences]
                
                repeating_patterns.append({
                    "pattern": list(pattern),
                    "occurrences": len(occurrences),
                    "first_occurrence": occurrences[0],
                    "first_cycle": pattern_cycles[0] if pattern_cycles else None,
                    "avg_gap_steps": avg_gap,
                    "avg_gap_cycles": (pattern_cycles[-1] - pattern_cycles[0]) / (len(pattern_cycles) - 1)
                                     if len(pattern_cycles) > 1 else 0
                })
        
        # Sort by number of occurrences
        repeating_patterns.sort(key=lambda x: x["occurrences"], reverse=True)
        
        return repeating_patterns
    
    def analyze_state_transitions(self, num_samples: int = 1000) -> Dict[str, Any]:
        """
        Analyze system state transitions to identify patterns.
        
        Args:
            num_samples: Maximum number of transitions to analyze
            
        Returns:
            Dictionary with transition analysis results
        """
        if len(self.state_history) < 2:
            logger.warning("Not enough state history for transition analysis")
            return {"error": "Not enough state history"}
            
        # Sample state transitions
        history_list = list(self.state_history)
        
        if len(history_list) <= num_samples:
            samples = list(range(len(history_list) - 1))
        else:
            # Uniform sampling
            samples = np.linspace(0, len(history_list) - 2, num_samples, dtype=int)
            
        # Analyze transitions
        cycle_deltas = []
        scanline_transitions = []
        register_transitions = defaultdict(list)
        
        for i in samples:
            current_state = history_list[i]
            next_state = history_list[i+1]
            
            # Analyze cycle transitions
            if "cycle" in current_state and "cycle" in next_state:
                cycle_deltas.append(next_state["cycle"] - current_state["cycle"])
                
            # Analyze scanline transitions
            if "scanline" in current_state and "scanline" in next_state:
                if current_state["scanline"] != next_state["scanline"]:
                    scanline_transitions.append({
                        "from": current_state["scanline"],
                        "to": next_state["scanline"],
                        "cycle": next_state.get("cycle", None)
                    })
                    
            # Analyze register transitions
            if "registers" in current_state and "registers" in next_state:
                for reg_name in set(current_state["registers"]).intersection(next_state["registers"]):
                    current_val = current_state["registers"][reg_name]
                    next_val = next_state["registers"][reg_name]
                    
                    if current_val != next_val:
                        register_transitions[reg_name].append({
                            "from": current_val,
                            "to": next_val,
                            "cycle": next_state.get("cycle", None)
                        })
        
        # Analyze results
        cycle_stats = {
            "min": min(cycle_deltas) if cycle_deltas else None,
            "max": max(cycle_deltas) if cycle_deltas else None,
            "avg": sum(cycle_deltas) / len(cycle_deltas) if cycle_deltas else None,
            "most_common": max(set(cycle_deltas), key=cycle_deltas.count) if cycle_deltas else None
        }
        
        # Identify most active registers
        register_activity = {
            reg_name: len(transitions) for reg_name, transitions in register_transitions.items()
        }
        
        return {
            "cycle_transition_stats": cycle_stats,
            "scanline_transitions": scanline_transitions[:10],  # Limit to 10 for brevity
            "register_activity": register_activity,
            "most_active_registers": sorted(register_activity.items(), key=lambda x: x[1], reverse=True)[:5]
        }