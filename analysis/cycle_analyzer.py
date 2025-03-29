"""
Cycle-accurate analysis tools for hardware timing inspection.
Analyzes and extracts insights from hardware cycle data.
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
import matplotlib.pyplot as plt
from collections import defaultdict

logger = logging.getLogger("QuantumSignalEmulator.CycleAnalyzer")

class CycleAnalyzer:
    """
    Analyzes cycle-accurate timing data from hardware emulation.
    Provides tools for identifying patterns, anomalies, and metrics in
    cycle-level hardware behavior.
    """
    
    def __init__(self, precision: float = 0.1):
        """
        Initialize the cycle analyzer.
        
        Args:
            precision: Analysis precision threshold (ns)
        """
        self.precision = precision
        self.analysis_cache = {}
        logger.info(f"Initialized cycle analyzer with precision {precision}ns")
    
    def extract_signal_data(self, state_history: List[Dict[str, Any]]) -> np.ndarray:
        """
        Extract signal data from state history for quantum analysis.
        
        Args:
            state_history: List of system state snapshots
            
        Returns:
            Numpy array of signal values suitable for analysis
        """
        if not state_history:
            logger.warning("No state history to extract signal from")
            return np.array([])
            
        # Extract register transitions as signal
        # This approach looks at how register values change over time
        signal_data = []
        
        for i in range(1, len(state_history)):
            prev_state = state_history[i-1]
            curr_state = state_history[i]
            
            if "registers" in curr_state and "registers" in prev_state:
                # Calculate delta of register values
                curr_registers = curr_state["registers"]
                prev_registers = prev_state["registers"]
                
                # Combine all register changes into a signal value
                register_delta = 0
                for reg_name in curr_registers:
                    if reg_name in prev_registers:
                        register_delta += abs(curr_registers[reg_name] - prev_registers[reg_name])
                
                signal_data.append(register_delta)
            else:
                # If no register data, use cycle information
                cycle_delta = curr_state.get("cycle", i) - prev_state.get("cycle", i-1)
                signal_data.append(cycle_delta)
        
        return np.array(signal_data)
    
    def analyze_timing_patterns(self, state_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze timing patterns in cycle data.
        
        Args:
            state_history: List of system state snapshots
            
        Returns:
            Dictionary with timing analysis results
        """
        if not state_history:
            logger.warning("No state history to analyze timing patterns")
            return {"error": "No state history available"}
            
        # Extract cycle timing data
        cycles = [state.get("cycle", i) for i, state in enumerate(state_history)]
        scanlines = [state.get("scanline", 0) for state in state_history]
        dots = [state.get("dot", 0) for state in state_history]
        
        # Calculate cycle deltas (time between cycles)
        cycle_deltas = [cycles[i] - cycles[i-1] for i in range(1, len(cycles))]
        
        # Identify common timing patterns
        cycle_patterns = self._find_cycle_patterns(cycle_deltas)
        
        # Analyze scanline timing
        scanline_transitions = []
        for i in range(1, len(scanlines)):
            if scanlines[i] != scanlines[i-1]:
                scanline_transitions.append({
                    "from_scanline": scanlines[i-1],
                    "to_scanline": scanlines[i],
                    "at_cycle": cycles[i],
                    "dot": dots[i]
                })
        
        # Calculate scanline durations
        scanline_durations = defaultdict(list)
        current_scanline = scanlines[0]
        start_cycle = cycles[0]
        
        for i in range(1, len(scanlines)):
            if scanlines[i] != current_scanline:
                duration = cycles[i] - start_cycle
                scanline_durations[current_scanline].append(duration)
                current_scanline = scanlines[i]
                start_cycle = cycles[i]
        
        # Calculate average duration per scanline
        avg_scanline_durations = {
            scanline: sum(durations) / len(durations) if durations else 0
            for scanline, durations in scanline_durations.items()
        }
        
        # Detect timing anomalies
        anomalies = self._detect_timing_anomalies(
            cycle_deltas, 
            scanline_durations, 
            avg_scanline_durations
        )
        
        return {
            "cycle_patterns": cycle_patterns,
            "scanline_transitions": scanline_transitions[:10],  # Limit to first 10 for brevity
            "avg_scanline_durations": avg_scanline_durations,
            "timing_anomalies": anomalies,
            "statistics": {
                "total_cycles": cycles[-1] - cycles[0] if len(cycles) > 1 else 0,
                "min_cycle_delta": min(cycle_deltas) if cycle_deltas else 0,
                "max_cycle_delta": max(cycle_deltas) if cycle_deltas else 0,
                "avg_cycle_delta": sum(cycle_deltas) / len(cycle_deltas) if cycle_deltas else 0
            }
        }
    
    def _find_cycle_patterns(self, cycle_deltas: List[int]) -> List[Dict[str, Any]]:
        """
        Find common patterns in cycle timing.
        
        Args:
            cycle_deltas: List of cycle time differences
            
        Returns:
            List of identified patterns
        """
        if not cycle_deltas:
            return []
            
        # Find repeating patterns of different lengths
        patterns = []
        
        # Check for patterns of length 1-8 cycles
        for pattern_length in range(1, min(9, len(cycle_deltas) // 4)):
            pattern_counts = defaultdict(int)
            
            # Slide through the deltas looking for repeating patterns
            for i in range(len(cycle_deltas) - pattern_length + 1):
                # Convert segment to tuple for hashing
                segment = tuple(cycle_deltas[i:i+pattern_length])
                pattern_counts[segment] += 1
            
            # Find patterns that repeat at least 3 times
            common_patterns = {
                pattern: count for pattern, count in pattern_counts.items()
                if count >= 3
            }
            
            # Add to results, most frequent first
            for pattern, count in sorted(common_patterns.items(), key=lambda x: x[1], reverse=True)[:3]:
                patterns.append({
                    "pattern": list(pattern),
                    "length": pattern_length,
                    "occurrences": count
                })
        
        return patterns
    
    def _detect_timing_anomalies(self, 
                               cycle_deltas: List[int], 
                               scanline_durations: Dict[int, List[int]],
                               avg_scanline_durations: Dict[int, float]) -> List[Dict[str, Any]]:
        """
        Detect timing anomalies in the cycle data.
        
        Args:
            cycle_deltas: List of cycle time differences
            scanline_durations: Dictionary of durations per scanline
            avg_scanline_durations: Average duration per scanline
            
        Returns:
            List of detected anomalies
        """
        anomalies = []
        
        # Skip if not enough data
        if len(cycle_deltas) < 10:
            return anomalies
            
        # Calculate statistics for anomaly detection
        mean_delta = sum(cycle_deltas) / len(cycle_deltas)
        std_delta = np.std(cycle_deltas)
        
        # Detect cycle timing anomalies (3 sigma rule)
        threshold = 3 * std_delta
        
        for i, delta in enumerate(cycle_deltas):
            if abs(delta - mean_delta) > threshold:
                anomalies.append({
                    "type": "cycle_timing",
                    "position": i,
                    "expected": mean_delta,
                    "actual": delta,
                    "deviation": delta - mean_delta
                })
                
        # Detect scanline duration anomalies
        for scanline, durations in scanline_durations.items():
            if len(durations) <= 1:
                continue
                
            avg_duration = avg_scanline_durations[scanline]
            std_duration = np.std(durations)
            
            for i, duration in enumerate(durations):
                if abs(duration - avg_duration) > 2 * std_duration:  # 2 sigma for scanlines
                    anomalies.append({
                        "type": "scanline_duration",
                        "scanline": scanline,
                        "occurrence": i,
                        "expected": avg_duration,
                        "actual": duration,
                        "deviation": duration - avg_duration
                    })
        
        return anomalies[:10]  # Limit to top 10 anomalies
    
    def plot_cycle_histogram(self, state_history: List[Dict[str, Any]], 
                           bins: int = 50, figsize: Tuple[int, int] = (10, 6)) -> None:
        """
        Plot histogram of cycle timing distribution.
        
        Args:
            state_history: List of system state snapshots
            bins: Number of histogram bins
            figsize: Figure size (width, height) in inches
        """
        if not state_history:
            logger.warning("No state history to plot cycle histogram")
            return
            
        # Extract cycle timing data
        cycles = [state.get("cycle", i) for i, state in enumerate(state_history)]
        
        # Calculate cycle deltas (time between cycles)
        cycle_deltas = [cycles[i] - cycles[i-1] for i in range(1, len(cycles))]
        
        if not cycle_deltas:
            logger.warning("Not enough cycle data to plot histogram")
            return
            
        # Create plot
        plt.figure(figsize=figsize)
        plt.hist(cycle_deltas, bins=bins, alpha=0.7, color='blue')
        plt.xlabel('Cycle Duration')
        plt.ylabel('Frequency')
        plt.title('Cycle Timing Distribution')
        plt.grid(True, alpha=0.3)
        
        # Add statistics to plot
        mean_delta = sum(cycle_deltas) / len(cycle_deltas)
        std_delta = np.std(cycle_deltas)
        
        stats_text = f"Mean: {mean_delta:.2f}\nStd Dev: {std_delta:.2f}\nMin: {min(cycle_deltas)}\nMax: {max(cycle_deltas)}"
        plt.annotate(stats_text, xy=(0.95, 0.95), xycoords='axes fraction', 
                    fontsize=10, ha='right', va='top', 
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
        
        plt.tight_layout()
        plt.show()
    
    def analyze_register_activity(self, state_history: List[Dict[str, Any]], 
                                register_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Analyze register activity patterns.
        
        Args:
            state_history: List of system state snapshots
            register_names: Specific registers to analyze (all if None)
            
        Returns:
            Dictionary with register activity analysis
        """
        if not state_history:
            logger.warning("No state history to analyze register activity")
            return {"error": "No state history available"}
            
        # Find available registers if not specified
        if register_names is None:
            register_names = set()
            for state in state_history:
                if "registers" in state:
                    register_names.update(state["registers"].keys())
            register_names = list(register_names)
        
        if not register_names:
            logger.warning("No registers found in state history")
            return {"error": "No register data available"}
            
        # Analyze each register
        register_analysis = {}
        
        for reg_name in register_names:
            # Extract register values
            values = []
            for state in state_history:
                if "registers" in state and reg_name in state["registers"]:
                    values.append(state["registers"][reg_name])
                else:
                    # Use previous value or 0 if not available
                    values.append(values[-1] if values else 0)
            
            if not values:
                continue
                
            # Calculate change frequency
            changes = sum(1 for i in range(1, len(values)) if values[i] != values[i-1])
            change_freq = changes / len(values) if len(values) > 1 else 0
            
            # Find most common values
            value_counts = defaultdict(int)
            for v in values:
                value_counts[v] += 1
                
            # Calculate statistics
            register_analysis[reg_name] = {
                "change_frequency": change_freq,
                "total_changes": changes,
                "min_value": min(values),
                "max_value": max(values),
                "mean_value": sum(values) / len(values),
                "most_common_values": sorted(value_counts.items(), key=lambda x: x[1], reverse=True)[:3],
                "value_entropy": self._calculate_entropy(value_counts)
            }
        
        return register_analysis
    
    def _calculate_entropy(self, value_counts: Dict[int, int]) -> float:
        """
        Calculate Shannon entropy of value distribution.
        
        Args:
            value_counts: Dictionary of value frequencies
            
        Returns:
            Entropy value
        """
        total = sum(value_counts.values())
        probabilities = [count / total for count in value_counts.values()]
        entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)
        return entropy