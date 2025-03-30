"""
Tests for the CycleAnalyzer module.

This module contains unit tests for the CycleAnalyzer class, which is responsible
for analyzing cycle-accurate timing data from hardware emulation.
"""
import unittest
import numpy as np
from quantum_signal_emulator.analysis.cycle_analyzer import CycleAnalyzer

class TestCycleAnalyzer(unittest.TestCase):
    """
    Test cases for the CycleAnalyzer class.
    """
    
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = CycleAnalyzer(precision=0.1)
        
        # Create sample state history
        self.state_history = [
            {
                "cycle": 0,
                "scanline": 0,
                "dot": 0,
                "registers": {
                    "A": 0,
                    "X": 0,
                    "Y": 0,
                    "PC": 0x8000
                }
            },
            {
                "cycle": 3,
                "scanline": 0,
                "dot": 3,
                "registers": {
                    "A": 10,
                    "X": 0,
                    "Y": 0,
                    "PC": 0x8002
                }
            },
            {
                "cycle": 6,
                "scanline": 0,
                "dot": 6,
                "registers": {
                    "A": 10,
                    "X": 5,
                    "Y": 0,
                    "PC": 0x8004
                }
            },
            {
                "cycle": 9,
                "scanline": 0,
                "dot": 9,
                "registers": {
                    "A": 10,
                    "X": 5,
                    "Y": 15,
                    "PC": 0x8006
                }
            },
            {
                "cycle": 12,
                "scanline": 1,
                "dot": 0,
                "registers": {
                    "A": 20,
                    "X": 5,
                    "Y": 15,
                    "PC": 0x8008
                }
            }
        ]
    
    def test_extract_signal_data(self):
        """Test extracting signal data from state history."""
        signal_data = self.analyzer.extract_signal_data(self.state_history)
        
        # Should have one fewer elements than state history
        self.assertEqual(len(signal_data), len(self.state_history) - 1)
        
        # Signal should be non-empty
        self.assertGreater(len(signal_data), 0)
        
        # Signal should be 1D array
        self.assertEqual(len(signal_data.shape), 1)
    
    def test_analyze_timing_patterns(self):
        """Test analyzing timing patterns."""
        results = self.analyzer.analyze_timing_patterns(self.state_history)
        
        # Check if results dictionary has expected keys
        self.assertIn("cycle_patterns", results)
        self.assertIn("scanline_transitions", results)
        self.assertIn("statistics", results)
        
        # Check statistics
        stats = results["statistics"]
        self.assertEqual(stats["total_cycles"], 12)  # Last cycle - first cycle
        self.assertEqual(stats["min_cycle_delta"], 3)  # Each step is 3 cycles
        self.assertEqual(stats["max_cycle_delta"], 3)
        self.assertEqual(stats["avg_cycle_delta"], 3.0)
        
        # Check scanline transitions
        transitions = results["scanline_transitions"]
        self.assertEqual(len(transitions), 1)  # One transition from scanline 0 to 1
        self.assertEqual(transitions[0]["from_scanline"], 0)
        self.assertEqual(transitions[0]["to_scanline"], 1)
    
    def test_analyze_register_activity(self):
        """Test analyzing register activity."""
        results = self.analyzer.analyze_register_activity(self.state_history)
        
        # Check if all registers are analyzed
        self.assertIn("A", results)
        self.assertIn("X", results)
        self.assertIn("Y", results)
        self.assertIn("PC", results)
        
        # Check register A analysis
        reg_a = results["A"]
        self.assertEqual(reg_a["change_frequency"], 0.5)  # Changed in 2 of 4 transitions
        self.assertEqual(reg_a["total_changes"], 2)
        self.assertEqual(reg_a["min_value"], 0)
        self.assertEqual(reg_a["max_value"], 20)
        
        # Check register PC analysis
        reg_pc = results["PC"]
        self.assertEqual(reg_pc["total_changes"], 4)  # Changed in every transition
        
    def test_empty_state_history(self):
        """Test behavior with empty state history."""
        empty_history = []
        
        # Should return empty array
        signal_data = self.analyzer.extract_signal_data(empty_history)
        self.assertEqual(len(signal_data), 0)
        
        # Should return error dictionary
        timing_results = self.analyzer.analyze_timing_patterns(empty_history)
        self.assertIn("error", timing_results)
        
        # Should return error dictionary
        register_results = self.analyzer.analyze_register_activity(empty_history)
        self.assertIn("error", register_results)
    
    def test_find_cycle_patterns(self):
        """Test finding cycle patterns."""
        # Simple repeating pattern: 3, 3, 3, 3
        cycle_deltas = [3, 3, 3, 3]
        patterns = self.analyzer._find_cycle_patterns(cycle_deltas)
        
        # Should detect the pattern [3]
        self.assertGreater(len(patterns), 0)
        pattern = patterns[0]
        self.assertEqual(pattern["pattern"], [3])
        self.assertGreaterEqual(pattern["occurrences"], 3)  # At least 3 occurrences

if __name__ == '__main__':
    unittest.main()