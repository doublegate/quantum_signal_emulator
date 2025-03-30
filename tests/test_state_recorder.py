"""
Tests for the StateRecorder module.

This module contains unit tests for the StateRecorder class, which is responsible
for recording and managing system state history.
"""
import unittest
import os
import tempfile
from quantum_signal_emulator.analysis.state_recorder import StateRecorder

class TestStateRecorder(unittest.TestCase):
    """
    Test cases for the StateRecorder class.
    """
    
    def setUp(self):
        """Set up test fixtures."""
        # Create recorder with smaller history size for testing
        self.recorder = StateRecorder(max_history=10, compression_ratio=2)
        
        # Sample state dictionaries
        self.states = [
            {
                "cycle": i * 3,
                "scanline": i // 3,
                "dot": i % 3,
                "registers": {
                    "A": i,
                    "X": i * 2,
                    "PC": 0x8000 + i * 2
                }
            }
            for i in range(15)  # Create 15 sample states
        ]
    
    def test_record_state(self):
        """Test recording states."""
        # Record several states
        for state in self.states[:5]:
            self.recorder.record_state(state)
        
        # Check if states were recorded
        history = self.recorder.get_state_history()
        self.assertEqual(len(history), 5)
        
        # Check if stats were updated
        stats = self.recorder.get_statistics()
        self.assertEqual(stats["total_records"], 5)
        self.assertEqual(set(stats["unique_registers"]), {"A", "X", "PC"})
    
    def test_max_history_limit(self):
        """Test that max history limit is enforced."""
        # Record more states than max_history
        for state in self.states:
            self.recorder.record_state(state)
        
        # Should only keep the latest max_history states
        history = self.recorder.get_state_history()
        self.assertEqual(len(history), 10)  # max_history is 10
        
        # Should have the most recent states
        self.assertEqual(history[-1]["cycle"], 42)  # Last state's cycle
    
    def test_compression(self):
        """Test compression of state history."""
        # Record several states
        for state in self.states[:6]:
            self.recorder.record_state(state)
        
        # With compression_ratio=2, should have 3 compressed states
        stats = self.recorder.get_statistics()
        self.assertEqual(stats["compressed_history_size"], 3)
    
    def test_get_state_by_cycle(self):
        """Test retrieving state by cycle."""
        # Record states
        for state in self.states[:5]:
            self.recorder.record_state(state)
        
        # Get state at cycle 6
        state = self.recorder.get_state_by_cycle(6)
        self.assertIsNotNone(state)
        self.assertEqual(state["cycle"], 6)
        self.assertEqual(state["registers"]["A"], 2)
        
        # Get state at non-existent cycle should return nearest
        state = self.recorder.get_state_by_cycle(7)
        self.assertIsNotNone(state)
        self.assertEqual(state["cycle"], 6)  # Nearest cycle
    
    def test_get_register_history(self):
        """Test retrieving register history."""
        # Record states
        for state in self.states[:5]:
            self.recorder.record_state(state)
        
        # Get history for register A
        reg_history = self.recorder.get_register_history("A")
        
        # Should have cycles and values lists
        self.assertIn("cycles", reg_history)
        self.assertIn("values", reg_history)
        
        # Should have 5 values
        self.assertEqual(len(reg_history["values"]), 5)
        
        # Values should match what we set
        self.assertEqual(reg_history["values"], [0, 1, 2, 3, 4])
    
    def test_save_and_load_history(self):
        """Test saving and loading history."""
        # Record some states
        for state in self.states[:3]:
            self.recorder.record_state(state)
        
        # Save to a temporary file
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as temp:
            filename = temp.name
        
        try:
            # Save history
            result = self.recorder.save_history(filename, format='pickle')
            self.assertTrue(result)
            
            # Create a new recorder
            new_recorder = StateRecorder(max_history=10)
            
            # Load history
            result = new_recorder.load_history(filename)
            self.assertTrue(result)
            
            # Check if states were loaded
            history = new_recorder.get_state_history()
            self.assertEqual(len(history), 3)
            
            # Check if values match
            self.assertEqual(history[0]["cycle"], 0)
            self.assertEqual(history[1]["cycle"], 3)
            self.assertEqual(history[2]["cycle"], 6)
            
        finally:
            # Clean up
            if os.path.exists(filename):
                os.remove(filename)
    
    def test_record_filter(self):
        """Test filtering of registers during recording."""
        # Create recorder with filter
        filtered_recorder = StateRecorder(record_filter=["A", "PC"])
        
        # Record a state
        state = {
            "cycle": 0,
            "scanline": 0,
            "registers": {
                "A": 10,
                "X": 20,
                "Y": 30,
                "PC": 0x8000
            }
        }
        
        filtered_recorder.record_state(state)
        
        # Check if only filtered registers were recorded
        history = filtered_recorder.get_state_history()
        recorded_registers = history[0]["registers"]
        self.assertIn("A", recorded_registers)
        self.assertIn("PC", recorded_registers)
        self.assertNotIn("X", recorded_registers)
        self.assertNotIn("Y", recorded_registers)
    
    def test_find_register_value_changes(self):
        """Test finding register value changes."""
        # Record states
        for state in self.states[:5]:
            self.recorder.record_state(state)
        
        # Find changes for register A
        changes = self.recorder.find_register_value_changes("A")
        
        # Should have 4 changes (for 5 states)
        self.assertEqual(len(changes), 4)
        
        # Check first change
        first_change = changes[0]
        self.assertEqual(first_change["old_value"], 0)
        self.assertEqual(first_change["new_value"], 1)

if __name__ == '__main__':
    unittest.main()