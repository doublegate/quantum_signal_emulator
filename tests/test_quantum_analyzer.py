"""
Tests for the QuantumAnalyzer module.

This module contains unit tests for the QuantumAnalyzer class, which is responsible
for performing quantum analysis of hardware signals.
"""
import unittest
import numpy as np
from quantum_signal_emulator.analysis.quantum_analyzer import QuantumAnalyzer

class TestQuantumAnalyzer(unittest.TestCase):
    """
    Test cases for the QuantumAnalyzer class.
    """
    
    def setUp(self):
        """Set up test fixtures."""
        # Create analyzer in classical mode to avoid Qiskit dependency in tests
        self.analyzer = QuantumAnalyzer(num_qubits=4, quantum_mode='classical')
        
        # Create sample state history with register transitions
        self.state_history = [
            {
                "cycle": i,
                "scanline": i // 10,
                "dot": i % 10,
                "registers": {
                    "A": int(10 * np.sin(i / 5.0) + 10),
                    "X": i % 4,
                    "Y": i // 3,
                    "PC": 0x8000 + i * 2
                }
            }
            for i in range(50)  # Create 50 sample states with sine wave pattern
        ]
        
        # Create sample signal data
        self.signal_data = np.sin(np.linspace(0, 4*np.pi, 100))
    
    def test_extract_signal_from_states(self):
        """Test extracting signal data from state history."""
        signal_data = self.analyzer._extract_signal_from_states(self.state_history)
        
        # Signal should not be empty
        self.assertGreater(len(signal_data), 0)
        
        # Signal should have one dimension after PCA
        self.assertEqual(len(signal_data.shape), 1)
    
    def test_classical_analysis(self):
        """Test classical signal analysis."""
        results = self.analyzer._classical_analysis(self.signal_data)
        
        # Check if results have expected keys
        self.assertIn("method", results)
        self.assertEqual(results["method"], "classical")
        self.assertIn("frequency_data", results)
        self.assertIn("entropy", results)
        self.assertIn("repeating_patterns", results)
        self.assertIn("analysis_summary", results)
        
        # Frequency data should contain frequencies and amplitudes
        freq_data = results["frequency_data"]
        self.assertIn("frequencies", freq_data)
        self.assertIn("amplitudes", freq_data)
        
        # Entropy should be a positive value
        self.assertGreater(results["entropy"], 0)
        
        # Analysis summary should be a non-empty string
        self.assertIsInstance(results["analysis_summary"], str)
        self.assertGreater(len(results["analysis_summary"]), 0)
    
    def test_analyze_hardware_state(self):
        """Test complete hardware state analysis."""
        results = self.analyzer.analyze_hardware_state(self.state_history)
        
        # Check if results have expected keys depending on mode
        if results.get("method") == "quantum":
            self.assertIn("quantum_fingerprint", results)
            self.assertIn("quantum_entropy", results)
        else:
            self.assertIn("frequency_data", results)
            self.assertIn("analysis_summary", results)
        
    def test_empty_state_history(self):
        """Test behavior with empty state history."""
        empty_history = []
        results = self.analyzer.analyze_hardware_state(empty_history)
        
        # Should return error
        self.assertIn("error", results)
    
    def test_calculate_coherence_measure(self):
        """Test calculation of coherence measure."""
        # Create simple frequency data
        freq_data = {
            "amplitudes": [0.1, 0.5, 0.2, 0.1, 0.1],
            "frequency_bins": [1, 2, 3, 4, 5]
        }
        
        # Calculate coherence
        coherence = self.analyzer._calculate_coherence_measure(freq_data)
        
        # Coherence should be between 0 and 1
        self.assertGreaterEqual(coherence, 0.0)
        self.assertLessEqual(coherence, 1.0)
    
    def test_process_quantum_results(self):
        """Test processing of simulated quantum results."""
        # Mock counts dictionary from quantum measurement
        counts = {
            '0000': 100,
            '0001': 50,
            '0010': 200,
            '0100': 30,
            '1000': 20
        }
        
        results = self.analyzer._process_quantum_results(counts)
        
        # Check if results have expected structure
        self.assertIn("frequency_bins", results)
        self.assertIn("amplitudes", results)
        self.assertIn("dominant_frequencies", results)
        
        # Check if dominant frequencies are sorted by amplitude
        dom_freqs = results["dominant_frequencies"]
        if len(dom_freqs) >= 2:
            self.assertGreaterEqual(dom_freqs[0]["amplitude"], dom_freqs[1]["amplitude"])
    
    def test_calculate_quantum_entropy(self):
        """Test calculation of quantum entropy."""
        # Mock counts dictionary
        counts = {
            '0000': 100,
            '0001': 50,
            '0010': 200,
            '0100': 30,
            '1000': 20
        }
        
        entropy = self.analyzer._calculate_quantum_entropy(counts)
        
        # Entropy should be positive
        self.assertGreater(entropy, 0)
        
        # For 5 states with these probabilities, entropy should be less than 3 bits
        self.assertLess(entropy, 3.0)
        
        # For uniform distribution, entropy should be higher
        uniform_counts = {format(i, '04b'): 25 for i in range(16)}
        uniform_entropy = self.analyzer._calculate_quantum_entropy(uniform_counts)
        self.assertGreater(uniform_entropy, entropy)

if __name__ == '__main__':
    unittest.main()