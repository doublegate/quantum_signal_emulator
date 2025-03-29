import unittest
from quantum_signal_emulator.common.quantum_processor import QuantumSignalProcessor
import numpy as np

class TestQuantumProcessor(unittest.TestCase):
    def setUp(self):
        self.processor = QuantumSignalProcessor(num_qubits=4)
        
    def test_quantum_entropy(self):
        # Create a simple test signal
        test_signal = np.sin(np.linspace(0, 2*np.pi, 16))
        
        # Calculate entropy
        counts = {'0000': 100, '0001': 50, '0010': 25, '0011': 10}
        entropy = self.processor._calculate_quantum_entropy(counts)
        
        # Entropy should be positive
        self.assertGreater(entropy, 0)
        
        # Maximum possible entropy for 4 qubits would be 4.0
        self.assertLessEqual(entropy, 4.0)

if __name__ == '__main__':
    unittest.main()
