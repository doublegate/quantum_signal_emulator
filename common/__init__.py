"""
Common functionality shared across all system implementations.
"""
from .interfaces import CPU, Memory, VideoProcessor, AudioProcessor, System
from .visualizer import SignalVisualizer
from .quantum_processor import QuantumSignalProcessor