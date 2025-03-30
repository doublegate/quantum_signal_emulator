#!/usr/bin/env python3
"""
Quantum analysis example for the Quantum Signal Emulator.

This example demonstrates how to perform quantum-inspired analysis on a video game system
using the Quantum Signal Emulator. It shows how to extract signal data, perform quantum
analysis, and visualize the results.

Usage:
    python quantum_analysis.py --rom <path_to_rom> --system <system_type>
"""
import argparse
import logging
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path to allow running from examples directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from quantum_signal_emulator.systems.system_factory import SystemFactory
from quantum_signal_emulator.analysis.cycle_analyzer import CycleAnalyzer
from quantum_signal_emulator.analysis.quantum_analyzer import QuantumAnalyzer
from quantum_signal_emulator.common.visualizer import SignalVisualizer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("QuantumAnalysisExample")

def main():
    """Run the quantum analysis example."""
    parser = argparse.ArgumentParser(description="Quantum analysis example for Quantum Signal Emulator")
    parser.add_argument('--rom', type=str, help='Path to ROM file')
    parser.add_argument('--frames', type=int, default=1, help='Number of frames to analyze')
    parser.add_argument('--system', type=str, choices=['nes', 'snes', 'genesis', 'atari2600'],
                        default='nes', help='System to analyze')
    parser.add_argument('--qubits', type=int, default=8, help='Number of qubits for quantum analysis')
    parser.add_argument('--mode', type=str, choices=['quantum', 'classical', 'hybrid'],
                        default='hybrid', help='Analysis mode')
    parser.add_argument('--no-synthetic', action='store_true',
                        help='Disable synthetic signal generation if no ROM provided')
    
    args = parser.parse_args()
    
    # Create system
    logger.info(f"Creating {args.system} system")
    try:
        system = SystemFactory.create_system(args.system)
    except ValueError as e:
        logger.error(f"Error creating system: {e}")
        return 1
    
    # Create analysis tools
    cycle_analyzer = CycleAnalyzer()
    quantum_analyzer = QuantumAnalyzer(num_qubits=args.qubits, quantum_mode=args.mode)
    visualizer = SignalVisualizer()
    
    # Setup signal source
    if args.rom:
        # Load ROM
        logger.info(f"Loading ROM: {args.rom}")
        try:
            system.load_rom(args.rom)
            system.reset()
        except Exception as e:
            logger.error(f"Error loading ROM: {e}")
            return 1
            
        # Run emulation
        logger.info(f"Running {args.frames} frames of emulation")
        for i in range(args.frames):
            logger.info(f"Running frame {i+1}/{args.frames}")
            system.run_frame()
        
        # Get state history
        if hasattr(system, 'state_history'):
            state_history = system.state_history
        else:
            logger.error("System does not provide state history")
            return 1
            
        # Extract signal from state history
        logger.info("Extracting signal from state history")
        signal_data = cycle_analyzer.extract_signal_data(state_history)
        
    elif not args.no_synthetic:
        # Generate synthetic signal for demonstration
        logger.info("Generating synthetic signal for analysis")
        
        # Create a complex synthetic signal with multiple frequency components
        t = np.linspace(0, 10, 1000)  # Time array
        
        # Base signal (combination of sine waves)
        signal = np.sin(2 * np.pi * 1.0 * t)          # 1 Hz component
        signal += 0.5 * np.sin(2 * np.pi * 2.0 * t)   # 2 Hz component
        signal += 0.3 * np.sin(2 * np.pi * 3.5 * t)   # 3.5 Hz component
        
        # Add some noise
        signal += 0.1 * np.random.randn(len(t))
        
        # Add a "register jump" effect
        for i in range(100, 200):
            signal[i] += 1.0
        for i in range(500, 600):
            signal[i] -= 0.8
            
        # Create synthetic state history
        state_history = []
        for i in range(len(signal)):
            state = {
                "cycle": i,
                "scanline": i // 100,
                "dot": i % 100,
                "registers": {
                    "A": int(10 * signal[i] + 10),
                    "X": int(5 * np.cos(t[i]) + 5),
                    "Y": int(7 * np.sin(2 * t[i]) + 7),
                    "PC": 0x8000 + i * 2
                }
            }
            state_history.append(state)
            
        # Extract signal for analysis
        signal_data = signal
        
        # Plot the synthetic signal
        plt.figure(figsize=(10, 6))
        plt.plot(t, signal)
        plt.title("Synthetic Signal for Quantum Analysis")
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.show()
        
    else:
        logger.error("No ROM provided and synthetic signal generation disabled")
        return 1
    
    # Perform quantum analysis
    logger.info(f"Performing {args.mode} analysis with {args.qubits} qubits")
    quantum_results = quantum_analyzer.analyze_hardware_state(state_history)
    
    # Print analysis results
    print("\nQuantum Analysis Results:")
    print(f"Analysis method: {quantum_results.get('method', 'Unknown')}")
    
    if "quantum_entropy" in quantum_results:
        print(f"Quantum entropy: {quantum_results['quantum_entropy']:.4f} bits")
        
    if "coherence_measure" in quantum_results:
        print(f"Coherence measure: {quantum_results['coherence_measure']:.4f}")
        
    if "analysis_summary" in quantum_results:
        print(f"\nAnalysis summary:")
        print(quantum_results["analysis_summary"])
    
    # Visualize quantum results
    logger.info("Visualizing quantum analysis results")
    visualizer.plot_quantum_results(quantum_results)
    
    # Visualize state space
    logger.info("Visualizing system state space")
    visualizer.visualize_state_space(system)
    
    logger.info("Quantum analysis complete")
    return 0

if __name__ == "__main__":
    sys.exit(main())