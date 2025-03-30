#!/usr/bin/env python3
"""
Basic analysis example for the Quantum Signal Emulator.

This example demonstrates how to perform basic cycle analysis on a NES system
using the Quantum Signal Emulator. It shows how to load a ROM, run emulation,
and analyze the results.

Usage:
    python basic_analysis.py --rom <path_to_rom>
"""
import argparse
import logging
import sys
import os

# Add parent directory to path to allow running from examples directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from quantum_signal_emulator.systems.system_factory import SystemFactory
from quantum_signal_emulator.analysis.cycle_analyzer import CycleAnalyzer
from quantum_signal_emulator.common.visualizer import SignalVisualizer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("BasicAnalysisExample")

def main():
    """Run the basic analysis example."""
    parser = argparse.ArgumentParser(description="Basic analysis example for Quantum Signal Emulator")
    parser.add_argument('--rom', type=str, help='Path to ROM file')
    parser.add_argument('--frames', type=int, default=1, help='Number of frames to analyze')
    parser.add_argument('--system', type=str, choices=['nes', 'snes', 'genesis', 'atari2600'],
                        default='nes', help='System to analyze')
    
    args = parser.parse_args()
    
    if not args.rom:
        logger.error("No ROM file specified. Use --rom to specify a ROM file.")
        return 1
    
    # Create system
    logger.info(f"Creating {args.system} system")
    try:
        system = SystemFactory.create_system(args.system)
    except ValueError as e:
        logger.error(f"Error creating system: {e}")
        return 1
    
    # Load ROM
    logger.info(f"Loading ROM: {args.rom}")
    try:
        system.load_rom(args.rom)
        system.reset()
    except Exception as e:
        logger.error(f"Error loading ROM: {e}")
        return 1
    
    # Create analysis tools
    cycle_analyzer = CycleAnalyzer()
    visualizer = SignalVisualizer()
    
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
    
    # Analyze timing patterns
    logger.info("Analyzing timing patterns")
    timing_results = cycle_analyzer.analyze_timing_patterns(state_history)
    
    # Print timing statistics
    if "statistics" in timing_results:
        stats = timing_results["statistics"]
        print("\nTiming Statistics:")
        print(f"Total cycles: {stats.get('total_cycles', 'N/A')}")
        print(f"Min cycle delta: {stats.get('min_cycle_delta', 'N/A')}")
        print(f"Max cycle delta: {stats.get('max_cycle_delta', 'N/A')}")
        print(f"Avg cycle delta: {stats.get('avg_cycle_delta', 'N/A'):.2f}")
    
    # Analyze register activity
    logger.info("Analyzing register activity")
    register_results = cycle_analyzer.analyze_register_activity(state_history)
    
    # Print register statistics
    print("\nRegister Activity:")
    for reg_name, reg_data in list(register_results.items())[:5]:  # Show top 5 registers
        print(f"{reg_name}:")
        print(f"  Change frequency: {reg_data.get('change_frequency', 0)*100:.2f}%")
        print(f"  Min value: {reg_data.get('min_value', 'N/A')}")
        print(f"  Max value: {reg_data.get('max_value', 'N/A')}")
        print(f"  Mean value: {reg_data.get('mean_value', 'N/A'):.2f}")
    
    # Visualize cycle timing
    logger.info("Visualizing cycle timing")
    visualizer.visualize_cycle_timing(system)
    
    # Visualize register states
    logger.info("Visualizing register states")
    visualizer.plot_register_states(system)
    
    logger.info("Basic analysis complete")
    return 0

if __name__ == "__main__":
    sys.exit(main())