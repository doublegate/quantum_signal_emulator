"""
Main entry point for the Quantum Signal Emulator.

This module provides the main entry point for the Quantum Signal Emulator,
including argument parsing and execution flow control.
"""

import argparse
import logging
import time
import sys
from typing import Dict, Any, List, Optional

# Import from the package with absolute imports
from systems.system_factory import SystemFactory
from common.quantum_processor import QuantumSignalProcessor
from common.visualizer import SignalVisualizer
from analysis.cycle_analyzer import CycleAnalyzer
from analysis.state_recorder import StateRecorder
from constants import ANALYSIS_MODES

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("QuantumSignalEmulator")

def main() -> None:
    """
    Main function to parse arguments and run the program.
    
    This function parses command-line arguments, initializes the system and analysis
    tools, and runs the simulation with the specified parameters.
    """
    parser = argparse.ArgumentParser(description="Quantum Signal Emulator for Hardware Cycle Analysis")
    parser.add_argument('--system', type=str, choices=["nes", "snes", "genesis", "atari2600"], default='nes',
                       help='System type to emulate')
    parser.add_argument('--rom', type=str, help='Path to ROM file')
    parser.add_argument('--analysis-mode', type=str, choices=ANALYSIS_MODES,
                       default='hybrid', help='Analysis mode')
    parser.add_argument('--frames', type=int, default=1, help='Number of frames to simulate')
    parser.add_argument('--output', type=str, choices=['json', 'csv', 'binary'], default='json',
                      help='Output format for analysis results')
    parser.add_argument('--no-gpu', action='store_true', help='Disable GPU acceleration')
    parser.add_argument('--no-3d', action='store_true', help='Disable 3D visualizations')
    parser.add_argument('--no-visualization', action='store_true', help='Disable visualizations')
    parser.add_argument('--save-state', type=str, help='Path to save final state')
    parser.add_argument('--load-state', type=str, help='Path to load initial state')
    parser.add_argument('--config', type=str, help='Path to custom configuration file')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--log-level', type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO', help='Logging level')
    
    args = parser.parse_args()
    
    # Set logging level
    log_level = getattr(logging, args.log_level)
    if args.debug:
        log_level = logging.DEBUG
    logger.setLevel(log_level)
    
    # Print header
    print("="*80)
    print(f"  Quantum Signal Emulator for Hardware Cycle Analysis")
    print(f"  System: {args.system}")
    print(f"  Analysis Mode: {args.analysis_mode}")
    print(f"  Frames: {args.frames}")
    print("="*80)
    
    # Load custom configuration if provided
    custom_config = None
    if args.config:
        try:
            import json
            with open(args.config, 'r') as f:
                custom_config = json.load(f)
            logger.info(f"Loaded custom configuration from {args.config}")
        except Exception as e:
            logger.error(f"Error loading custom configuration: {e}")
            return 1
    
    # Create system
    try:
        system = SystemFactory.create_system(args.system, custom_config)
    except ValueError as e:
        logger.error(f"Error creating system: {e}")
        return 1
    
    # Initialize state recorder
    state_recorder = StateRecorder()
    
    # Register state recorder with system if supported
    if hasattr(system, 'register_state_recorder'):
        system.register_state_recorder(state_recorder)
    
    # Initialize analysis tools
    quantum_processor = QuantumSignalProcessor(num_qubits=8)
    visualizer = SignalVisualizer(use_3d=not args.no_3d)
    cycle_analyzer = CycleAnalyzer()
    
    # Load state if requested
    if args.load_state:
        try:
            logger.info(f"Loading state from {args.load_state}")
            state_recorder.load_history(args.load_state)
            
            # Apply loaded state to system if supported
            if hasattr(system, 'restore_state'):
                last_state = state_recorder.get_state_history()[-1] if state_recorder.state_history else None
                if last_state:
                    system.restore_state(last_state)
                    logger.info("Restored system state")
                else:
                    logger.warning("No state found in loaded history")
        except Exception as e:
            logger.error(f"Error loading state: {e}")
            return 1
    
    # Load ROM if provided
    if args.rom:
        try:
            logger.info(f"Loading ROM: {args.rom}")
            system.load_rom(args.rom)
            system.reset()
        except Exception as e:
            logger.error(f"Error loading ROM: {e}")
            return 1
    else:
        # If no ROM, run synthetic simulation if supported
        logger.info("No ROM provided, running synthetic simulation")
        if hasattr(system, 'run_synthetic'):
            system.run_synthetic()
        else:
            logger.warning(f"{args.system} does not support synthetic simulation without a ROM")
    
    # Run simulation
    start_time = time.time()
    
    logger.info(f"Running simulation for {args.frames} frames")
    for frame in range(args.frames):
        logger.info(f"Frame {frame+1}/{args.frames}")
        
        # Run one frame
        try:
            system_state = system.run_frame()
            
            # Record state if not already done by the system
            if not hasattr(system, 'register_state_recorder'):
                state_recorder.record_state(system_state)
                
        except Exception as e:
            logger.error(f"Error running frame {frame+1}: {e}")
            if args.debug:
                import traceback
                traceback.print_exc()
            break
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    # Save final state if requested
    if args.save_state:
        try:
            logger.info(f"Saving state to {args.save_state}")
            state_recorder.save_history(args.save_state, format=args.output)
        except Exception as e:
            logger.error(f"Error saving state: {e}")
    
    # Get state history for analysis
    state_history = state_recorder.get_state_history()
    
    # Skip analysis if no state history
    if not state_history:
        logger.warning("No state history available for analysis")
        return 1
    
    # Calculate performance metrics
    total_cycles = 0
    if hasattr(system, 'cycle_count'):
        total_cycles = system.cycle_count
    elif state_history and "cycle" in state_history[-1]:
        total_cycles = state_history[-1]["cycle"]
    
    cycles_per_second = total_cycles / execution_time if execution_time > 0 else 0
    
    # Perform quantum analysis if requested
    if args.analysis_mode in ['quantum', 'hybrid']:
        try:
            # Extract signal data from state history
            signal_data = cycle_analyzer.extract_signal_data(state_history)
            
            # Run quantum analysis
            quantum_results = quantum_processor.analyze_signal(signal_data)
            
            # Visualize quantum results if not disabled
            if not args.no_visualization and frame == args.frames - 1:  # Only visualize last frame
                visualizer.plot_quantum_results(quantum_results)
                
        except Exception as e:
            logger.error(f"Error in quantum analysis: {e}")
            if args.debug:
                import traceback
                traceback.print_exc()
    
    # Perform cycle analysis
    try:
        timing_analysis = cycle_analyzer.analyze_timing_patterns(state_history)
        register_analysis = cycle_analyzer.analyze_register_activity(state_history)
        
        # Print summary of timing analysis
        print("\nTiming Analysis Summary:")
        if "statistics" in timing_analysis:
            stats = timing_analysis["statistics"]
            print(f"Total cycles: {stats.get('total_cycles', 'N/A')}")
            print(f"Min cycle delta: {stats.get('min_cycle_delta', 'N/A')}")
            print(f"Max cycle delta: {stats.get('max_cycle_delta', 'N/A')}")
            print(f"Avg cycle delta: {stats.get('avg_cycle_delta', 'N/A')}")
            
        # Print most active registers
        print("\nMost Active Registers:")
        for reg_name, analysis in list(register_analysis.items())[:5]:
            print(f"{reg_name}: {analysis.get('change_frequency', 0)*100:.2f}% change frequency")
            
    except Exception as e:
        logger.error(f"Error in cycle analysis: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
    
    # Visualize results if not disabled
    if not args.no_visualization:
        try:
            visualizer.visualize_cycle_timing(system)
            visualizer.plot_register_states(system)
            visualizer.visualize_state_space(system)
        except Exception as e:
            logger.error(f"Error in visualization: {e}")
            if args.debug:
                import traceback
                traceback.print_exc()
    
    # Print summary
    print("\nSimulation Results Summary:")
    print(f"System: {args.system}")
    print(f"Frames simulated: {args.frames}")
    print(f"Cycles simulated: {total_cycles}")
    print(f"Execution time: {execution_time:.2f} seconds")
    print(f"Performance: {cycles_per_second:.2f} cycles/second")
    
    logger.info("Quantum Signal Emulator execution complete")
    return 0

if __name__ == "__main__":
    sys.exit(main())