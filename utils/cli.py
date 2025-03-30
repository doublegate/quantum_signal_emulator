#!/usr/bin/env python3
"""
Command-line interface for the Quantum Signal Emulator.

This module provides a CLI for quick analysis and utility functions without
needing to run the full emulator. It includes commands for ROM analysis,
state inspection, data conversion, and other utilities.
"""

import argparse
import logging
import os
import sys
import json
import yaml
from typing import Dict, List, Tuple, Optional, Any, Union
import importlib.util

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("QuantumSignalEmulator.CLI")

def get_base_parser() -> argparse.ArgumentParser:
    """
    Create base argument parser for command-line interface.
    
    Returns:
        Base argument parser
    """
    parser = argparse.ArgumentParser(
        description="Quantum Signal Emulator CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  analyze   Analyze a ROM file
  convert   Convert between data formats
  extract   Extract data from a ROM
  visualize Create visualizations from data
  validate  Validate configuration or data
  import    Import data from external tools
  help      Show help for commands
""")
    
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress output except errors")
    parser.add_argument("--output", "-o", help="Output file path")
    parser.add_argument("--format", "-f", choices=["json", "yaml", "csv", "text"],
                       default="text", help="Output format")
    
    return parser

def setup_analyze_parser(subparsers) -> None:
    """
    Setup parser for 'analyze' command.
    
    Args:
        subparsers: Subparsers object from base parser
    """
    parser = subparsers.add_parser("analyze", help="Analyze a ROM file")
    
    parser.add_argument("rom_path", help="Path to ROM file")
    parser.add_argument("--system", "-s", choices=["nes", "snes", "genesis", "atari2600"],
                       help="System type (auto-detect if not specified)")
    parser.add_argument("--mode", "-m", choices=["basic", "deep", "timing", "quantum", "all"],
                       default="basic", help="Analysis mode")
    parser.add_argument("--quantum-qubits", type=int, default=8,
                       help="Number of qubits for quantum analysis")
    parser.add_argument("--timing-precision", type=float, default=0.1,
                       help="Timing precision threshold (ns)")
    parser.add_argument("--frames", "-n", type=int, default=1,
                       help="Number of frames to analyze (for 'deep' mode)")

def setup_convert_parser(subparsers) -> None:
    """
    Setup parser for 'convert' command.
    
    Args:
        subparsers: Subparsers object from base parser
    """
    parser = subparsers.add_parser("convert", help="Convert between data formats")
    
    parser.add_argument("input_path", help="Path to input file")
    parser.add_argument("--input-format", "-i", choices=["auto", "json", "csv", "binary", "npy", "log"],
                       default="auto", help="Input format")
    parser.add_argument("--system", "-s", choices=["nes", "snes", "genesis", "atari2600"],
                       help="System type (required for some formats)")
    parser.add_argument("--to", "-t", choices=["json", "csv", "numpy", "text", "image"],
                       required=True, help="Output format")

def setup_extract_parser(subparsers) -> None:
    """
    Setup parser for 'extract' command.
    
    Args:
        subparsers: Subparsers object from base parser
    """
    parser = subparsers.add_parser("extract", help="Extract data from a ROM")
    
    parser.add_argument("rom_path", help="Path to ROM file")
    parser.add_argument("--system", "-s", choices=["nes", "snes", "genesis", "atari2600"],
                       help="System type (auto-detect if not specified)")
    parser.add_argument("--extract-type", "-e", choices=["header", "prg", "chr", "audio", "all"],
                       default="header", help="Type of data to extract")
    parser.add_argument("--output-dir", "-d", help="Output directory for extracted files")

def setup_visualize_parser(subparsers) -> None:
    """
    Setup parser for 'visualize' command.
    
    Args:
        subparsers: Subparsers object from base parser
    """
    parser = subparsers.add_parser("visualize", help="Create visualizations from data")
    
    parser.add_argument("input_path", help="Path to input data file")
    parser.add_argument("--type", "-t", choices=["timing", "registers", "signals", "state-space", "quantum", "all"],
                       default="timing", help="Visualization type")
    parser.add_argument("--input-format", "-i", choices=["auto", "json", "csv", "numpy", "log"],
                       default="auto", help="Input format")
    parser.add_argument("--no-3d", action="store_true", help="Disable 3D visualizations")
    parser.add_argument("--dark-mode", action="store_true", help="Use dark mode for plots")
    parser.add_argument("--register", "-r", action="append", help="Specific registers to visualize")
    parser.add_argument("--save-image", "-s", help="Save visualization to image file")

def setup_validate_parser(subparsers) -> None:
    """
    Setup parser for 'validate' command.
    
    Args:
        subparsers: Subparsers object from base parser
    """
    parser = subparsers.add_parser("validate", help="Validate configuration or data")
    
    parser.add_argument("input_path", help="Path to file to validate")
    parser.add_argument("--type", "-t", choices=["config", "state", "signals", "auto"],
                       default="auto", help="Type of file to validate")
    parser.add_argument("--schema", "-s", help="Path to custom schema file")
    parser.add_argument("--strict", action="store_true", help="Enable strict validation")

def setup_import_parser(subparsers) -> None:
    """
    Setup parser for 'import' command.
    
    Args:
        subparsers: Subparsers object from base parser
    """
    parser = subparsers.add_parser("import", help="Import data from external tools")
    
    parser.add_argument("input_path", help="Path to input file")
    parser.add_argument("--source", "-s", choices=["fceux", "mesen", "bizhawk", "mame", 
                                                "genesis_plus_gx", "stella", "logic_analyzer", "auto"],
                       default="auto", help="Source format")
    parser.add_argument("--system", choices=["nes", "snes", "genesis", "atari2600"],
                       help="System type (auto-detect if not specified)")
    parser.add_argument("--convert", "-c", action="store_true",
                       help="Convert to standard state history format")

def setup_help_parser(subparsers) -> None:
    """
    Setup parser for 'help' command.
    
    Args:
        subparsers: Subparsers object from base parser
    """
    parser = subparsers.add_parser("help", help="Show help for commands")
    parser.add_argument("command", nargs="?", help="Command to show help for")

def ensure_module(module_name: str) -> bool:
    """
    Ensure a module is available, logging error if not.
    
    Args:
        module_name: Name of module to check
        
    Returns:
        True if module is available, False otherwise
    """
    if importlib.util.find_spec(module_name) is None:
        logger.error(f"Required module {module_name} not found. Please install it using:")
        logger.error(f"  pip install {module_name}")
        return False
    return True

def handle_analyze_command(args) -> int:
    """
    Handle 'analyze' command.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Exit code (0 for success, non-zero for error)
    """
    # Ensure required modules
    if not ensure_module("quantum_signal_emulator"):
        return 1
    
    try:
        from quantum_signal_emulator.utils.data_extractor import DataExtractor
        
        logger.info(f"Analyzing ROM: {args.rom_path}")
        
        # Create data extractor
        extractor = DataExtractor(debug_mode=args.debug)
        
        # Load ROM
        if not extractor.load_rom(args.rom_path):
            logger.error("Failed to load ROM")
            return 1
        
        # Perform analysis based on mode
        if args.mode == "basic":
            # Basic header analysis
            result = extractor.export_analysis(args.output or "rom_analysis.json", 
                                            format=args.format if args.format != "text" else "json")
            
            if not result:
                logger.error("Failed to export analysis")
                return 1
                
            if args.output:
                logger.info(f"Analysis saved to {args.output}")
            else:
                # Print analysis to console
                if args.format == "text":
                    header_data = extractor.header_data
                    print("\nROM Analysis:")
                    print("="*50)
                    print(f"System: {header_data.get('system', extractor.system_type)}")
                    print(f"Format: {header_data.get('format', 'Unknown')}")
                    
                    # Print more details
                    if "rom_size" in header_data:
                        size_kb = header_data["rom_size"] / 1024
                        print(f"ROM Size: {size_kb:.1f} KB")
                    
                    if hasattr(extractor, 'code_analysis'):
                        print("\nCode Analysis:")
                        for key, value in extractor.code_analysis.items():
                            if key != "common_opcodes":
                                print(f"  {key}: {value}")
                        
                        print("  Common opcodes:")
                        for opcode, count in extractor.code_analysis.get("common_opcodes", [])[:5]:
                            print(f"    0x{opcode:02X}: {count} occurrences")
                
        elif args.mode == "timing":
            # Timing analysis
            timing_data = extractor.extract_instruction_timing()
            
            if "error" in timing_data:
                logger.error(f"Timing analysis failed: {timing_data['error']}")
                return 1
                
            # Save or print results
            if args.output:
                with open(args.output, 'w') as f:
                    if args.format == "json":
                        json.dump(timing_data, f, indent=2)
                    elif args.format == "yaml":
                        yaml.dump(timing_data, f, default_flow_style=False)
                    elif args.format == "csv":
                        # Convert to simple CSV format
                        import csv
                        writer = csv.writer(f)
                        writer.writerow(["key", "value"])
                        
                        def flatten_dict(d, prefix=""):
                            for k, v in d.items():
                                if isinstance(v, dict):
                                    yield from flatten_dict(v, f"{prefix}{k}.")
                                elif isinstance(v, list):
                                    writer.writerow([f"{prefix}{k}", f"<list with {len(v)} items>"])
                                else:
                                    yield [f"{prefix}{k}", v]
                        
                        writer.writerows(flatten_dict(timing_data))
                    else:
                        # Text format
                        f.write("Timing Analysis:\n")
                        f.write("="*50 + "\n")
                        
                        for key, value in timing_data.items():
                            if isinstance(value, dict):
                                f.write(f"\n{key}:\n")
                                for k, v in value.items():
                                    f.write(f"  {k}: {v}\n")
                            elif isinstance(value, list):
                                f.write(f"\n{key}:\n")
                                for item in value[:10]:  # Limit to 10 items
                                    f.write(f"  {item}\n")
                                if len(value) > 10:
                                    f.write(f"  ... and {len(value) - 10} more items\n")
                            else:
                                f.write(f"{key}: {value}\n")
                
                logger.info(f"Timing analysis saved to {args.output}")
            else:
                # Print to console
                print("\nTiming Analysis:")
                print("="*50)
                
                if "total_instructions" in timing_data:
                    print(f"Total instructions: {timing_data['total_instructions']}")
                
                if "total_cycles" in timing_data:
                    print(f"Total cycles: {timing_data['total_cycles']}")
                
                if "common_sequences" in timing_data:
                    print("\nCommon instruction sequences:")
                    for i, seq in enumerate(timing_data["common_sequences"][:5]):
                        opcodes_hex = [f"0x{op:02X}" for op in seq.get("opcodes", [])]
                        print(f"  Sequence {i+1}: {opcodes_hex}")
                        print(f"    Occurrences: {seq.get('occurrences', 0)}")
                        print(f"    Total cycles: {seq.get('total_cycles', 0)}")
        
        elif args.mode == "deep" or args.mode == "quantum" or args.mode == "all":
            # These modes require emulation - need to run the actual emulator
            logger.info(f"Analysis mode '{args.mode}' requires running the emulator")
            logger.info("Please use the main emulator command for this analysis type:")
            print(f"\n  quantum-signal-emulator --system {args.system or 'auto'} --rom {args.rom_path} " +
                 f"--analysis-mode {'quantum' if args.mode in ['quantum', 'all'] else 'hybrid'} " +
                 f"--frames {args.frames}")
            
            return 0
            
        return 0
        
    except ImportError as e:
        logger.error(f"Required module not found: {e}")
        return 1
    except Exception as e:
        logger.error(f"Error during analysis: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1

def handle_convert_command(args) -> int:
    """
    Handle 'convert' command.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Exit code (0 for success, non-zero for error)
    """
    try:
        logger.info(f"Converting {args.input_path} to {args.to} format")
        
        # Determine input format if auto
        input_format = args.input_format
        if input_format == "auto":
            _, ext = os.path.splitext(args.input_path)
            ext = ext.lower()
            
            if ext in ['.json']:
                input_format = "json"
            elif ext in ['.csv']:
                input_format = "csv"
            elif ext in ['.bin', '.dat']:
                input_format = "binary"
            elif ext in ['.npy', '.npz']:
                input_format = "npy"
            elif ext in ['.log', '.txt']:
                input_format = "log"
            else:
                logger.error(f"Could not determine format for {args.input_path}")
                return 1
        
        # Determine output path
        output_path = args.output
        if not output_path:
            base_name, _ = os.path.splitext(args.input_path)
            
            if args.to == "json":
                output_path = f"{base_name}.json"
            elif args.to == "csv":
                output_path = f"{base_name}.csv"
            elif args.to == "numpy":
                output_path = f"{base_name}.npy"
            elif args.to == "text":
                output_path = f"{base_name}.txt"
            elif args.to == "image":
                output_path = f"{base_name}.png"
        
        # Load input file
        if input_format == "json":
            # Ensure required module
            if not ensure_module("json"):
                return 1
                
            with open(args.input_path, 'r') as f:
                data = json.load(f)
                
        elif input_format == "csv":
            # Ensure required module
            if not ensure_module("pandas"):
                return 1
                
            import pandas as pd
            data = pd.read_csv(args.input_path).to_dict()
            
        elif input_format == "npy":
            # Ensure required module
            if not ensure_module("numpy"):
                return 1
                
            import numpy as np
            data = np.load(args.input_path, allow_pickle=True)
            
        elif input_format == "binary":
            with open(args.input_path, 'rb') as f:
                data = f.read()
                
        elif input_format == "log":
            with open(args.input_path, 'r') as f:
                data = f.readlines()
                
        # Convert to output format
        if args.to == "json":
            # Ensure required module
            if not ensure_module("json"):
                return 1
                
            # Convert to JSON
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x))
                
        elif args.to == "csv":
            # Ensure required module
            if not ensure_module("pandas"):
                return 1
                
            import pandas as pd
            
            # Convert to DataFrame
            if isinstance(data, dict):
                df = pd.DataFrame(data)
            elif isinstance(data, list):
                df = pd.DataFrame(data)
            else:
                logger.error(f"Could not convert {type(data)} to CSV")
                return 1
                
            # Save to CSV
            df.to_csv(output_path, index=False)
            
        elif args.to == "numpy":
            # Ensure required module
            if not ensure_module("numpy"):
                return 1
                
            import numpy as np
            
            # Convert to NumPy array
            if isinstance(data, dict):
                np.savez(output_path, **data)
            else:
                np.save(output_path, data)
                
        elif args.to == "text":
            # Convert to text
            with open(output_path, 'w') as f:
                if isinstance(data, dict):
                    yaml.dump(data, f, default_flow_style=False)
                elif isinstance(data, list):
                    for item in data:
                        f.write(f"{item}\n")
                else:
                    f.write(str(data))
                    
        elif args.to == "image":
            # Ensure required modules
            if not ensure_module("numpy") or not ensure_module("matplotlib"):
                return 1
                
            import numpy as np
            import matplotlib.pyplot as plt
            
            # Convert to image
            if hasattr(data, 'shape') and len(data.shape) >= 2:
                # NumPy array - plot as image
                plt.figure(figsize=(10, 8))
                plt.imshow(data, cmap='viridis')
                plt.colorbar()
                plt.title(f"Data from {os.path.basename(args.input_path)}")
                plt.savefig(output_path, dpi=300)
                plt.close()
            elif isinstance(data, dict) and any(hasattr(v, 'shape') for v in data.values()):
                # Dict with arrays - plot first suitable array
                for key, value in data.items():
                    if hasattr(value, 'shape') and len(value.shape) >= 1:
                        plt.figure(figsize=(10, 8))
                        
                        if len(value.shape) == 1:
                            # 1D array - line plot
                            plt.plot(value)
                            plt.title(f"{key} from {os.path.basename(args.input_path)}")
                            plt.xlabel("Index")
                            plt.ylabel("Value")
                        else:
                            # 2D array - heatmap
                            plt.imshow(value, cmap='viridis')
                            plt.colorbar()
                            plt.title(f"{key} from {os.path.basename(args.input_path)}")
                            
                        plt.savefig(output_path, dpi=300)
                        plt.close()
                        break
            else:
                logger.error(f"Could not convert {type(data)} to image")
                return 1
        
        logger.info(f"Conversion complete. Output saved to {output_path}")
        return 0
        
    except Exception as e:
        logger.error(f"Error during conversion: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1

def handle_extract_command(args) -> int:
    """
    Handle 'extract' command.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Exit code (0 for success, non-zero for error)
    """
    # Ensure required modules
    if not ensure_module("quantum_signal_emulator"):
        return 1
    
    try:
        from quantum_signal_emulator.utils.data_extractor import DataExtractor
        
        logger.info(f"Extracting data from ROM: {args.rom_path}")
        
        # Create data extractor
        extractor = DataExtractor(debug_mode=args.debug)
        
        # Load ROM
        if not extractor.load_rom(args.rom_path):
            logger.error("Failed to load ROM")
            return 1
        
        # Determine output directory
        output_dir = args.output_dir
        if not output_dir:
            base_name, _ = os.path.splitext(args.rom_path)
            output_dir = f"{base_name}_extracted"
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract data based on type
        extracted_files = []
        
        if args.extract_type == "header" or args.extract_type == "all":
            # Extract header
            header_path = os.path.join(output_dir, "header.json")
            with open(header_path, 'w') as f:
                json.dump(extractor.header_data, f, indent=2)
            extracted_files.append(header_path)
        
        if args.extract_type == "prg" or args.extract_type == "all":
            # Extract PRG ROM
            if hasattr(extractor, 'prg_rom') and extractor.prg_rom:
                prg_path = os.path.join(output_dir, "prg.bin")
                with open(prg_path, 'wb') as f:
                    f.write(extractor.prg_rom)
                extracted_files.append(prg_path)
        
        if args.extract_type == "chr" or args.extract_type == "all":
            # Extract CHR ROM
            if hasattr(extractor, 'chr_rom') and extractor.chr_rom:
                chr_path = os.path.join(output_dir, "chr.bin")
                with open(chr_path, 'wb') as f:
                    f.write(extractor.chr_rom)
                extracted_files.append(chr_path)
        
        if args.extract_type == "audio" or args.extract_type == "all":
            # Audio extraction not implemented
            logger.warning("Audio extraction not implemented")
        
        if extracted_files:
            logger.info(f"Extracted {len(extracted_files)} files to {output_dir}:")
            for file_path in extracted_files:
                logger.info(f"  {os.path.basename(file_path)}")
        else:
            logger.warning("No files were extracted")
        
        return 0
        
    except ImportError as e:
        logger.error(f"Required module not found: {e}")
        return 1
    except Exception as e:
        logger.error(f"Error during extraction: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1

def handle_visualize_command(args) -> int:
    """
    Handle 'visualize' command.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Exit code (0 for success, non-zero for error)
    """
    # Ensure required modules
    if not ensure_module("quantum_signal_emulator") or not ensure_module("matplotlib"):
        return 1
    
    try:
        from quantum_signal_emulator.common.visualizer import SignalVisualizer
        
        logger.info(f"Creating visualization from: {args.input_path}")
        
        # Load input data
        input_format = args.input_format
        
        if input_format == "auto":
            # Determine format from extension
            _, ext = os.path.splitext(args.input_path)
            ext = ext.lower()
            
            if ext in ['.json']:
                input_format = "json"
            elif ext in ['.csv']:
                input_format = "csv"
            elif ext in ['.npy', '.npz']:
                input_format = "numpy"
            elif ext in ['.log', '.txt']:
                input_format = "log"
            else:
                logger.error(f"Could not determine format for {args.input_path}")
                return 1
        
        # Load data
        data = None
        if input_format == "json":
            with open(args.input_path, 'r') as f:
                data = json.load(f)
        elif input_format == "csv":
            # Ensure required module
            if not ensure_module("pandas"):
                return 1
                
            import pandas as pd
            data = pd.read_csv(args.input_path)
        elif input_format == "numpy":
            # Ensure required module
            if not ensure_module("numpy"):
                return 1
                
            import numpy as np
            data = np.load(args.input_path, allow_pickle=True)
        elif input_format == "log":
            with open(args.input_path, 'r') as f:
                data = f.readlines()
        
        # Create visualizer
        visualizer = SignalVisualizer(use_3d=not args.no_3d, dark_mode=args.dark_mode)
        
        # Create visualization based on type
        import matplotlib.pyplot as plt
        
        if args.type == "timing" or args.type == "all":
            # Create timing visualization
            if isinstance(data, dict) and "state_history" in data:
                # Use built-in visualizer function
                mock_system = type('MockSystem', (), {'state_history': data["state_history"]})
                visualizer.visualize_cycle_timing(mock_system)
            else:
                # Create custom timing plot
                plt.figure(figsize=(12, 8))
                
                if isinstance(data, dict) and "cycle_patterns" in data:
                    # Plot from cycle analyzer output
                    if "statistics" in data:
                        stats = data["statistics"]
                        plt.subplot(2, 1, 1)
                        plt.bar(["Min", "Max", "Avg"], 
                              [stats.get("min_cycle_delta", 0), 
                               stats.get("max_cycle_delta", 0), 
                               stats.get("avg_cycle_delta", 0)])
                        plt.title("Cycle Delta Statistics")
                        plt.ylabel("Cycles")
                        
                        plt.subplot(2, 1, 2)
                        if "cycle_patterns" in data:
                            patterns = data["cycle_patterns"]
                            pattern_labels = [f"Pattern {i+1}" for i in range(len(patterns))]
                            occurrences = [p.get("occurrences", 0) for p in patterns]
                            plt.bar(pattern_labels, occurrences)
                            plt.title("Cycle Pattern Occurrences")
                            plt.ylabel("Count")
                    
                elif isinstance(data, pd.DataFrame) and "cycle" in data.columns:
                    # Plot from DataFrame
                    if "scanline" in data.columns:
                        plt.plot(data["cycle"], data["scanline"])
                        plt.title("Scanline vs Cycle")
                        plt.xlabel("Cycle")
                        plt.ylabel("Scanline")
                
                if args.save_image:
                    plt.savefig(args.save_image, dpi=300)
                    logger.info(f"Timing visualization saved to {args.save_image}")
                else:
                    plt.tight_layout()
                    plt.show()
        
        if args.type == "registers" or args.type == "all":
            # Create register visualization
            if isinstance(data, dict) and "state_history" in data:
                # Use built-in visualizer function
                
                # Filter registers if specified
                register_names = args.register
                
                mock_system = type('MockSystem', (), {'state_history': data["state_history"]})
                visualizer.plot_register_states(mock_system, register_names=register_names)
            else:
                # Create custom register plot
                plt.figure(figsize=(12, 8))
                plt.title("Register States")
                
                if isinstance(data, pd.DataFrame):
                    # Try to find register columns
                    register_cols = [col for col in data.columns if col.startswith("reg_") or col in args.register]
                    
                    if register_cols:
                        for col in register_cols:
                            plt.plot(data.index, data[col], label=col)
                        plt.legend()
                        plt.xlabel("Sample")
                        plt.ylabel("Value")
                    else:
                        plt.text(0.5, 0.5, "No register data found", 
                               ha='center', va='center', transform=plt.gca().transAxes)
                else:
                    plt.text(0.5, 0.5, "Unsupported data format for register visualization", 
                           ha='center', va='center', transform=plt.gca().transAxes)
                
                if args.save_image:
                    plt.savefig(args.save_image, dpi=300)
                    logger.info(f"Register visualization saved to {args.save_image}")
                else:
                    plt.tight_layout()
                    plt.show()
        
        if args.type == "signals" or args.type == "all":
            # Create signal visualization
            plt.figure(figsize=(12, 8))
            plt.title("Signal Data")
            
            if isinstance(data, dict) and "signals" in data:
                # Plot from signals dictionary
                signals = data["signals"]
                
                if isinstance(signals, dict):
                    # Plot each signal
                    for i, (name, values) in enumerate(signals.items()):
                        if hasattr(values, '__len__') and len(values) > 1:
                            plt.subplot(len(signals), 1, i+1)
                            plt.plot(values)
                            plt.ylabel(name)
                            
                            if i == len(signals) - 1:
                                plt.xlabel("Sample")
            elif isinstance(data, np.ndarray):
                # Plot from NumPy array
                if len(data.shape) == 1:
                    # 1D array
                    plt.plot(data)
                    plt.xlabel("Sample")
                    plt.ylabel("Value")
                elif len(data.shape) == 2:
                    # 2D array - plot each row as a signal
                    for i in range(min(data.shape[0], 5)):  # Limit to 5 signals
                        plt.subplot(min(data.shape[0], 5), 1, i+1)
                        plt.plot(data[i])
                        plt.ylabel(f"Signal {i}")
                        
                        if i == min(data.shape[0], 5) - 1:
                            plt.xlabel("Sample")
            elif isinstance(data, pd.DataFrame):
                # Plot from DataFrame
                numerical_cols = data.select_dtypes(include=['number']).columns
                
                for i, col in enumerate(numerical_cols[:5]):  # Limit to 5 signals
                    plt.subplot(len(numerical_cols[:5]), 1, i+1)
                    plt.plot(data[col])
                    plt.ylabel(col)
                    
                    if i == len(numerical_cols[:5]) - 1:
                        plt.xlabel("Sample")
            
            if args.save_image:
                plt.savefig(args.save_image, dpi=300)
                logger.info(f"Signal visualization saved to {args.save_image}")
            else:
                plt.tight_layout()
                plt.show()
        
        if args.type == "state-space" or args.type == "all":
            # Create state space visualization
            if isinstance(data, dict) and "state_history" in data:
                # Use built-in visualizer function
                mock_system = type('MockSystem', (), {'state_history': data["state_history"]})
                visualizer.visualize_state_space(mock_system)
            else:
                # State space visualization requires high-dimensional data
                logger.warning("State space visualization requires state history data")
                
                # Create basic plot for other data types
                if isinstance(data, np.ndarray) and len(data.shape) >= 2:
                    # Use dimensionality reduction
                    if not ensure_module("sklearn"):
                        return 1
                        
                    from sklearn.decomposition import PCA
                    
                    # Apply PCA
                    if data.shape[1] >= 3:
                        pca = PCA(n_components=3)
                        reduced_data = pca.fit_transform(data)
                        
                        # Create 3D plot
                        if not args.no_3d:
                            from mpl_toolkits.mplot3d import Axes3D
                            fig = plt.figure(figsize=(10, 8))
                            ax = fig.add_subplot(111, projection='3d')
                            
                            ax.scatter(reduced_data[:, 0], reduced_data[:, 1], reduced_data[:, 2], 
                                     c=np.arange(len(reduced_data)), cmap='viridis')
                            ax.set_title("PCA State Space")
                            ax.set_xlabel("Component 1")
                            ax.set_ylabel("Component 2")
                            ax.set_zlabel("Component 3")
                        else:
                            # 2D Plot
                            plt.figure(figsize=(10, 8))
                            plt.scatter(reduced_data[:, 0], reduced_data[:, 1], 
                                      c=np.arange(len(reduced_data)), cmap='viridis')
                            plt.title("PCA State Space")
                            plt.xlabel("Component 1")
                            plt.ylabel("Component 2")
                    else:
                        logger.warning("Data dimensionality too low for meaningful state space visualization")
                
                if args.save_image:
                    plt.savefig(args.save_image, dpi=300)
                    logger.info(f"State space visualization saved to {args.save_image}")
                else:
                    plt.tight_layout()
                    plt.show()
        
        if args.type == "quantum" or args.type == "all":
            # Create quantum visualization
            if isinstance(data, dict) and ("quantum_data" in data or "frequency_data" in data):
                # Use built-in visualizer function
                if "quantum_entropy" in data or "frequency_data" in data:
                    visualizer.plot_quantum_results(data)
                else:
                    logger.warning("Quantum results data not found in input")
            else:
                logger.warning("Quantum visualization requires quantum analysis results")
        
        return 0
        
    except ImportError as e:
        logger.error(f"Required module not found: {e}")
        return 1
    except Exception as e:
        logger.error(f"Error during visualization: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1

def handle_validate_command(args) -> int:
    """
    Handle 'validate' command.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Exit code (0 for success, non-zero for error)
    """
    # Ensure required modules
    if not ensure_module("quantum_signal_emulator"):
        return 1
    
    try:
        logger.info(f"Validating file: {args.input_path}")
        
        # Determine file type if auto
        file_type = args.type
        if file_type == "auto":
            # Guess from file extension
            _, ext = os.path.splitext(args.input_path)
            ext = ext.lower()
            
            if ext in ['.json', '.yaml', '.yml']:
                # Try to load and check fields
                try:
                    if ext == '.json':
                        with open(args.input_path, 'r') as f:
                            data = json.load(f)
                    else:
                        with open(args.input_path, 'r') as f:
                            data = yaml.safe_load(f)
                    
                    if isinstance(data, dict):
                        if 'state_history' in data:
                            file_type = "state"
                        elif 'signals' in data:
                            file_type = "signals"
                        elif 'system' in data:
                            file_type = "config"
                        else:
                            file_type = "unknown"
                    else:
                        file_type = "unknown"
                except:
                    file_type = "unknown"
            else:
                file_type = "unknown"
                
        # Validate based on file type
        if file_type == "config":
            from quantum_signal_emulator.utils.config_manager import ConfigManager
            
            # Create config manager
            config_manager = ConfigManager()
            
            # Load and validate config
            try:
                if args.input_path.endswith('.json'):
                    with open(args.input_path, 'r') as f:
                        config_data = json.load(f)
                else:
                    with open(args.input_path, 'r') as f:
                        config_data = yaml.safe_load(f)
                
                # Validate
                validation_errors = config_manager.validate_config(config_data)
                
                if validation_errors:
                    logger.error("Configuration validation failed:")
                    for error in validation_errors:
                        logger.error(f"  {error}")
                    return 1
                else:
                    logger.info("Configuration validation successful")
                    return 0
            except Exception as e:
                logger.error(f"Error loading configuration: {e}")
                return 1
                
        elif file_type == "state":
            # Validate state history
            try:
                if args.input_path.endswith('.json'):
                    with open(args.input_path, 'r') as f:
                        data = json.load(f)
                else:
                    with open(args.input_path, 'r') as f:
                        data = yaml.safe_load(f)
                
                if not isinstance(data, dict) or "state_history" not in data:
                    logger.error("Invalid state history file: 'state_history' field missing")
                    return 1
                
                state_history = data["state_history"]
                
                if not isinstance(state_history, list):
                    logger.error("Invalid state history: not a list")
                    return 1
                
                if not state_history:
                    logger.warning("State history is empty")
                    return 0
                
                # Check required fields in state entries
                invalid_entries = 0
                for i, state in enumerate(state_history):
                    if not isinstance(state, dict):
                        logger.error(f"Invalid state entry {i}: not a dictionary")
                        invalid_entries += 1
                        continue
                    
                    if args.strict:
                        # Check for required fields
                        required_fields = ["cycle"]
                        missing_fields = [field for field in required_fields if field not in state]
                        
                        if missing_fields:
                            logger.error(f"Invalid state entry {i}: missing required fields {missing_fields}")
                            invalid_entries += 1
                
                if invalid_entries > 0:
                    logger.error(f"Found {invalid_entries} invalid state entries")
                    return 1
                
                logger.info("State history validation successful")
                logger.info(f"State history contains {len(state_history)} entries")
                return 0
                
            except Exception as e:
                logger.error(f"Error validating state history: {e}")
                return 1
                
        elif file_type == "signals":
            # Validate signal data
            try:
                if args.input_path.endswith('.json'):
                    with open(args.input_path, 'r') as f:
                        data = json.load(f)
                else:
                    with open(args.input_path, 'r') as f:
                        data = yaml.safe_load(f)
                
                if not isinstance(data, dict) or "signals" not in data:
                    logger.error("Invalid signals file: 'signals' field missing")
                    return 1
                
                signals = data["signals"]
                
                if not isinstance(signals, dict):
                    logger.error("Invalid signals data: not a dictionary")
                    return 1
                
                if not signals:
                    logger.warning("Signals dictionary is empty")
                    return 0
                
                # Check signal entries
                invalid_signals = 0
                for name, values in signals.items():
                    if not hasattr(values, '__len__'):
                        logger.error(f"Invalid signal '{name}': not a sequence")
                        invalid_signals += 1
                
                if invalid_signals > 0:
                    logger.error(f"Found {invalid_signals} invalid signals")
                    return 1
                
                logger.info("Signals validation successful")
                logger.info(f"Found {len(signals)} signals")
                return 0
                
            except Exception as e:
                logger.error(f"Error validating signals: {e}")
                return 1
                
        else:
            logger.error(f"Unsupported file type for validation: {file_type}")
            return 1
        
    except ImportError as e:
        logger.error(f"Required module not found: {e}")
        return 1
    except Exception as e:
        logger.error(f"Error during validation: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1

def handle_import_command(args) -> int:
    """
    Handle 'import' command.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Exit code (0 for success, non-zero for error)
    """
    # Ensure required modules
    if not ensure_module("quantum_signal_emulator"):
        return 1
    
    try:
        from quantum_signal_emulator.utils.data_importer import DataImporter
        
        logger.info(f"Importing data from: {args.input_path}")
        
        # Create data importer
        importer = DataImporter(debug_mode=args.debug)
        
        # Determine source format if auto
        source_format = args.source
        if source_format == "auto":
            # Guess from file extension
            _, ext = os.path.splitext(args.input_path)
            ext = ext.lower()
            
            if ext in ['.fm2', '.fc0', '.fs']:
                source_format = "fceux"
            elif ext in ['.mst', '.mss']:
                source_format = "mesen"
            elif ext in ['.state']:
                # Check file size - BizHawk typically has larger savestates
                file_size = os.path.getsize(args.input_path)
                if file_size > 100000:  # 100 KB
                    source_format = "bizhawk"
                else:
                    source_format = "stella"  # For Atari
            elif ext in ['.sta']:
                source_format = "mame"
            elif ext in ['.gs0', '.gs1', '.gs2']:
                source_format = "genesis_plus_gx"
            elif ext in ['.vcd', '.csv']:
                source_format = "logic_analyzer"
            else:
                # Try to guess from content for log files
                if ext in ['.log', '.txt']:
                    with open(args.input_path, 'r') as f:
                        content = f.read(1000)  # Read first 1000 bytes
                        
                        if "FCEUX" in content:
                            source_format = "fceux"
                        elif "Mesen" in content:
                            source_format = "mesen"
                        elif "BizHawk" in content:
                            source_format = "bizhawk"
                        elif "MAME" in content:
                            source_format = "mame"
                        elif "Genesis Plus" in content:
                            source_format = "genesis_plus_gx"
                        elif "Stella" in content or "TIA" in content:
                            source_format = "stella"
                        else:
                            # Default to logic analyzer for data-like files
                            source_format = "logic_analyzer"
                else:
                    logger.error(f"Could not determine source format for {args.input_path}")
                    return 1
        
        # Import data
        import_result = importer.import_data(args.input_path, source_format, args.system)
        
        if "error" in import_result:
            logger.error(f"Import failed: {import_result['error']}")
            return 1
        
        # Convert to standard format if requested
        if args.convert:
            conversion_result = importer.convert_to_state_history(import_result)
            
            if conversion_result:
                import_result["state_history"] = conversion_result
                import_result["converted"] = True
                logger.info(f"Converted to standard state history format with {len(conversion_result)} entries")
        
        # Save result
        output_path = args.output
        if not output_path:
            base_name, _ = os.path.splitext(args.input_path)
            output_path = f"{base_name}_imported.json"
        
        with open(output_path, 'w') as f:
            if args.format == "json":
                json.dump(import_result, f, indent=2)
            elif args.format == "yaml":
                yaml.dump(import_result, f, default_flow_style=False)
            else:
                # Text format
                f.write(f"Import Results for {args.input_path}\n")
                f.write("="*50 + "\n\n")
                
                f.write(f"Source format: {source_format}\n")
                f.write(f"System: {import_result.get('system', 'unknown')}\n")
                
                if "metadata" in import_result:
                    f.write("\nMetadata:\n")
                    for key, value in import_result["metadata"].items():
                        f.write(f"  {key}: {value}\n")
                
                if "state_history" in import_result:
                    state_history = import_result["state_history"]
                    f.write(f"\nState History: {len(state_history)} entries\n")
                    
                    if state_history:
                        f.write("  First state:\n")
                        for key, value in state_history[0].items():
                            if isinstance(value, dict):
                                f.write(f"    {key}: {{{len(value)} items}}\n")
                            else:
                                f.write(f"    {key}: {value}\n")
        
        logger.info(f"Import complete. Output saved to {output_path}")
        return 0
        
    except ImportError as e:
        logger.error(f"Required module not found: {e}")
        return 1
    except Exception as e:
        logger.error(f"Error during import: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1

def handle_help_command(args) -> int:
    """
    Handle 'help' command.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Exit code (0 for success, non-zero for error)
    """
    if args.command is None:
        # Show general help
        parser = get_base_parser()
        parser.print_help()
        return 0
    
    # Show help for specific command
    parser = get_base_parser()
    subparsers = parser.add_subparsers(dest="command")
    
    setup_analyze_parser(subparsers)
    setup_convert_parser(subparsers)
    setup_extract_parser(subparsers)
    setup_visualize_parser(subparsers)
    setup_validate_parser(subparsers)
    setup_import_parser(subparsers)
    setup_help_parser(subparsers)
    
    # Get the parser for the requested command
    command_parser = None
    for action in subparsers._actions:
        if isinstance(action, argparse._SubParsersAction):
            if args.command in action.choices:
                command_parser = action.choices[args.command]
                break
    
    if command_parser:
        command_parser.print_help()
    else:
        logger.error(f"Unknown command: {args.command}")
        parser.print_help()
        return 1
    
    return 0

def main() -> int:
    """
    Main entry point for CLI.
    
    Returns:
        Exit code (0 for success, non-zero for error)
    """
    # Create parser and subparsers
    parser = get_base_parser()
    subparsers = parser.add_subparsers(dest="command")
    
    # Add command parsers
    setup_analyze_parser(subparsers)
    setup_convert_parser(subparsers)
    setup_extract_parser(subparsers)
    setup_visualize_parser(subparsers)
    setup_validate_parser(subparsers)
    setup_import_parser(subparsers)
    setup_help_parser(subparsers)
    
    # Parse arguments
    args = parser.parse_args()
    
    # Configure logging
    if args.quiet:
        logging.root.setLevel(logging.ERROR)
    elif args.verbose:
        logging.root.setLevel(logging.DEBUG)
    
    # Handle command
    if args.command == "analyze":
        return handle_analyze_command(args)
    elif args.command == "convert":
        return handle_convert_command(args)
    elif args.command == "extract":
        return handle_extract_command(args)
    elif args.command == "visualize":
        return handle_visualize_command(args)
    elif args.command == "validate":
        return handle_validate_command(args)
    elif args.command == "import":
        return handle_import_command(args)
    elif args.command == "help":
        return handle_help_command(args)
    else:
        # No command specified, show help
        parser.print_help()
        return 0

if __name__ == "__main__":
    sys.exit(main())