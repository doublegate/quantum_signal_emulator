"""
Data import utilities for external emulator data and hardware captures.

This module provides utilities for importing data from external emulators,
hardware captures, and other sources for comparison and analysis in the
Quantum Signal Emulator.
"""

import os
import json
import csv
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union, BinaryIO
import re
import gzip
import pickle
from collections import defaultdict

logger = logging.getLogger("QuantumSignalEmulator.DataImporter")

class DataImporter:
    """
    Import external emulator data and hardware captures.
    
    This class provides methods for importing data from various external
    sources, such as other emulators' logs, hardware captures, and
    timing analysis tools, for use in the Quantum Signal Emulator.
    """
    
    def __init__(self, debug_mode: bool = False):
        """
        Initialize the data importer.
        
        Args:
            debug_mode: Enable additional debug information
        """
        self.debug_mode = debug_mode
        self.importers = {
            'fceux': self._import_fceux,
            'mesen': self._import_mesen,
            'bizhawk': self._import_bizhawk,
            'mame': self._import_mame,
            'genesis_plus_gx': self._import_genesis_plus_gx,
            'stella': self._import_stella,
            'logic_analyzer': self._import_logic_analyzer,
            'quantum_signals': self._import_quantum_signals
        }
        
        logger.info("DataImporter initialized")
    
    def import_data(self, filepath: str, format_type: str, 
                  target_system: Optional[str] = None) -> Dict[str, Any]:
        """
        Import data from external file.
        
        Args:
            filepath: Path to data file
            format_type: Data format type
            target_system: Target system type (optional)
            
        Returns:
            Dictionary with imported data
        """
        if not os.path.exists(filepath):
            logger.error(f"Import file not found: {filepath}")
            return {"error": f"File not found: {filepath}"}
        
        # Check if format is supported
        if format_type not in self.importers:
            supported = ", ".join(self.importers.keys())
            logger.error(f"Unsupported import format: {format_type}. Supported formats: {supported}")
            return {"error": f"Unsupported format: {format_type}"}
        
        # Call appropriate importer
        try:
            logger.info(f"Importing {format_type} data from {filepath}")
            importer = self.importers[format_type]
            import_result = importer(filepath, target_system)
            
            if "error" in import_result:
                logger.error(f"Import error: {import_result['error']}")
            else:
                logger.info(f"Successfully imported {format_type} data")
                
            return import_result
        except Exception as e:
            logger.error(f"Error importing data: {e}")
            import traceback
            traceback.print_exc()
            return {"error": f"Import error: {str(e)}"}
    
    def _import_fceux(self, filepath: str, 
                    target_system: Optional[str] = None) -> Dict[str, Any]:
        """
        Import FCEUX trace log or savestate.
        
        Args:
            filepath: Path to FCEUX data file
            target_system: Target system type
            
        Returns:
            Dictionary with imported data
        """
        # Default to NES for FCEUX
        if target_system is None:
            target_system = "nes"
        elif target_system != "nes":
            return {"error": f"FCEUX importer only supports NES, not {target_system}"}
        
        _, ext = os.path.splitext(filepath)
        
        if ext.lower() == '.fm2':
            # FM2 movie file
            return self._import_fceux_movie(filepath)
        elif ext.lower() == '.log' or ext.lower() == '.txt':
            # Trace log
            return self._import_fceux_trace(filepath)
        elif ext.lower() == '.fs' or ext.lower() == '.fc0':
            # Savestate
            return self._import_fceux_savestate(filepath)
        else:
            return {"error": f"Unsupported FCEUX file format: {ext}"}
    
    def _import_fceux_trace(self, filepath: str) -> Dict[str, Any]:
        """
        Import FCEUX trace log.
        
        Args:
            filepath: Path to trace log
            
        Returns:
            Dictionary with imported trace data
        """
        try:
            with open(filepath, 'r') as f:
                lines = f.readlines()
                
            # Parse trace log
            # Format is typically: <PC>: <opcode> <args>   A:#<A> X:#<X> Y:#<Y> P:#<P> SP:#<SP>
            trace_data = []
            cycle = 0
            
            for line in lines:
                line = line.strip()
                if not line or line.startswith(';'):
                    continue
                    
                # Extract CPU state
                try:
                    # Parse PC and instruction
                    pc_part = line.split(':', 1)[0].strip()
                    pc = int(pc_part, 16)
                    
                    # Parse registers
                    reg_parts = re.findall(r'([A-Z]+):#(\$[0-9A-F]+|\d+)', line)
                    registers = {}
                    
                    for reg, value in reg_parts:
                        if value.startswith('$'):
                            value = int(value[1:], 16)
                        else:
                            value = int(value)
                        registers[reg] = value
                    
                    # Create state entry
                    state = {
                        "cycle": cycle,
                        "pc": pc,
                        "registers": registers
                    }
                    
                    trace_data.append(state)
                    
                    # Increment cycle (approximate)
                    cycle += 1
                except Exception as e:
                    logger.warning(f"Error parsing trace line: {line}, {e}")
                    continue
            
            return {
                "format": "fceux_trace",
                "system": "nes",
                "state_history": trace_data,
                "metadata": {
                    "total_cycles": cycle,
                    "total_instructions": len(trace_data)
                }
            }
        except Exception as e:
            return {"error": f"Failed to parse FCEUX trace: {e}"}
    
    def _import_fceux_movie(self, filepath: str) -> Dict[str, Any]:
        """
        Import FCEUX FM2 movie file.
        
        Args:
            filepath: Path to FM2 file
            
        Returns:
            Dictionary with imported movie data
        """
        try:
            with open(filepath, 'r') as f:
                lines = f.readlines()
                
            # Parse FM2 movie file
            metadata = {}
            input_log = []
            
            for line in lines:
                line = line.strip()
                
                if line.startswith('|'):
                    # Input data
                    input_log.append(line)
                elif line and not line.startswith(';'):
                    # Metadata
                    if '=' in line:
                        key, value = line.split('=', 1)
                        metadata[key.strip()] = value.strip()
            
            # Parse input data
            # Format: |<frame>|<controller1>|<controller2>|...
            frames = []
            
            for input_line in input_log:
                parts = input_line.split('|')
                if len(parts) >= 3:
                    try:
                        frame_num = int(parts[1])
                        controller1 = parts[2]
                        controller2 = parts[3] if len(parts) > 3 else ""
                        
                        frames.append({
                            "frame": frame_num,
                            "input1": controller1,
                            "input2": controller2
                        })
                    except ValueError:
                        continue
            
            return {
                "format": "fceux_movie",
                "system": "nes",
                "metadata": metadata,
                "frames": frames
            }
        except Exception as e:
            return {"error": f"Failed to parse FCEUX movie: {e}"}
    
    def _import_fceux_savestate(self, filepath: str) -> Dict[str, Any]:
        """
        Import FCEUX savestate.
        
        Args:
            filepath: Path to savestate file
            
        Returns:
            Dictionary with imported savestate data
        """
        # Savestates are binary files with complex format
        # This is a simplified implementation
        try:
            with open(filepath, 'rb') as f:
                data = f.read()
            
            # Look for CPU and PPU state markers
            cpu_state = {}
            ppu_state = {}
            
            # Example: try to find register values in savestate
            # (this is a crude approximation - real implementation would need detailed format knowledge)
            
            # Provide basic information
            return {
                "format": "fceux_savestate",
                "system": "nes",
                "binary_size": len(data),
                "cpu_state": cpu_state,
                "ppu_state": ppu_state,
                "note": "Full savestate parsing is not implemented"
            }
        except Exception as e:
            return {"error": f"Failed to parse FCEUX savestate: {e}"}
    
    def _import_mesen(self, filepath: str, 
                    target_system: Optional[str] = None) -> Dict[str, Any]:
        """
        Import Mesen trace log or savestate.
        
        Args:
            filepath: Path to Mesen data file
            target_system: Target system type
            
        Returns:
            Dictionary with imported data
        """
        # Mesen supports NES and SNES
        if target_system is None:
            # Try to determine from file
            if filepath.lower().endswith('.mss'):
                target_system = "snes"
            else:
                target_system = "nes"
        
        _, ext = os.path.splitext(filepath)
        
        if ext.lower() == '.txt' or ext.lower() == '.log':
            # Trace log
            return self._import_mesen_trace(filepath, target_system)
        elif ext.lower() == '.mst' or ext.lower() == '.mss':
            # Savestate
            return self._import_mesen_savestate(filepath, target_system)
        else:
            return {"error": f"Unsupported Mesen file format: {ext}"}
    
    def _import_mesen_trace(self, filepath: str, 
                          target_system: str) -> Dict[str, Any]:
        """
        Import Mesen trace log.
        
        Args:
            filepath: Path to trace log
            target_system: Target system type
            
        Returns:
            Dictionary with imported trace data
        """
        try:
            with open(filepath, 'r') as f:
                lines = f.readlines()
                
            # Parse Mesen trace format
            # Format: <cycle> <scanline> <PC>: <opcode> <args> A:<A> X:<X> Y:<Y> P:<P> SP:<SP>
            trace_data = []
            
            for line in lines:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                    
                try:
                    # Mesen includes cycle and scanline info
                    parts = line.split(' ', 2)
                    if len(parts) >= 3:
                        cycle = int(parts[0])
                        scanline = int(parts[1])
                        
                        # Extract PC and registers
                        cpu_parts = parts[2].split(':', 1)
                        if len(cpu_parts) >= 2:
                            pc = int(cpu_parts[0], 16)
                            
                            # Parse registers
                            reg_parts = re.findall(r'([A-Z]+):([0-9A-F]+)', cpu_parts[1])
                            registers = {}
                            
                            for reg, value in reg_parts:
                                registers[reg] = int(value, 16)
                            
                            # Create state entry
                            state = {
                                "cycle": cycle,
                                "scanline": scanline,
                                "pc": pc,
                                "registers": registers
                            }
                            
                            trace_data.append(state)
                except Exception as e:
                    logger.warning(f"Error parsing Mesen trace line: {line}, {e}")
                    continue
            
            return {
                "format": f"mesen_trace_{target_system}",
                "system": target_system,
                "state_history": trace_data,
                "metadata": {
                    "total_instructions": len(trace_data)
                }
            }
        except Exception as e:
            return {"error": f"Failed to parse Mesen trace: {e}"}
    
    def _import_mesen_savestate(self, filepath: str, 
                              target_system: str) -> Dict[str, Any]:
        """
        Import Mesen savestate.
        
        Args:
            filepath: Path to savestate file
            target_system: Target system type
            
        Returns:
            Dictionary with imported savestate data
        """
        # Savestates are binary files with complex format
        # This is a simplified implementation
        try:
            with open(filepath, 'rb') as f:
                data = f.read()
            
            # Provide basic information
            return {
                "format": f"mesen_savestate_{target_system}",
                "system": target_system,
                "binary_size": len(data),
                "note": "Full savestate parsing is not implemented"
            }
        except Exception as e:
            return {"error": f"Failed to parse Mesen savestate: {e}"}
    
    def _import_bizhawk(self, filepath: str, 
                      target_system: Optional[str] = None) -> Dict[str, Any]:
        """
        Import BizHawk trace log or savestate.
        
        Args:
            filepath: Path to BizHawk data file
            target_system: Target system type
            
        Returns:
            Dictionary with imported data
        """
        # BizHawk supports multiple systems
        _, ext = os.path.splitext(filepath)
        
        if ext.lower() == '.txt' or ext.lower() == '.log':
            # Trace log
            return self._import_bizhawk_trace(filepath, target_system)
        elif ext.lower() == '.state':
            # Savestate
            return self._import_bizhawk_savestate(filepath, target_system)
        else:
            return {"error": f"Unsupported BizHawk file format: {ext}"}
    
    def _import_bizhawk_trace(self, filepath: str, 
                            target_system: Optional[str] = None) -> Dict[str, Any]:
        """
        Import BizHawk trace log.
        
        Args:
            filepath: Path to trace log
            target_system: Target system type
            
        Returns:
            Dictionary with imported trace data
        """
        try:
            with open(filepath, 'r') as f:
                lines = f.readlines()
                
            # Try to determine system type from trace format
            if target_system is None:
                # Check first few lines for system indicators
                for line in lines[:20]:
                    if 'NES' in line:
                        target_system = 'nes'
                        break
                    elif 'SNES' in line:
                        target_system = 'snes'
                        break
                    elif 'GEN' in line or 'SMS' in line:
                        target_system = 'genesis'
                        break
                    elif 'A26' in line:
                        target_system = 'atari2600'
                        break
                
                if target_system is None:
                    target_system = 'unknown'
            
            # Parse BizHawk trace format (varies by system)
            trace_data = []
            cycle = 0
            
            for line in lines:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                    
                try:
                    # BizHawk format varies but often includes register states
                    if target_system == 'nes':
                        # Example format for NES
                        if ':' in line:
                            # Extract PC
                            pc_part = line.split(':', 1)[0].strip()
                            try:
                                pc = int(pc_part, 16)
                            except ValueError:
                                continue
                            
                            # Extract registers
                            reg_parts = re.findall(r'([A-Z]+):([0-9A-F]+)', line)
                            registers = {}
                            
                            for reg, value in reg_parts:
                                registers[reg] = int(value, 16)
                            
                            # Create state entry
                            state = {
                                "cycle": cycle,
                                "pc": pc,
                                "registers": registers
                            }
                            
                            trace_data.append(state)
                            cycle += 1
                    else:
                        # Generic trace format
                        # Just count lines and create basic entries
                        trace_data.append({"cycle": cycle, "data": line})
                        cycle += 1
                        
                except Exception as e:
                    logger.warning(f"Error parsing BizHawk trace line: {line}, {e}")
                    continue
            
            return {
                "format": f"bizhawk_trace_{target_system}",
                "system": target_system,
                "state_history": trace_data,
                "metadata": {
                    "total_instructions": len(trace_data)
                }
            }
        except Exception as e:
            return {"error": f"Failed to parse BizHawk trace: {e}"}
    
    def _import_bizhawk_savestate(self, filepath: str, 
                                target_system: Optional[str] = None) -> Dict[str, Any]:
        """
        Import BizHawk savestate.
        
        Args:
            filepath: Path to savestate file
            target_system: Target system type
            
        Returns:
            Dictionary with imported savestate data
        """
        # BizHawk savestates are often in a custom format
        try:
            with open(filepath, 'rb') as f:
                data = f.read()
            
            # Try to determine system type from savestate
            if target_system is None:
                # Look for system identifier in binary data
                if b'NESCore' in data:
                    target_system = 'nes'
                elif b'SNESCore' in data:
                    target_system = 'snes'
                elif b'GenCore' in data:
                    target_system = 'genesis'
                elif b'A26Core' in data:
                    target_system = 'atari2600'
                else:
                    target_system = 'unknown'
            
            # Provide basic information
            return {
                "format": f"bizhawk_savestate_{target_system}",
                "system": target_system,
                "binary_size": len(data),
                "note": "Full savestate parsing is not implemented"
            }
        except Exception as e:
            return {"error": f"Failed to parse BizHawk savestate: {e}"}
    
    def _import_mame(self, filepath: str, 
                   target_system: Optional[str] = None) -> Dict[str, Any]:
        """
        Import MAME trace or savestate.
        
        Args:
            filepath: Path to MAME data file
            target_system: Target system type
            
        Returns:
            Dictionary with imported data
        """
        # MAME supports multiple systems
        _, ext = os.path.splitext(filepath)
        
        if ext.lower() == '.log':
            # Debug log
            return self._import_mame_log(filepath, target_system)
        elif ext.lower() == '.inp':
            # Input recording
            return self._import_mame_recording(filepath, target_system)
        elif ext.lower() == '.sta':
            # Savestate
            return self._import_mame_savestate(filepath, target_system)
        else:
            return {"error": f"Unsupported MAME file format: {ext}"}
    
    def _import_mame_log(self, filepath: str, 
                       target_system: Optional[str] = None) -> Dict[str, Any]:
        """
        Import MAME debug log.
        
        Args:
            filepath: Path to debug log
            target_system: Target system type
            
        Returns:
            Dictionary with imported log data
        """
        try:
            with open(filepath, 'r') as f:
                lines = f.readlines()
                
            # Try to determine system from log
            if target_system is None:
                # Look for system identifier in log
                for line in lines[:50]:
                    if re.search(r'(Driver|Machine):\s+([a-z0-9_]+)', line):
                        machine_match = re.search(r'(Driver|Machine):\s+([a-z0-9_]+)', line)
                        if machine_match:
                            machine = machine_match.group(2)
                            
                            # Map MAME machine name to our system type
                            if machine.startswith('nes'):
                                target_system = 'nes'
                            elif machine.startswith('snes'):
                                target_system = 'snes'
                            elif machine.startswith('genesis') or machine.startswith('megadriv'):
                                target_system = 'genesis'
                            elif machine.startswith('a2600'):
                                target_system = 'atari2600'
                            else:
                                target_system = machine
                            break
                
                if target_system is None:
                    target_system = 'unknown'
            
            # Parse MAME debug log
            # Format varies significantly by system and debug settings
            log_entries = []
            current_cycle = 0
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                # Extract cycle information if present
                cycle_match = re.search(r'cycle\s+(\d+)', line, re.IGNORECASE)
                if cycle_match:
                    current_cycle = int(cycle_match.group(1))
                
                # Add entry
                log_entries.append({
                    "cycle": current_cycle,
                    "text": line
                })
                
                # Increment cycle if no explicit info
                if not cycle_match:
                    current_cycle += 1
            
            return {
                "format": f"mame_log_{target_system}",
                "system": target_system,
                "entries": log_entries,
                "metadata": {
                    "total_entries": len(log_entries)
                }
            }
        except Exception as e:
            return {"error": f"Failed to parse MAME log: {e}"}
    
    def _import_mame_recording(self, filepath: str, 
                             target_system: Optional[str] = None) -> Dict[str, Any]:
        """
        Import MAME input recording.
        
        Args:
            filepath: Path to recording file
            target_system: Target system type
            
        Returns:
            Dictionary with imported recording data
        """
        try:
            with open(filepath, 'rb') as f:
                data = f.read()
            
            # MAME INP format is binary
            # Try to extract header information
            header = {}
            
            if data.startswith(b'MAMEINP\0'):
                # MAME input recording file
                if target_system is None:
                    # Try to extract system from header
                    if len(data) > 16:
                        # Machine name is typically stored in the header
                        machine_name = data[8:16].decode('ascii', errors='ignore').strip('\0')
                        
                        # Map to our system type
                        if machine_name.startswith('nes'):
                            target_system = 'nes'
                        elif machine_name.startswith('snes'):
                            target_system = 'snes'
                        elif machine_name.startswith('genesis') or machine_name.startswith('megadriv'):
                            target_system = 'genesis'
                        elif machine_name.startswith('a2600'):
                            target_system = 'atari2600'
                        else:
                            target_system = machine_name
                
                header = {
                    "signature": "MAMEINP",
                    "machine": target_system
                }
            
            return {
                "format": f"mame_recording_{target_system or 'unknown'}",
                "system": target_system or 'unknown',
                "header": header,
                "binary_size": len(data),
                "note": "Full MAME recording parsing is not implemented"
            }
        except Exception as e:
            return {"error": f"Failed to parse MAME recording: {e}"}
    
    def _import_mame_savestate(self, filepath: str, 
                             target_system: Optional[str] = None) -> Dict[str, Any]:
        """
        Import MAME savestate.
        
        Args:
            filepath: Path to savestate file
            target_system: Target system type
            
        Returns:
            Dictionary with imported savestate data
        """
        try:
            with open(filepath, 'rb') as f:
                data = f.read()
            
            # MAME savestate format is binary and complex
            if target_system is None:
                # Try to determine system from savestate (difficult without detailed format knowledge)
                target_system = 'unknown'
            
            return {
                "format": f"mame_savestate_{target_system}",
                "system": target_system,
                "binary_size": len(data),
                "note": "Full MAME savestate parsing is not implemented"
            }
        except Exception as e:
            return {"error": f"Failed to parse MAME savestate: {e}"}
    
    def _import_genesis_plus_gx(self, filepath: str, 
                              target_system: Optional[str] = None) -> Dict[str, Any]:
        """
        Import Genesis Plus GX data.
        
        Args:
            filepath: Path to data file
            target_system: Target system type
            
        Returns:
            Dictionary with imported data
        """
        # Genesis Plus GX is specific to Sega systems
        if target_system is None:
            target_system = "genesis"
        elif target_system not in ["genesis", "sms"]:
            return {"error": f"Genesis Plus GX importer only supports Genesis/SMS, not {target_system}"}
        
        _, ext = os.path.splitext(filepath)
        
        if ext.lower() == '.gs0' or ext.lower() == '.gs1' or ext.lower() == '.gs2':
            # Savestate
            return self._import_genesis_plus_savestate(filepath, target_system)
        elif ext.lower() == '.log' or ext.lower() == '.txt':
            # Debug log
            return self._import_genesis_plus_log(filepath, target_system)
        else:
            return {"error": f"Unsupported Genesis Plus GX file format: {ext}"}
    
    def _import_genesis_plus_savestate(self, filepath: str, 
                                     target_system: str) -> Dict[str, Any]:
        """
        Import Genesis Plus GX savestate.
        
        Args:
            filepath: Path to savestate file
            target_system: Target system type
            
        Returns:
            Dictionary with imported savestate data
        """
        try:
            with open(filepath, 'rb') as f:
                data = f.read()
            
            # Genesis Plus GX savestate format
            return {
                "format": f"genesis_plus_savestate_{target_system}",
                "system": target_system,
                "binary_size": len(data),
                "note": "Full Genesis Plus GX savestate parsing is not implemented"
            }
        except Exception as e:
            return {"error": f"Failed to parse Genesis Plus GX savestate: {e}"}
    
    def _import_genesis_plus_log(self, filepath: str, 
                               target_system: str) -> Dict[str, Any]:
        """
        Import Genesis Plus GX debug log.
        
        Args:
            filepath: Path to debug log
            target_system: Target system type
            
        Returns:
            Dictionary with imported log data
        """
        try:
            with open(filepath, 'r') as f:
                lines = f.readlines()
            
            # Parse log entries
            log_entries = []
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                log_entries.append(line)
            
            return {
                "format": f"genesis_plus_log_{target_system}",
                "system": target_system,
                "entries": log_entries,
                "metadata": {
                    "total_entries": len(log_entries)
                }
            }
        except Exception as e:
            return {"error": f"Failed to parse Genesis Plus GX log: {e}"}
    
    def _import_stella(self, filepath: str, 
                     target_system: Optional[str] = None) -> Dict[str, Any]:
        """
        Import Stella (Atari 2600) data.
        
        Args:
            filepath: Path to data file
            target_system: Target system type
            
        Returns:
            Dictionary with imported data
        """
        # Stella is specific to Atari 2600
        if target_system is None:
            target_system = "atari2600"
        elif target_system != "atari2600":
            return {"error": f"Stella importer only supports Atari 2600, not {target_system}"}
        
        _, ext = os.path.splitext(filepath)
        
        if ext.lower() == '.state':
            # Savestate
            return self._import_stella_savestate(filepath)
        elif ext.lower() == '.log' or ext.lower() == '.txt':
            # Debug log
            return self._import_stella_log(filepath)
        else:
            return {"error": f"Unsupported Stella file format: {ext}"}
    
    def _import_stella_savestate(self, filepath: str) -> Dict[str, Any]:
        """
        Import Stella savestate.
        
        Args:
            filepath: Path to savestate file
            
        Returns:
            Dictionary with imported savestate data
        """
        try:
            with open(filepath, 'rb') as f:
                data = f.read()
            
            # Stella savestate format
            return {
                "format": "stella_savestate",
                "system": "atari2600",
                "binary_size": len(data),
                "note": "Full Stella savestate parsing is not implemented"
            }
        except Exception as e:
            return {"error": f"Failed to parse Stella savestate: {e}"}
    
    def _import_stella_log(self, filepath: str) -> Dict[str, Any]:
        """
        Import Stella debug log.
        
        Args:
            filepath: Path to debug log
            
        Returns:
            Dictionary with imported log data
        """
        try:
            with open(filepath, 'r') as f:
                lines = f.readlines()
            
            # Parse log entries
            log_entries = []
            tia_states = []
            cpu_states = []
            
            current_cycle = 0
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Check for TIA state
                tia_match = re.search(r'TIA CL#(\d+)', line)
                if tia_match:
                    tia_scanline = int(tia_match.group(1))
                    tia_states.append({
                        "cycle": current_cycle,
                        "scanline": tia_scanline,
                        "data": line
                    })
                
                # Check for CPU state
                cpu_match = re.search(r'(\w+): ([0-9A-F]+)', line)
                if cpu_match and 'PC' in line:
                    # Likely a CPU state line
                    registers = {}
                    
                    for reg_match in re.finditer(r'(\w+): ([0-9A-F]+)', line):
                        reg = reg_match.group(1)
                        value = int(reg_match.group(2), 16)
                        registers[reg] = value
                    
                    if registers:
                        cpu_states.append({
                            "cycle": current_cycle,
                            "registers": registers,
                            "data": line
                        })
                
                # Add general log entry
                log_entries.append({
                    "cycle": current_cycle,
                    "text": line
                })
                
                # Increment cycle
                current_cycle += 1
            
            return {
                "format": "stella_log",
                "system": "atari2600",
                "entries": log_entries,
                "tia_states": tia_states,
                "cpu_states": cpu_states,
                "metadata": {
                    "total_entries": len(log_entries),
                    "tia_states": len(tia_states),
                    "cpu_states": len(cpu_states)
                }
            }
        except Exception as e:
            return {"error": f"Failed to parse Stella log: {e}"}
    
    def _import_logic_analyzer(self, filepath: str, 
                             target_system: Optional[str] = None) -> Dict[str, Any]:
        """
        Import logic analyzer data.
        
        Args:
            filepath: Path to data file
            target_system: Target system type
            
        Returns:
            Dictionary with imported data
        """
        # Logic analyzer data can be for any system
        _, ext = os.path.splitext(filepath)
        
        if ext.lower() == '.csv':
            # CSV format
            return self._import_logic_analyzer_csv(filepath, target_system)
        elif ext.lower() == '.vcd':
            # Value Change Dump format
            return self._import_logic_analyzer_vcd(filepath, target_system)
        elif ext.lower() == '.bin' or ext.lower() == '.dat':
            # Binary format
            return self._import_logic_analyzer_binary(filepath, target_system)
        else:
            return {"error": f"Unsupported logic analyzer file format: {ext}"}
    
    def _import_logic_analyzer_csv(self, filepath: str, 
                                 target_system: Optional[str] = None) -> Dict[str, Any]:
        """
        Import logic analyzer CSV data.
        
        Args:
            filepath: Path to CSV file
            target_system: Target system type
            
        Returns:
            Dictionary with imported data
        """
        try:
            # Read CSV file
            with open(filepath, 'r', newline='') as f:
                reader = csv.reader(f)
                headers = next(reader)
                
                # Parse data
                data = []
                signals = {header: [] for header in headers}
                
                for row in reader:
                    data.append(row)
                    for i, value in enumerate(row):
                        if i < len(headers):
                            try:
                                signals[headers[i]].append(float(value) if '.' in value else int(value))
                            except ValueError:
                                signals[headers[i]].append(value)
            
            # Convert to numpy arrays if possible
            for header in headers:
                try:
                    signals[header] = np.array(signals[header])
                except:
                    pass
            
            return {
                "format": "logic_analyzer_csv",
                "system": target_system,
                "headers": headers,
                "signals": signals,
                "metadata": {
                    "total_samples": len(data)
                }
            }
        except Exception as e:
            return {"error": f"Failed to parse logic analyzer CSV: {e}"}
    
    def _import_logic_analyzer_vcd(self, filepath: str, 
                                 target_system: Optional[str] = None) -> Dict[str, Any]:
        """
        Import logic analyzer VCD data.
        
        Args:
            filepath: Path to VCD file
            target_system: Target system type
            
        Returns:
            Dictionary with imported data
        """
        try:
            # VCD format is a specialized format for waveform data
            with open(filepath, 'r') as f:
                content = f.read()
            
            # Parse header information
            header = {}
            signals = {}
            
            # Extract date
            date_match = re.search(r'\$date\s+(.*?)\s+\$end', content, re.DOTALL)
            if date_match:
                header['date'] = date_match.group(1).strip()
            
            # Extract version
            version_match = re.search(r'\$version\s+(.*?)\s+\$end', content, re.DOTALL)
            if version_match:
                header['version'] = version_match.group(1).strip()
            
            # Extract timescale
            timescale_match = re.search(r'\$timescale\s+(.*?)\s+\$end', content, re.DOTALL)
            if timescale_match:
                header['timescale'] = timescale_match.group(1).strip()
            
            # Extract signal definitions
            scope_matches = re.finditer(r'\$scope\s+module\s+(.*?)\s+\$end(.*?)\$upscope', content, re.DOTALL)
            
            for scope_match in scope_matches:
                module = scope_match.group(1).strip()
                scope_content = scope_match.group(2)
                
                # Extract signal declarations
                var_matches = re.finditer(r'\$var\s+(\w+)\s+(\d+)\s+(\S+)\s+(\S+)(?:\s+\[(\d+):(\d+)\])?\s+\$end', scope_content)
                
                for var_match in var_matches:
                    var_type = var_match.group(1)
                    var_size = int(var_match.group(2))
                    var_id = var_match.group(3)
                    var_name = var_match.group(4)
                    
                    signals[var_id] = {
                        'name': var_name,
                        'type': var_type,
                        'size': var_size,
                        'module': module,
                        'values': []
                    }
            
            # Simple extraction of signal values
            # Full VCD parsing is complex and would require more detailed implementation
            
            return {
                "format": "logic_analyzer_vcd",
                "system": target_system,
                "header": header,
                "signals": signals,
                "metadata": {
                    "total_signals": len(signals)
                },
                "note": "Full VCD parsing is not implemented"
            }
        except Exception as e:
            return {"error": f"Failed to parse logic analyzer VCD: {e}"}
    
    def _import_logic_analyzer_binary(self, filepath: str, 
                                    target_system: Optional[str] = None) -> Dict[str, Any]:
        """
        Import logic analyzer binary data.
        
        Args:
            filepath: Path to binary file
            target_system: Target system type
            
        Returns:
            Dictionary with imported data
        """
        try:
            with open(filepath, 'rb') as f:
                data = f.read()
            
            # Binary formats vary significantly
            # This is a placeholder for custom binary format parsing
            
            return {
                "format": "logic_analyzer_binary",
                "system": target_system,
                "binary_size": len(data),
                "note": "Binary logic analyzer parsing is not implemented"
            }
        except Exception as e:
            return {"error": f"Failed to parse logic analyzer binary data: {e}"}
    
    def _import_quantum_signals(self, filepath: str, 
                              target_system: Optional[str] = None) -> Dict[str, Any]:
        """
        Import quantum signal data.
        
        Args:
            filepath: Path to data file
            target_system: Target system type
            
        Returns:
            Dictionary with imported data
        """
        # Handle quantum signal data in various formats
        _, ext = os.path.splitext(filepath)
        
        if ext.lower() == '.json':
            # JSON format
            return self._import_quantum_signals_json(filepath, target_system)
        elif ext.lower() == '.pkl' or ext.lower() == '.pickle':
            # Pickle format
            return self._import_quantum_signals_pickle(filepath, target_system)
        elif ext.lower() == '.npz':
            # NumPy archive
            return self._import_quantum_signals_numpy(filepath, target_system)
        else:
            return {"error": f"Unsupported quantum signals file format: {ext}"}
    
    def _import_quantum_signals_json(self, filepath: str, 
                                   target_system: Optional[str] = None) -> Dict[str, Any]:
        """
        Import quantum signal data in JSON format.
        
        Args:
            filepath: Path to JSON file
            target_system: Target system type
            
        Returns:
            Dictionary with imported data
        """
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Check for required fields
            if 'quantum_data' not in data:
                return {"error": "Not a valid quantum signals JSON file"}
            
            # Extract system type if not specified
            if target_system is None and 'system' in data:
                target_system = data['system']
            
            return {
                "format": "quantum_signals_json",
                "system": target_system,
                "data": data,
                "metadata": data.get('metadata', {})
            }
        except Exception as e:
            return {"error": f"Failed to parse quantum signals JSON: {e}"}
    
    def _import_quantum_signals_pickle(self, filepath: str, 
                                     target_system: Optional[str] = None) -> Dict[str, Any]:
        """
        Import quantum signal data in pickle format.
        
        Args:
            filepath: Path to pickle file
            target_system: Target system type
            
        Returns:
            Dictionary with imported data
        """
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            # Check for required fields
            if not isinstance(data, dict) or 'quantum_data' not in data:
                return {"error": "Not a valid quantum signals pickle file"}
            
            # Extract system type if not specified
            if target_system is None and 'system' in data:
                target_system = data['system']
            
            return {
                "format": "quantum_signals_pickle",
                "system": target_system,
                "data": data,
                "metadata": data.get('metadata', {})
            }
        except Exception as e:
            return {"error": f"Failed to parse quantum signals pickle: {e}"}
    
    def _import_quantum_signals_numpy(self, filepath: str, 
                                    target_system: Optional[str] = None) -> Dict[str, Any]:
        """
        Import quantum signal data in NumPy archive format.
        
        Args:
            filepath: Path to NPZ file
            target_system: Target system type
            
        Returns:
            Dictionary with imported data
        """
        try:
            # Load NumPy archive
            with np.load(filepath) as npz:
                # Convert to dictionary
                data = {key: npz[key] for key in npz.files}
            
            # Check for required arrays
            if 'quantum_data' not in data:
                return {"error": "Not a valid quantum signals NumPy archive"}
            
            # Extract system type if not specified
            if target_system is None and 'system' in data:
                target_system = str(data['system'])
            
            # Convert numpy arrays to lists for JSON serialization
            metadata = {}
            if 'metadata' in data:
                try:
                    metadata = data['metadata'].item()
                except:
                    metadata = {'note': 'Metadata could not be converted'}
            
            return {
                "format": "quantum_signals_numpy",
                "system": target_system,
                "arrays": {key: data[key].tolist() if hasattr(data[key], 'tolist') else data[key] 
                           for key in data},
                "metadata": metadata
            }
        except Exception as e:
            return {"error": f"Failed to parse quantum signals NumPy archive: {e}"}
    
    def convert_to_state_history(self, imported_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Convert imported data to standard state history format.
        
        Args:
            imported_data: Data from import_data method
            
        Returns:
            List of state dictionaries in standard format
        """
        if "error" in imported_data:
            logger.error(f"Cannot convert data with error: {imported_data['error']}")
            return []
            
        format_type = imported_data.get("format", "")
        system = imported_data.get("system", "unknown")
        
        # Handle different formats
        if "state_history" in imported_data:
            # Already in correct format
            return imported_data["state_history"]
            
        elif format_type.startswith("fceux_trace"):
            # Convert FCEUX trace
            return self._convert_fceux_trace(imported_data)
            
        elif format_type.startswith("mesen_trace"):
            # Convert Mesen trace
            return self._convert_mesen_trace(imported_data)
            
        elif format_type.startswith("bizhawk_trace"):
            # Convert BizHawk trace
            return self._convert_bizhawk_trace(imported_data)
            
        elif format_type.startswith("mame_log"):
            # Convert MAME log
            return self._convert_mame_log(imported_data)
            
        elif format_type.startswith("stella_log"):
            # Convert Stella log
            return self._convert_stella_log(imported_data)
            
        elif format_type.startswith("logic_analyzer_csv"):
            # Convert logic analyzer CSV
            return self._convert_logic_analyzer_csv(imported_data)
            
        elif format_type.startswith("quantum_signals"):
            # Convert quantum signals data
            return self._convert_quantum_signals(imported_data)
            
        else:
            logger.warning(f"No converter available for format: {format_type}")
            return []
    
    def _convert_fceux_trace(self, imported_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Convert FCEUX trace data to standard state history.
        
        Args:
            imported_data: FCEUX trace data
            
        Returns:
            List of state dictionaries
        """
        # FCEUX trace format might already be in the correct format
        if "entries" in imported_data:
            entries = imported_data["entries"]
            
            # Convert entries to state history
            state_history = []
            
            for entry in entries:
                if isinstance(entry, dict):
                    state_history.append(entry)
                else:
                    # Parse string entry
                    # Format varies, this is a simple parsing example
                    state = {"data": entry}
                    state_history.append(state)
            
            return state_history
            
        # Return as-is if no conversion needed
        return imported_data.get("state_history", [])
    
    def _convert_mesen_trace(self, imported_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Convert Mesen trace data to standard state history.
        
        Args:
            imported_data: Mesen trace data
            
        Returns:
            List of state dictionaries
        """
        # Return as-is if already in correct format
        return imported_data.get("state_history", [])
    
    def _convert_bizhawk_trace(self, imported_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Convert BizHawk trace data to standard state history.
        
        Args:
            imported_data: BizHawk trace data
            
        Returns:
            List of state dictionaries
        """
        # Return as-is if already in correct format
        return imported_data.get("state_history", [])
    
    def _convert_mame_log(self, imported_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Convert MAME log data to standard state history.
        
        Args:
            imported_data: MAME log data
            
        Returns:
            List of state dictionaries
        """
        if "entries" in imported_data:
            entries = imported_data["entries"]
            
            # Convert entries to state history
            state_history = []
            
            for entry in entries:
                if isinstance(entry, dict):
                    state_history.append(entry)
                else:
                    # Parse string entry
                    state = {"data": entry}
                    state_history.append(state)
            
            return state_history
            
        return []
    
    def _convert_stella_log(self, imported_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Convert Stella log data to standard state history.
        
        Args:
            imported_data: Stella log data
            
        Returns:
            List of state dictionaries
        """
        state_history = []
        
        # Use CPU states if available
        if "cpu_states" in imported_data:
            return imported_data["cpu_states"]
            
        # Use TIA states if available
        if "tia_states" in imported_data:
            return imported_data["tia_states"]
            
        # Use general entries
        if "entries" in imported_data:
            for entry in imported_data["entries"]:
                if isinstance(entry, dict):
                    state_history.append(entry)
                else:
                    state_history.append({"data": entry})
        
        return state_history
    
    def _convert_logic_analyzer_csv(self, imported_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Convert logic analyzer CSV data to standard state history.
        
        Args:
            imported_data: Logic analyzer CSV data
            
        Returns:
            List of state dictionaries
        """
        state_history = []
        
        if "signals" in imported_data and imported_data["signals"]:
            # Get time/sample index
            headers = imported_data["headers"]
            signals = imported_data["signals"]
            
            # Assume first column is time/sample or create index
            if headers and signals:
                time_header = headers[0]
                time_values = signals[time_header]
                
                # Determine number of samples
                num_samples = len(time_values)
                
                # Create state for each sample
                for i in range(num_samples):
                    state = {
                        "cycle": i,
                        "time": time_values[i] if time_values else i,
                        "signals": {}
                    }
                    
                    # Add signals
                    for header in headers[1:]:
                        if header in signals and i < len(signals[header]):
                            state["signals"][header] = signals[header][i]
                    
                    state_history.append(state)
        
        return state_history
    
    def _convert_quantum_signals(self, imported_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Convert quantum signals data to standard state history.
        
        Args:
            imported_data: Quantum signals data
            
        Returns:
            List of state dictionaries
        """
        state_history = []
        
        # Extract data based on format
        if "data" in imported_data and "quantum_data" in imported_data["data"]:
            quantum_data = imported_data["data"]["quantum_data"]
            
            # Convert quantum data to state history
            # Format depends on specific quantum data structure
            if isinstance(quantum_data, list):
                # List of states
                for i, state_data in enumerate(quantum_data):
                    if isinstance(state_data, dict):
                        state = state_data.copy()
                        if "cycle" not in state:
                            state["cycle"] = i
                        state_history.append(state)
            elif isinstance(quantum_data, dict):
                # Dictionary of states or signals
                if "states" in quantum_data:
                    return quantum_data["states"]
                else:
                    # Create states from signals
                    signals = {}
                    for key, value in quantum_data.items():
                        if isinstance(value, (list, np.ndarray)):
                            signals[key] = value
                    
                    # Determine number of samples
                    num_samples = max(len(signal) for signal in signals.values()) if signals else 0
                    
                    # Create state for each sample
                    for i in range(num_samples):
                        state = {
                            "cycle": i,
                            "quantum_signals": {}
                        }
                        
                        # Add signals
                        for key, signal in signals.items():
                            if i < len(signal):
                                state["quantum_signals"][key] = signal[i]
                        
                        state_history.append(state)
        
        # Check arrays for NumPy format
        elif "arrays" in imported_data and "quantum_data" in imported_data["arrays"]:
            quantum_data = imported_data["arrays"]["quantum_data"]
            
            # Convert quantum data array to state history
            if isinstance(quantum_data, list):
                # Assume list of values at different time steps
                for i, value in enumerate(quantum_data):
                    state = {
                        "cycle": i,
                        "quantum_value": value
                    }
                    
                    state_history.append(state)
        
        return state_history