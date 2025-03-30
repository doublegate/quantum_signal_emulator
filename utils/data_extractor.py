"""
Data extraction tools for ROM and system state analysis.

This module provides tools for extracting and analyzing data from ROM files
and system state history. It supports various ROM formats and can extract
timing data, register patterns, and other useful information for analysis.
"""

import os
import logging
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union, BinaryIO
from collections import defaultdict
import struct

# Configure logging
logger = logging.getLogger("QuantumSignalEmulator.DataExtractor")

class DataExtractor:
    """
    Data extraction and analysis tools for ROM and system state data.
    
    This class provides methods for extracting meaningful data from ROM files
    and system state history. It supports detection of timing patterns, register
    usage patterns, and other metrics useful for hardware analysis.
    """
    
    def __init__(self, debug_mode: bool = False):
        """
        Initialize the data extractor.
        
        Args:
            debug_mode: Enable additional debug information
        """
        self.debug_mode = debug_mode
        self.rom_data = None
        self.rom_format = None
        self.system_type = None
        self.header_data = {}
        
        # Register ROM format handlers
        self.format_handlers = {
            'nes': self._parse_nes_rom,
            'snes': self._parse_snes_rom,
            'genesis': self._parse_genesis_rom,
            'atari2600': self._parse_atari_rom
        }
        
        logger.info("Initialized data extractor")
    
    def load_rom(self, rom_path: str) -> bool:
        """
        Load and analyze a ROM file.
        
        Args:
            rom_path: Path to ROM file
            
        Returns:
            True if ROM loaded successfully, False otherwise
        """
        try:
            # Check if file exists
            if not os.path.exists(rom_path):
                logger.error(f"ROM file not found: {rom_path}")
                return False
                
            # Load file data
            with open(rom_path, 'rb') as f:
                self.rom_data = f.read()
                
            # Detect ROM format based on extension and content
            _, ext = os.path.splitext(rom_path)
            self.rom_format = self._detect_rom_format(ext, self.rom_data)
            
            if not self.rom_format:
                logger.error(f"Unknown ROM format: {rom_path}")
                return False
                
            # Parse ROM using appropriate handler
            logger.info(f"Detected ROM format: {self.rom_format}")
            self.system_type = self.rom_format
            handler = self.format_handlers.get(self.rom_format)
            
            if handler:
                handler(self.rom_data)
                return True
            else:
                logger.error(f"No handler available for ROM format: {self.rom_format}")
                return False
                
        except Exception as e:
            logger.error(f"Error loading ROM: {e}")
            if self.debug_mode:
                import traceback
                traceback.print_exc()
            return False
    
    def _detect_rom_format(self, extension: str, data: bytes) -> Optional[str]:
        """
        Detect ROM format based on file extension and contents.
        
        Args:
            extension: File extension (with dot)
            data: ROM file data
            
        Returns:
            Detected ROM format or None if unknown
        """
        # Normalize extension
        ext = extension.lower()
        
        # Try extension-based detection first
        if ext in ['.nes']:
            # Validate NES format (check for 'NES' + MS-DOS EOF)
            if data[:4] == b'NES\x1a':
                return 'nes'
                
        elif ext in ['.sfc', '.smc']:
            # Check for SNES header or characteristics
            # SMC format has a 512-byte header
            if len(data) % 1024 == 512:
                # Likely SMC format with header
                return 'snes'
            # SFC format has no header and size is multiple of 1024
            elif len(data) % 1024 == 0:
                return 'snes'
                
        elif ext in ['.md', '.bin', '.gen']:
            # Check for SEGA header text
            if b'SEGA' in data[:0x100]:
                return 'genesis'
                
        elif ext in ['.a26', '.bin']:
            # Atari ROMs have no specific header
            # Simple heuristic: Atari 2600 ROMs are typically small (2KB-32KB)
            if 2048 <= len(data) <= 32768:
                return 'atari2600'
        
        # If extension-based detection failed, try content-based detection
        
        # Check for NES header
        if data[:4] == b'NES\x1a':
            return 'nes'
            
        # Check for SEGA Genesis header
        if b'SEGA' in data[:0x100]:
            return 'genesis'
            
        # Check for SNES characteristics
        # This is a simplification; real detection would be more complex
        if len(data) >= 32768:  # At least 32KB
            # Check for reset vectors at specific locations
            if len(data) % 1024 == 0:
                return 'snes'
                
        # Fallback: Unable to detect
        return None
    
    def _parse_nes_rom(self, data: bytes) -> None:
        """
        Parse NES ROM format.
        
        Args:
            data: ROM data
        """
        # Parse iNES header
        if len(data) < 16:
            logger.error("Invalid NES ROM: too small")
            return
            
        # Extract header data
        self.header_data = {
            'format': 'iNES',
            'prg_rom_size': data[4] * 16384,  # PRG ROM size in bytes (16KB units)
            'chr_rom_size': data[5] * 8192,   # CHR ROM size in bytes (8KB units)
            'mapper': ((data[6] >> 4) | (data[7] & 0xF0)),  # Mapper number
            'mirroring': 'vertical' if (data[6] & 1) else 'horizontal',
            'battery': bool(data[6] & 2),
            'trainer': bool(data[6] & 4),
            'four_screen': bool(data[6] & 8),
            'vs_unisystem': bool(data[7] & 1),
            'playchoice10': bool(data[7] & 2),
            'nes2_format': (data[7] & 0x0C) == 0x08,
            'prg_ram_size': data[8] * 8192 if data[8] > 0 else 8192,  # PRG RAM size (8KB units)
            'tv_system': 'PAL' if (data[9] & 1) else 'NTSC'
        }
        
        # Extract ROM sections
        offset = 16  # Start after header
        
        # Skip trainer if present
        if self.header_data['trainer']:
            offset += 512
            
        # Extract PRG ROM
        prg_size = self.header_data['prg_rom_size']
        if offset + prg_size <= len(data):
            self.prg_rom = data[offset:offset+prg_size]
            offset += prg_size
        else:
            logger.warning("PRG ROM data truncated or missing")
            self.prg_rom = data[offset:]
            return
            
        # Extract CHR ROM if present
        chr_size = self.header_data['chr_rom_size']
        if chr_size > 0 and offset + chr_size <= len(data):
            self.chr_rom = data[offset:offset+chr_size]
        else:
            logger.info("No CHR ROM present or data truncated")
            self.chr_rom = None
            
        # Analyze program code
        self._analyze_nes_code()
    
    def _parse_snes_rom(self, data: bytes) -> None:
        """
        Parse SNES ROM format.
        
        Args:
            data: ROM data
        """
        # Check for SMC header (512 bytes)
        has_header = len(data) % 1024 == 512
        
        if has_header:
            # Skip 512-byte header
            rom_data = data[512:]
            self.header_data['format'] = 'SMC'
        else:
            rom_data = data
            self.header_data['format'] = 'SFC'
            
        # Determine ROM type and size
        rom_size = len(rom_data)
        
        # Get ROM title (21 bytes at specified offset, depending on ROM type)
        # For LoROM, title is at 0x7FC0, for HiROM at 0xFFC0
        if rom_size >= 0x8000:  # At least 32KB to have header
            # Try both LoROM and HiROM locations
            lo_title_offset = 0x7FC0
            hi_title_offset = 0xFFC0
            
            # Check if we can extract title from either location
            if lo_title_offset + 21 <= rom_size:
                lo_title = rom_data[lo_title_offset:lo_title_offset+21].decode('ascii', errors='ignore').strip('\x00')
            else:
                lo_title = ""
                
            if hi_title_offset + 21 <= rom_size:
                hi_title = rom_data[hi_title_offset:hi_title_offset+21].decode('ascii', errors='ignore').strip('\x00')
            else:
                hi_title = ""
                
            # Determine ROM type based on title validity
            if lo_title and all(32 <= ord(c) < 127 for c in lo_title):
                self.header_data['mapping'] = 'LoROM'
                self.header_data['title'] = lo_title
                self.header_data['rom_size'] = rom_size
                
                # Extract other header info from LoROM location
                if lo_title_offset + 48 <= rom_size:
                    self.header_data['rom_type'] = rom_data[lo_title_offset+0x16]
                    self.header_data['rom_speed'] = "Fast" if (rom_data[lo_title_offset+0x25] & 0x10) else "Slow"
                    self.header_data['has_ram'] = (rom_data[lo_title_offset+0x16] & 0x02) != 0
                    self.header_data['has_battery'] = (rom_data[lo_title_offset+0x16] & 0x04) != 0
                    
            elif hi_title and all(32 <= ord(c) < 127 for c in hi_title):
                self.header_data['mapping'] = 'HiROM'
                self.header_data['title'] = hi_title
                self.header_data['rom_size'] = rom_size
                
                # Extract other header info from HiROM location
                if hi_title_offset + 48 <= rom_size:
                    self.header_data['rom_type'] = rom_data[hi_title_offset+0x16]
                    self.header_data['rom_speed'] = "Fast" if (rom_data[hi_title_offset+0x25] & 0x10) else "Slow"
                    self.header_data['has_ram'] = (rom_data[hi_title_offset+0x16] & 0x02) != 0
                    self.header_data['has_battery'] = (rom_data[hi_title_offset+0x16] & 0x04) != 0
            else:
                logger.warning("Could not determine SNES ROM mapping type")
                self.header_data['mapping'] = 'Unknown'
                self.header_data['rom_size'] = rom_size
        else:
            logger.warning("SNES ROM too small to contain header information")
            self.header_data['mapping'] = 'Unknown'
            self.header_data['rom_size'] = rom_size
            
        # Store ROM data
        self.rom_data = rom_data
    
    def _parse_genesis_rom(self, data: bytes) -> None:
        """
        Parse Sega Genesis/Mega Drive ROM format.
        
        Args:
            data: ROM data
        """
        # Check for SEGA header
        sega_offset = -1
        for i in range(min(0x100, len(data) - 4)):
            if data[i:i+4] == b'SEGA':
                sega_offset = i
                break
                
        if sega_offset == -1:
            logger.warning("SEGA header not found in Genesis ROM")
            self.header_data = {
                'format': 'Unknown Sega',
                'rom_size': len(data)
            }
            return
            
        # Parse header (typically at 0x100-0x1FF)
        header_start = sega_offset
        
        # Extract system and copyright info
        if header_start + 16 <= len(data):
            system = data[header_start+0x0:header_start+0x10].decode('ascii', errors='ignore').strip()
            self.header_data['system'] = system
            
        if header_start + 32 <= len(data):
            copyright_info = data[header_start+0x10:header_start+0x30].decode('ascii', errors='ignore').strip()
            self.header_data['copyright'] = copyright_info
            
        # Extract domestic and international names
        if header_start + 80 <= len(data):
            domestic_name = data[header_start+0x30:header_start+0x50].decode('ascii', errors='ignore').strip()
            self.header_data['domestic_name'] = domestic_name
            
        if header_start + 128 <= len(data):
            int_name = data[header_start+0x50:header_start+0x80].decode('ascii', errors='ignore').strip()
            self.header_data['international_name'] = int_name
            
        # Extract ROM info
        if header_start + 160 <= len(data):
            rom_type = data[header_start+0x80:header_start+0xA0].decode('ascii', errors='ignore').strip()
            self.header_data['rom_type'] = rom_type
            
        # Store full header and ROM data
        self.header_data['format'] = 'Sega Genesis'
        self.header_data['rom_size'] = len(data)
        self.header_data['header_offset'] = header_start
        
        # Store ROM data
        self.rom_data = data
    
    def _parse_atari_rom(self, data: bytes) -> None:
        """
        Parse Atari 2600 ROM format.
        
        Args:
            data: ROM data
        """
        # Atari 2600 ROMs have no standard header
        # They're typically 2KB, 4KB, or 8KB binaries
        
        rom_size = len(data)
        
        # Store basic information
        self.header_data = {
            'format': 'Atari 2600',
            'rom_size': rom_size,
            'bankswitching': self._detect_atari_bankswitching(data)
        }
        
        # Store ROM data
        self.rom_data = data
    
    def _detect_atari_bankswitching(self, data: bytes) -> str:
        """
        Detect Atari 2600 bankswitching method.
        
        Args:
            data: ROM data
            
        Returns:
            Detected bankswitching method
        """
        # Simple heuristic based on ROM size
        rom_size = len(data)
        
        if rom_size <= 4096:
            return "None/Standard"  # 2K or 4K ROM, no bankswitching needed
        elif rom_size == 8192:
            return "F8"  # 8K ROM, likely F8 bankswitching
        elif rom_size == 12288:
            return "FA"  # 12K ROM, likely FA bankswitching
        elif rom_size == 16384:
            return "F6"  # 16K ROM, likely F6 bankswitching
        elif rom_size == 32768:
            return "F4"  # 32K ROM, likely F4 bankswitching
        else:
            return "Unknown"  # Non-standard size
    
    def _analyze_nes_code(self) -> None:
        """
        Analyze NES program code for patterns and metrics.
        """
        if not hasattr(self, 'prg_rom') or not self.prg_rom:
            return
            
        # Opcode frequency analysis
        opcodes = defaultdict(int)
        for byte in self.prg_rom:
            opcodes[byte] += 1
            
        # Find most common opcodes
        common_opcodes = sorted(opcodes.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Store analysis results
        self.code_analysis = {
            'total_bytes': len(self.prg_rom),
            'unique_bytes': len(opcodes),
            'common_opcodes': common_opcodes,
            'opcode_entropy': self._calculate_entropy(list(opcodes.values()))
        }
    
    def _calculate_entropy(self, values: List[int]) -> float:
        """
        Calculate Shannon entropy of a value distribution.
        
        Args:
            values: List of counts or values
            
        Returns:
            Entropy value in bits
        """
        total = sum(values)
        probabilities = [v / total for v in values if v > 0]
        return -sum(p * np.log2(p) for p in probabilities)
    
    def extract_instruction_timing(self) -> Dict[str, Any]:
        """
        Extract instruction timing patterns from ROM.
        
        Returns:
            Dictionary with timing analysis results
        """
        if not self.rom_data or not self.rom_format:
            return {"error": "No ROM loaded"}
            
        if self.rom_format == 'nes':
            return self._extract_nes_timing()
        elif self.rom_format == 'snes':
            return self._extract_snes_timing()
        elif self.rom_format == 'genesis':
            return self._extract_genesis_timing()
        elif self.rom_format == 'atari2600':
            return self._extract_atari_timing()
        else:
            return {"error": "Unsupported system for timing extraction"}
    
    def _extract_nes_timing(self) -> Dict[str, Any]:
        """
        Extract instruction timing for NES ROMs.
        
        Returns:
            Dictionary with NES timing analysis
        """
        if not hasattr(self, 'prg_rom') or not self.prg_rom:
            return {"error": "No PRG ROM data available"}
            
        # 6502 instruction cycle counts
        # Format: {opcode: (cycles, additional_if_page_crossed, bytes)}
        cycles_6502 = {
            # Logical and Arithmetic
            0x69: (2, 0, 2), 0x65: (3, 0, 2), 0x75: (4, 0, 2),  # ADC
            0x6D: (4, 0, 3), 0x7D: (4, 1, 3), 0x79: (4, 1, 3),
            0x61: (6, 0, 2), 0x71: (5, 1, 2),
            
            0x29: (2, 0, 2), 0x25: (3, 0, 2), 0x35: (4, 0, 2),  # AND
            0x2D: (4, 0, 3), 0x3D: (4, 1, 3), 0x39: (4, 1, 3),
            0x21: (6, 0, 2), 0x31: (5, 1, 2),
            
            # ... many more opcodes would be defined here ...
            
            # Just a few common ones for this example:
            0xA9: (2, 0, 2), 0xA5: (3, 0, 2), 0xB5: (4, 0, 2),  # LDA
            0xAD: (4, 0, 3), 0xBD: (4, 1, 3), 0xB9: (4, 1, 3),
            0xA1: (6, 0, 2), 0xB1: (5, 1, 2),
            
            0x4C: (3, 0, 3), 0x6C: (5, 0, 3),  # JMP
            
            0x20: (6, 0, 3),  # JSR
            0x60: (6, 0, 1),  # RTS
            
            0xEA: (2, 0, 1),  # NOP
        }
        
        # Analyze PRG ROM
        total_cycles = 0
        total_bytes = 0
        instructions = []
        
        # Simple linear analysis (not handling jumps/branches)
        i = 0
        while i < len(self.prg_rom):
            opcode = self.prg_rom[i]
            
            if opcode in cycles_6502:
                base_cycles, extra_cycles, size = cycles_6502[opcode]
                
                # Check if we have enough data to read the full instruction
                if i + size <= len(self.prg_rom):
                    # Extract instruction bytes
                    instr_bytes = self.prg_rom[i:i+size]
                    
                    # Store instruction info
                    instructions.append({
                        'address': i,
                        'opcode': opcode,
                        'bytes': list(instr_bytes),
                        'size': size,
                        'cycles': base_cycles
                    })
                    
                    # Update counters
                    total_cycles += base_cycles
                    total_bytes += size
                    i += size
                else:
                    # Not enough data, move to next byte
                    i += 1
            else:
                # Unknown opcode, treat as data
                i += 1
        
        # Calculate timing metrics
        avg_cycles_per_byte = total_cycles / total_bytes if total_bytes > 0 else 0
        avg_cycles_per_instruction = total_cycles / len(instructions) if instructions else 0
        
        # Find common instruction sequences
        sequences = self._find_common_sequences(instructions, 3)
        
        return {
            'total_instructions': len(instructions),
            'total_cycles': total_cycles,
            'total_bytes': total_bytes,
            'avg_cycles_per_byte': avg_cycles_per_byte,
            'avg_cycles_per_instruction': avg_cycles_per_instruction,
            'common_sequences': sequences
        }
    
    def _extract_snes_timing(self) -> Dict[str, Any]:
        """
        Extract instruction timing for SNES ROMs.
        
        Returns:
            Dictionary with SNES timing analysis
        """
        # For SNES, this would be a 65C816 analysis
        # Similar to NES but with 16-bit modes and new addressing modes
        return {"error": "SNES timing analysis not yet implemented"}
    
    def _extract_genesis_timing(self) -> Dict[str, Any]:
        """
        Extract instruction timing for Genesis ROMs.
        
        Returns:
            Dictionary with Genesis timing analysis
        """
        # For Genesis, this would analyze 68000 instructions
        return {"error": "Genesis timing analysis not yet implemented"}
    
    def _extract_atari_timing(self) -> Dict[str, Any]:
        """
        Extract instruction timing for Atari 2600 ROMs.
        
        Returns:
            Dictionary with Atari timing analysis
        """
        # Similar to NES but with 6507 (subset of 6502)
        return {"error": "Atari timing analysis not yet implemented"}
    
    def _find_common_sequences(self, instructions: List[Dict[str, Any]], min_length: int) -> List[Dict[str, Any]]:
        """
        Find common instruction sequences in code.
        
        Args:
            instructions: List of instruction dictionaries
            min_length: Minimum sequence length
            
        Returns:
            List of common sequence patterns
        """
        if not instructions or len(instructions) < min_length:
            return []
            
        # Extract opcode sequences
        sequences = {}
        
        for i in range(len(instructions) - min_length + 1):
            # Create sequence of opcodes
            seq = tuple(instr['opcode'] for instr in instructions[i:i+min_length])
            
            # Record occurrence
            if seq in sequences:
                sequences[seq].append(i)
            else:
                sequences[seq] = [i]
        
        # Filter to sequences that appear multiple times
        common_sequences = {seq: occurrences for seq, occurrences in sequences.items() if len(occurrences) > 1}
        
        # Sort by number of occurrences
        sorted_sequences = sorted(common_sequences.items(), key=lambda x: len(x[1]), reverse=True)
        
        # Format results (limit to top 10)
        result = []
        for seq, occurrences in sorted_sequences[:10]:
            # Calculate total cycles for sequence
            total_cycles = sum(instructions[occurrences[0] + i]['cycles'] for i in range(min_length))
            
            result.append({
                'opcodes': list(seq),
                'occurrences': len(occurrences),
                'first_address': instructions[occurrences[0]]['address'],
                'total_cycles': total_cycles
            })
            
        return result
    
    def analyze_state_history(self, state_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze system state history for timing and register patterns.
        
        Args:
            state_history: List of system state snapshots
            
        Returns:
            Dictionary with state analysis results
        """
        if not state_history:
            return {"error": "No state history provided"}
            
        # Extract timing information
        cycles = [state.get("cycle", i) for i, state in enumerate(state_history)]
        scanlines = [state.get("scanline", 0) for state in state_history]
        
        # Calculate cycle deltas
        cycle_deltas = [cycles[i] - cycles[i-1] for i in range(1, len(cycles))]
        
        # Extract register information
        register_values = defaultdict(list)
        for state in state_history:
            if "registers" in state:
                for reg, value in state["registers"].items():
                    register_values[reg].append(value)
        
        # Calculate register statistics
        register_stats = {}
        for reg, values in register_values.items():
            if not values:
                continue
                
            # Calculate changes
            changes = sum(1 for i in range(1, len(values)) if values[i] != values[i-1])
            
            # Calculate statistics
            register_stats[reg] = {
                "min": min(values),
                "max": max(values),
                "mean": np.mean(values),
                "std": np.std(values),
                "changes": changes,
                "change_rate": changes / (len(values) - 1) if len(values) > 1 else 0
            }
        
        # Calculate timing statistics
        if cycle_deltas:
            timing_stats = {
                "min_delta": min(cycle_deltas),
                "max_delta": max(cycle_deltas),
                "mean_delta": np.mean(cycle_deltas),
                "std_delta": np.std(cycle_deltas),
                "total_cycles": cycles[-1] - cycles[0] if len(cycles) > 1 else 0
            }
        else:
            timing_stats = {"error": "Not enough cycle data for statistics"}
        
        # Calculate scanline statistics
        if scanlines:
            unique_scanlines = len(set(scanlines))
            max_scanline = max(scanlines)
            
            # Detect scanline transitions
            transitions = []
            for i in range(1, len(scanlines)):
                if scanlines[i] != scanlines[i-1]:
                    transitions.append({
                        "from": scanlines[i-1],
                        "to": scanlines[i],
                        "at_cycle": cycles[i] if i < len(cycles) else None
                    })
            
            scanline_stats = {
                "unique_scanlines": unique_scanlines,
                "max_scanline": max_scanline,
                "transitions": transitions[:10]  # Limit to first 10 for brevity
            }
        else:
            scanline_stats = {"error": "No scanline data available"}
        
        return {
            "timing": timing_stats,
            "scanlines": scanline_stats,
            "registers": register_stats
        }
    
    def export_analysis(self, filename: str, format: str = 'json') -> bool:
        """
        Export ROM and analysis data to file.
        
        Args:
            filename: Output filename
            format: Export format ('json', 'csv', or 'text')
            
        Returns:
            True if export successful, False otherwise
        """
        if not self.rom_data or not self.rom_format:
            logger.error("No ROM data to export")
            return False
            
        try:
            # Create export data structure
            export_data = {
                "rom_format": self.rom_format,
                "header_data": self.header_data
            }
            
            # Add code analysis if available
            if hasattr(self, 'code_analysis'):
                export_data["code_analysis"] = self.code_analysis
            
            # Export timing data if available
            timing_data = self.extract_instruction_timing()
            if "error" not in timing_data:
                export_data["timing_analysis"] = timing_data
            
            # Export based on format
            if format == 'json':
                with open(filename, 'w') as f:
                    json.dump(export_data, f, indent=2)
                    
            elif format == 'csv':
                # Create a flattened structure for CSV export
                flattened = self._flatten_dict(export_data)
                
                # Convert to DataFrame and save
                df = pd.DataFrame([flattened])
                df.to_csv(filename, index=False)
                
            elif format == 'text':
                with open(filename, 'w') as f:
                    f.write(f"ROM Analysis: {self.rom_format.upper()}\n")
                    f.write("="*50 + "\n\n")
                    
                    # Write header data
                    f.write("Header Information:\n")
                    for key, value in self.header_data.items():
                        f.write(f"  {key}: {value}\n")
                    f.write("\n")
                    
                    # Write code analysis if available
                    if hasattr(self, 'code_analysis'):
                        f.write("Code Analysis:\n")
                        for key, value in self.code_analysis.items():
                            if key == 'common_opcodes':
                                f.write("  Common opcodes:\n")
                                for opcode, count in value:
                                    f.write(f"    0x{opcode:02X}: {count} occurrences\n")
                            else:
                                f.write(f"  {key}: {value}\n")
                        f.write("\n")
                    
                    # Write timing analysis if available
                    if "error" not in timing_data:
                        f.write("Timing Analysis:\n")
                        for key, value in timing_data.items():
                            if key == 'common_sequences':
                                f.write("  Common instruction sequences:\n")
                                for seq in value:
                                    opcodes_hex = [f"0x{op:02X}" for op in seq['opcodes']]
                                    f.write(f"    Sequence: {opcodes_hex}\n")
                                    f.write(f"      Occurrences: {seq['occurrences']}\n")
                                    f.write(f"      Total cycles: {seq['total_cycles']}\n")
                            else:
                                f.write(f"  {key}: {value}\n")
            else:
                logger.error(f"Unsupported export format: {format}")
                return False
                
            logger.info(f"Analysis exported to {filename} in {format} format")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting analysis: {e}")
            if self.debug_mode:
                import traceback
                traceback.print_exc()
            return False
    
    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = '', sep: str = '_') -> Dict[str, Any]:
        """
        Flatten a nested dictionary for CSV export.
        
        Args:
            d: Dictionary to flatten
            parent_key: Parent key for nested items
            sep: Separator for keys
            
        Returns:
            Flattened dictionary
        """
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep).items())
            elif isinstance(v, list):
                # For lists, store length and first few items
                items.append((f"{new_key}_length", len(v)))
                for i, item in enumerate(v[:3]):  # Store first 3 items
                    items.append((f"{new_key}_{i}", str(item)))
            else:
                items.append((new_key, v))
                
        return dict(items)