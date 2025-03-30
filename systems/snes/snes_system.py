"""
SNES (Super Nintendo Entertainment System) system implementation.

This module provides a complete system implementation for the SNES,
integrating the CPU, PPU, APU, and memory components. It handles system
initialization, ROM loading, and emulation cycle management with cycle-precise
timing to ensure accurate hardware emulation.
"""

import logging
import os
import time
from typing import Dict, List, Optional, Any

from ...common.interfaces import System
from .cpu import CPU65C816
from .ppu import SNESPPU
from .memory import SNESMemory
from .dsp import SNESDSP

logger = logging.getLogger("QuantumSignalEmulator.SNES.System")

class SNESSystem(System):
    """
    Complete SNES system emulation.
    
    Integrates all SNES components (CPU, PPU, APU/DSP, memory) into a complete
    system. Manages system timing, ROM loading, and component interaction
    with cycle-precise accuracy.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the SNES system.
        
        Args:
            config: System configuration dictionary
        """
        self.config = config
        
        # Create components
        self.memory = SNESMemory(config)
        self.cpu = CPU65C816()
        self.ppu = SNESPPU(self.memory)
        
        # Create DSP (audio processor)
        if hasattr(self, 'dsp') and self.dsp is None:
            self.dsp = SNESDSP()
        else:
            self.dsp = None
        
        # Connect components
        self.cpu.set_memory(self.memory)
        self.memory.connect_ppu(self.ppu)
        
        if self.dsp:
            self.memory.connect_apu(self.dsp)
        
        # System state
        self.cycle_count = 0
        self.frame_count = 0
        self.state_history = []
        
        # Timing
        self.last_frame_time = time.time()
        self.running = False
        
        # ROM information
        self.rom_loaded = False
        self.rom_name = ""
        
        logger.info("SNES system initialized")
    
    def load_rom(self, rom_path: str) -> None:
        """
        Load a SNES ROM file.
        
        Args:
            rom_path: Path to ROM file
        """
        try:
            with open(rom_path, 'rb') as f:
                rom_data = f.read()
                
            # Get ROM name from filename
            self.rom_name = os.path.basename(rom_path)
            
            # Load ROM into memory
            self.memory.load_rom(rom_data)
            self.rom_loaded = True
            
            logger.info(f"Loaded ROM: {self.rom_name}")
            
        except Exception as e:
            logger.error(f"Failed to load ROM: {e}")
            self.rom_loaded = False
    
    def reset(self) -> None:
        """Reset the system to initial state."""
        if not self.rom_loaded:
            logger.error("Cannot reset: No ROM loaded")
            return
            
        # Reset all components
        self.cpu.reset()
        self.memory.reset()
        
        if hasattr(self.ppu, 'reset'):
            self.ppu.reset()
            
        if self.dsp and hasattr(self.dsp, 'reset'):
            self.dsp.reset()
        
        # Reset system state
        self.cycle_count = 0
        self.frame_count = 0
        self.state_history = []
        self.last_frame_time = time.time()
        
        logger.info("System reset")
    
    def run_frame(self) -> Dict[str, Any]:
        """
        Run the system for one frame.
        
        Returns:
            System state at the end of the frame
        """
        if not self.rom_loaded:
            logger.error("Cannot run: No ROM loaded")
            return {}
            
        frame_completed = False
        frame_cycles = 0
        
        # Run until a frame is completed
        while not frame_completed:
            try:
                # Execute one CPU instruction
                cpu_cycles = self.cpu.step()
                
                # Run PPU
                frame_completed = self.ppu.step(cpu_cycles)
                
                # Run APU/DSP if available
                if self.dsp:
                    self.dsp.step(cpu_cycles)
                
                # Update cycle counts
                self.cycle_count += cpu_cycles
                frame_cycles += cpu_cycles
                
                # Record system state periodically
                if frame_cycles % 100 == 0:
                    self._record_state()
                    
            except Exception as e:
                logger.error(f"Error during emulation: {e}")
                break
        
        # Update frame count
        self.frame_count += 1
        
        # Calculate frame rate
        current_time = time.time()
        frame_time = current_time - self.last_frame_time
        fps = 1.0 / frame_time if frame_time > 0 else 0
        self.last_frame_time = current_time
        
        logger.debug(f"Frame {self.frame_count} completed: {frame_cycles} cycles, {fps:.2f} FPS")
        
        # Return system state
        return self.get_system_state()
    
    def get_system_state(self) -> Dict[str, Any]:
        """
        Get the current system state.
        
        Returns:
            Dictionary with system state
        """
        return {
            "cycle_count": self.cycle_count,
            "frame_count": self.frame_count,
            "rom_name": self.rom_name,
            "cpu_state": self.cpu.get_state(),
            "ppu_state": self.ppu.get_state(),
            "dsp_state": self.dsp.get_state() if self.dsp and hasattr(self.dsp, 'get_state') else {},
            "frame_buffer": self.ppu.get_frame_buffer()
        }
    
    def _record_state(self) -> None:
        """Record the current system state for analysis."""
        # Create a simplified state snapshot for analysis
        state = {
            "cycle": self.cycle_count,
            "scanline": self.ppu.v_counter if hasattr(self.ppu, 'v_counter') else 0,
            "dot": self.ppu.h_counter if hasattr(self.ppu, 'h_counter') else 0,
            "registers": {
                # CPU registers
                "A": self.cpu.A,
                "X": self.cpu.X,
                "Y": self.cpu.Y,
                "SP": self.cpu.SP,
                "PC": self.cpu.PC,
                "P": self.cpu.P,
                "D": self.cpu.D,
                "DB": self.cpu.DB,
                "PBR": self.cpu.PBR,
                
                # PPU registers (add a few key ones)
                "BGMODE": self.ppu.registers.get(self.ppu.BGMODE, 0) if hasattr(self.ppu, 'registers') else 0,
                "INIDISP": self.ppu.registers.get(self.ppu.INIDISP, 0) if hasattr(self.ppu, 'registers') else 0,
                "CGADD": self.ppu.registers.get(self.ppu.CGADD, 0) if hasattr(self.ppu, 'registers') else 0
            }
        }
        
        self.state_history.append(state)
        
        # Keep history to a reasonable size
        if len(self.state_history) > 100000:
            self.state_history = self.state_history[-50000:]