"""
Sega Genesis / Mega Drive system implementation.

This module provides a complete system implementation for the Sega Genesis/Mega Drive,
integrating the 68000 CPU, Z80 CPU, VDP, FM sound, and memory components. It handles
system initialization, ROM loading, and emulation cycle management with cycle-precise
timing to ensure accurate hardware emulation.
"""

import logging
import os
import time
from typing import Dict, List, Optional, Any

from ...common.interfaces import System
from .m68k_cpu import M68KCPU
from .z80_cpu import Z80CPU
from .vdp import GenesisVDP
from .memory import GenesisMemory
from .fm_sound import GenesisFM

logger = logging.getLogger("QuantumSignalEmulator.Genesis.System")

class GenesisSystem(System):
    """
    Complete Genesis/Mega Drive system emulation.
    
    Integrates all Genesis components (M68K CPU, Z80 CPU, VDP, FM sound, memory) 
    into a complete system. Manages system timing, ROM loading, and component 
    interaction with cycle-precise accuracy.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Genesis system.
        
        Args:
            config: System configuration dictionary
        """
        self.config = config
        
        # Create and connect components
        self.memory = GenesisMemory(config)
        self.m68k_cpu = M68KCPU()
        self.z80_cpu = Z80CPU()
        self.vdp = GenesisVDP()
        self.fm_sound = GenesisFM()
        
        # Connect components
        self.m68k_cpu.set_memory(self.memory)
        self.z80_cpu.set_memory(self.memory)
        self.memory.connect_vdp(self.vdp)
        self.memory.connect_fm_sound(self.fm_sound)
        
        # System state
        self.cycle_count = 0
        self.frame_count = 0
        self.state_history = []
        
        # Timing
        self.frame_cycles = 0
        self.last_frame_time = time.time()
        self.running = False
        
        # ROM information
        self.rom_loaded = False
        self.rom_name = ""
        
        # Z80 state
        self.z80_active = False
        
        logger.info("Genesis system initialized")
    
    def load_rom(self, rom_path: str) -> None:
        """
        Load a Genesis ROM file.
        
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
        self.m68k_cpu.reset()
        self.z80_cpu.reset()
        self.memory.reset()
        self.vdp.reset()
        self.fm_sound.reset()
        
        # Reset system state
        self.cycle_count = 0
        self.frame_count = 0
        self.frame_cycles = 0
        self.state_history = []
        self.last_frame_time = time.time()
        
        # Z80 starts in reset state
        self.z80_active = False
        
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
        self.frame_cycles = 0
        
        # Run until a frame is completed
        while not frame_completed:
            # Execute one M68K CPU instruction
            try:
                # Run 68000 CPU
                m68k_cycles = self.m68k_cpu.step()
                
                # Run Z80 CPU if active
                if self.z80_active:
                    # Z80 runs at 1/2 the clock rate of the 68000
                    z80_cycles = m68k_cycles // 2
                    for _ in range(z80_cycles):
                        self.z80_cpu.step()
                
                # Run VDP
                frame_completed = self.vdp.step(m68k_cycles)
                
                # Run FM sound
                self.fm_sound.step(m68k_cycles)
                
                # Update cycle counts
                self.cycle_count += m68k_cycles
                self.frame_cycles += m68k_cycles
            except Exception as e:
                logger.error(f"Error during emulation: {e}")
                break
                
            # Record system state periodically
            if self.frame_cycles % 100 == 0:
                self._record_state()
        
        # Update frame count
        self.frame_count += 1
        
        # Calculate frame rate
        current_time = time.time()
        frame_time = current_time - self.last_frame_time
        fps = 1.0 / frame_time if frame_time > 0 else 0
        self.last_frame_time = current_time
        
        logger.debug(f"Frame {self.frame_count} completed: {self.frame_cycles} cycles, {fps:.2f} FPS")
        
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
            "m68k_state": self.m68k_cpu.get_state(),
            "z80_state": self.z80_cpu.get_state(),
            "vdp_state": self.vdp.get_state(),
            "fm_state": self.fm_sound.get_state(),
            "frame_buffer": self.vdp.get_frame_buffer()
        }
    
    def _record_state(self) -> None:
        """Record the current system state for analysis."""
        # Create a simplified state snapshot for analysis
        state = {
            "cycle": self.cycle_count,
            "scanline": self.vdp.scanline if hasattr(self.vdp, 'scanline') else 0,
            "dot": self.vdp.dot if hasattr(self.vdp, 'dot') else 0,
            "registers": {
                # M68K CPU registers
                "D0": self.m68k_cpu.d_regs[0],
                "D1": self.m68k_cpu.d_regs[1],
                "A0": self.m68k_cpu.a_regs[0],
                "A1": self.m68k_cpu.a_regs[1],
                "PC": self.m68k_cpu.pc,
                "SR": self.m68k_cpu.sr,
                
                # Z80 CPU registers if active
                "Z80_A": self.z80_cpu.a if self.z80_active else 0,
                "Z80_F": self.z80_cpu.f if self.z80_active else 0,
                "Z80_PC": self.z80_cpu.pc if self.z80_active else 0,
                
                # VDP registers
                "VDP_MODE1": self.vdp.registers[0] if hasattr(self.vdp, 'registers') else 0,
                "VDP_MODE2": self.vdp.registers[1] if hasattr(self.vdp, 'registers') else 0,
                "VDP_HINT": self.vdp.registers[10] if hasattr(self.vdp, 'registers') else 0
            }
        }
        
        self.state_history.append(state)
        
        # Keep history to a reasonable size
        if len(self.state_history) > 100000:
            self.state_history = self.state_history[-50000:]
    
    def set_z80_active(self, active: bool) -> None:
        """
        Set Z80 activity state.
        
        Args:
            active: True to enable Z80, False to disable
        """
        self.z80_active = active
        logger.debug(f"Z80 state set to {'active' if active else 'inactive'}")