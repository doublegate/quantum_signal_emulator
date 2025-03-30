"""
Atari 2600 system implementation.

This module provides a complete system implementation for the Atari 2600,
integrating the CPU, TIA, RIOT, and memory components. It handles system
initialization, ROM loading, and emulation cycle management with cycle-precise
timing to ensure accurate hardware emulation.
"""

import logging
import os
import time
from typing import Dict, List, Optional, Any

from ...common.interfaces import System
from .cpu import CPU6507
from .tia import AtariTIA
from .riot import AtariRIOT
from .memory import AtariMemory
from .cartridge import CartridgeFactory, detect_cartridge_type

logger = logging.getLogger("QuantumSignalEmulator.Atari2600.System")

class AtariSystem(System):
    """
    Complete Atari 2600 system emulation.
    
    Integrates all Atari 2600 components (CPU, TIA, RIOT, memory) into a
    complete system. Manages system timing, ROM loading, and component
    interaction with cycle-precise accuracy.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Atari 2600 system.
        
        Args:
            config: System configuration dictionary
        """
        self.config = config
        
        # Create and connect components
        self.memory = AtariMemory(config)
        self.cpu = CPU6507()
        self.tia = AtariTIA()
        self.riot = AtariRIOT()
        
        # Connect components
        self.cpu.set_memory(self.memory)
        self.memory.connect_tia(self.tia)
        self.memory.connect_riot(self.riot)
        
        # System state
        self.cycle_count = 0
        self.frame_count = 0
        self.tv_format = "NTSC"  # NTSC or PAL
        self.state_history = []
        
        # Timing
        self.frame_cycles = 0
        self.last_frame_time = time.time()
        self.running = False
        
        # ROM information
        self.rom_loaded = False
        self.rom_name = ""
        self.cart_type = ""
        
        logger.info("Atari 2600 system initialized")
    
    def load_rom(self, rom_path: str) -> None:
        """
        Load an Atari 2600 ROM file.
        
        Args:
            rom_path: Path to ROM file
        """
        try:
            with open(rom_path, 'rb') as f:
                rom_data = f.read()
                
            # Get ROM name from filename
            self.rom_name = os.path.basename(rom_path)
            
            # Detect cartridge type
            self.cart_type = detect_cartridge_type(rom_data)
            
            # Load ROM into memory
            self.memory.load_rom(rom_data)
            self.rom_loaded = True
            
            logger.info(f"Loaded ROM: {self.rom_name}, type: {self.cart_type}")
            
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
        self.tia.reset() if hasattr(self.tia, 'reset') else None
        self.riot.reset() if hasattr(self.riot, 'reset') else None
        
        # Reset system state
        self.cycle_count = 0
        self.frame_count = 0
        self.frame_cycles = 0
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
        self.frame_cycles = 0
        
        # Run until a frame is completed
        while not frame_completed:
            # Execute one CPU instruction
            try:
                # Check if CPU is halted by WSYNC
                if hasattr(self.tia, 'wsync_halt') and self.tia.wsync_halt:
                    # CPU is halted, just run TIA until WSYNC is cleared
                    cycles_to_run = 1
                    self.tia.step(cycles_to_run)
                    self.riot.step(cycles_to_run)
                    self.cycle_count += cycles_to_run
                    self.frame_cycles += cycles_to_run
                else:
                    # Normal CPU execution
                    cpu_cycles = self.cpu.step()
                    
                    # Run TIA and RIOT for the same number of cycles
                    frame_completed = self.tia.step(cpu_cycles)
                    self.riot.step(cpu_cycles)
                    
                    # Update cycle counts
                    self.cycle_count += cpu_cycles
                    self.frame_cycles += cpu_cycles
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
            "tv_format": self.tv_format,
            "rom_name": self.rom_name,
            "cart_type": self.cart_type,
            "cpu_state": self.cpu.get_state(),
            "tia_state": self.tia.get_state(),
            "riot_state": self.riot.get_state() if hasattr(self.riot, 'get_state') else {},
            "frame_buffer": self.tia.get_frame_buffer()
        }
    
    def _record_state(self) -> None:
        """Record the current system state for analysis."""
        # Create a simplified state snapshot for analysis
        state = {
            "cycle": self.cycle_count,
            "scanline": self.tia.scanline if hasattr(self.tia, 'scanline') else 0,
            "dot": self.tia.cycle if hasattr(self.tia, 'cycle') else 0,
            "registers": {
                # CPU registers
                "A": self.cpu.A,
                "X": self.cpu.X,
                "Y": self.cpu.Y,
                "PC": self.cpu.PC,
                "SP": self.cpu.SP,
                "P": self.cpu.P,
                # TIA registers if available
                "VSYNC": getattr(self.tia, 'registers', {}).get(self.tia.VSYNC, 0) if hasattr(self.tia, 'VSYNC') else 0,
                "VBLANK": getattr(self.tia, 'registers', {}).get(self.tia.VBLANK, 0) if hasattr(self.tia, 'VBLANK') else 0,
                "COLUBK": getattr(self.tia, 'registers', {}).get(self.tia.COLUBK, 0) if hasattr(self.tia, 'COLUBK') else 0,
                # RIOT registers if available
                "SWCHA": getattr(self.riot, 'registers', {}).get(self.riot.SWCHA, 0) if hasattr(self.riot, 'SWCHA') else 0,
                "INTIM": getattr(self.riot, 'registers', {}).get(self.riot.INTIM, 0) if hasattr(self.riot, 'INTIM') else 0
            }
        }
        
        self.state_history.append(state)
        
        # Keep history to a reasonable size
        if len(self.state_history) > 100000:
            self.state_history = self.state_history[-50000:]
    
    def set_joystick_state(self, player: int, state: Dict[str, bool]) -> None:
        """
        Set joystick state for a player.
        
        Args:
            player: Player number (0 or 1)
            state: Dictionary with joystick state
        """
        if hasattr(self.riot, 'set_joystick_state'):
            self.riot.set_joystick_state(player, state)
    
    def set_console_switches(self, switches: Dict[str, bool]) -> None:
        """
        Set console switch state.
        
        Args:
            switches: Dictionary with switch state
        """
        if hasattr(self.riot, 'set_console_switches'):
            self.riot.set_console_switches(switches)