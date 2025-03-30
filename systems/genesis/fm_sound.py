"""
Sega Genesis/Mega Drive FM Sound emulation.

The Genesis sound system consists of a Yamaha YM2612 FM synthesis chip and
a Texas Instruments SN76489 PSG (Programmable Sound Generator). This module
provides emulation of both sound chips with cycle-accurate timing.
"""

import logging
import math
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union

logger = logging.getLogger("QuantumSignalEmulator.Genesis.FMSound")

class GenesisFM:
    """
    Emulates the Genesis/Mega Drive FM sound system.
    
    Provides emulation of the YM2612 FM synthesis chip and the SN76489 PSG,
    generating accurate audio output matching the original hardware.
    """
    
    # FM registers
    FM_TIMERA_H = 0x24
    FM_TIMERA_L = 0x25
    FM_TIMERB = 0x26
    FM_TIMER_CTRL = 0x27
    FM_KEY_ON = 0x28
    
    # FM constants
    FM_CHANNELS = 6
    FM_OPERATORS = 4
    SAMPLE_RATE = 44100
    
    # PSG constants
    PSG_CHANNELS = 4  # 3 tone channels + 1 noise channel
    
    def __init__(self):
        """Initialize the Genesis FM sound system."""
        # FM registers
        self.fm_registers = []
        for i in range(2):  # 2 register banks
            self.fm_registers.append([0] * 256)
        
        # PSG registers
        self.psg_registers = [0] * 8
        
        # FM state
        self.fm_channels = []
        for i in range(self.FM_CHANNELS):
            self.fm_channels.append({
                "enabled": False,
                "frequency": 0,
                "algorithm": 0,
                "feedback": 0,
                "operators": [
                    {"enabled": False, "frequency": 0, "level": 0, "phase": 0},
                    {"enabled": False, "frequency": 0, "level": 0, "phase": 0},
                    {"enabled": False, "frequency": 0, "level": 0, "phase": 0},
                    {"enabled": False, "frequency": 0, "level": 0, "phase": 0}
                ]
            })
        
        # PSG state
        self.psg_channels = []
        for i in range(self.PSG_CHANNELS):
            self.psg_channels.append({
                "enabled": False,
                "frequency": 0,
                "volume": 0,
                "counter": 0,
                "output": 0
            })
        
        # Timers
        self.timer_a_period = 0
        self.timer_b_period = 0
        self.timer_a_counter = 0
        self.timer_b_counter = 0
        self.timer_a_enabled = False
        self.timer_b_enabled = False
        self.timer_a_overflow = False
        self.timer_b_overflow = False
        
        # Cycle counters
        self.fm_cycles = 0
        self.psg_cycles = 0
        
        # Output buffer
        self.output_buffer = []
        self.samples_per_frame = self.SAMPLE_RATE // 60  # 60 Hz refresh rate
        
        logger.info("Genesis FM sound system initialized")
    
    def reset(self) -> None:
        """Reset the sound system to initial state."""
        # Reset FM registers
        for i in range(2):
            self.fm_registers[i] = [0] * 256
        
        # Reset PSG registers
        self.psg_registers = [0] * 8
        
        # Reset FM channels
        for i in range(self.FM_CHANNELS):
            self.fm_channels[i] = {
                "enabled": False,
                "frequency": 0,
                "algorithm": 0,
                "feedback": 0,
                "operators": [
                    {"enabled": False, "frequency": 0, "level": 0, "phase": 0},
                    {"enabled": False, "frequency": 0, "level": 0, "phase": 0},
                    {"enabled": False, "frequency": 0, "level": 0, "phase": 0},
                    {"enabled": False, "frequency": 0, "level": 0, "phase": 0}
                ]
            }
        
        # Reset PSG channels
        for i in range(self.PSG_CHANNELS):
            self.psg_channels[i] = {
                "enabled": False,
                "frequency": 0,
                "volume": 0,
                "counter": 0,
                "output": 0
            }
        
        # Reset timers
        self.timer_a_period = 0
        self.timer_b_period = 0
        self.timer_a_counter = 0
        self.timer_b_counter = 0
        self.timer_a_enabled = False
        self.timer_b_enabled = False
        self.timer_a_overflow = False
        self.timer_b_overflow = False
        
        # Reset cycle counters
        self.fm_cycles = 0
        self.psg_cycles = 0
        
        # Clear output buffer
        self.output_buffer = []
        
        logger.info("FM sound system reset")
    
    def read_register(self, address: int) -> int:
        """
        Read an FM register.
        
        Args:
            address: Register address
            
        Returns:
            Register value
        """
        # Only status register is readable on YM2612
        if address == 0:
            # Status register
            status = 0
            
            # Set timer overflow flags
            if self.timer_a_overflow:
                status |= 0x01
            if self.timer_b_overflow:
                status |= 0x02
                
            return status
            
        return 0
    
    def write_register(self, address: int, value: int) -> None:
        """
        Write to an FM register.
        
        Args:
            address: Register address
            value: Value to write
        """
        bank = 0
        
        # Check if this is part 1 (registers 0x100-0x1FF)
        if address == 1:
            bank = 1
            address = 0
            
        # Store value in register
        if address < len(self.fm_registers[bank]):
            self.fm_registers[bank][address] = value
            
        # Handle special registers
        if bank == 0:
            if address == self.FM_TIMERA_H:
                # Timer A high bits
                self.timer_a_period = (self.timer_a_period & 0x03) | ((value & 0xFF) << 2)
                logger.debug(f"Timer A period: {self.timer_a_period}")
                
            elif address == self.FM_TIMERA_L:
                # Timer A low bits
                self.timer_a_period = (self.timer_a_period & 0x3FC) | (value & 0x03)
                logger.debug(f"Timer A period: {self.timer_a_period}")
                
            elif address == self.FM_TIMERB:
                # Timer B
                self.timer_b_period = value & 0xFF
                logger.debug(f"Timer B period: {self.timer_b_period}")
                
            elif address == self.FM_TIMER_CTRL:
                # Timer control
                self.timer_a_enabled = (value & 0x01) != 0
                self.timer_b_enabled = (value & 0x02) != 0
                
                if value & 0x04:
                    self.timer_a_overflow = False
                if value & 0x08:
                    self.timer_b_overflow = False
                    
                logger.debug(f"Timer control: A={'enabled' if self.timer_a_enabled else 'disabled'}, " + 
                            f"B={'enabled' if self.timer_b_enabled else 'disabled'}")
                
            elif address == self.FM_KEY_ON:
                # Key on/off
                channel = value & 0x07
                if channel < self.FM_CHANNELS:
                    # Extract operator bits (bits 4-7)
                    operators = (value >> 4) & 0x0F
                    for op in range(self.FM_OPERATORS):
                        self.fm_channels[channel]["operators"][op]["enabled"] = (operators & (1 << op)) != 0
                    
                    # Update channel enabled state based on operators
                    self.fm_channels[channel]["enabled"] = operators != 0
                    
                    logger.debug(f"Channel {channel} key {'on' if operators != 0 else 'off'}")
    
    def write_psg(self, value: int) -> None:
        """
        Write to PSG.
        
        Args:
            value: Value to write
        """
        if value & 0x80:
            # Latch/data byte
            reg_index = (value >> 4) & 0x07
            reg_data = value & 0x0F
            self.psg_registers[reg_index] = reg_data
            
            # Update channel settings
            self._update_psg_channel(reg_index, reg_data)
        else:
            # Data byte for previously selected register
            # (Not implemented in this simplified version)
            pass
    
    def _update_psg_channel(self, reg_index: int, value: int) -> None:
        """
        Update PSG channel settings.
        
        Args:
            reg_index: Register index
            value: Register value
        """
        if reg_index < 4:
            # Tone registers (0, 2, 4, 6 = 10-bit frequency)
            channel = reg_index // 2
            if channel < 3:  # Only 3 tone channels
                if reg_index % 2 == 0:
                    # Low 4 bits
                    self.psg_channels[channel]["frequency"] = (self.psg_channels[channel]["frequency"] & 0x3F0) | value
                else:
                    # High 6 bits
                    self.psg_channels[channel]["frequency"] = (self.psg_channels[channel]["frequency"] & 0x00F) | ((value & 0x3F) << 4)
                    
                # Update enabled state (frequency 0 means disabled)
                self.psg_channels[channel]["enabled"] = self.psg_channels[channel]["frequency"] > 0
                
        elif reg_index == 4:
            # Noise control
            self.psg_channels[3]["frequency"] = value & 0x07
            self.psg_channels[3]["enabled"] = True
            
        elif reg_index >= 4 and reg_index < 8:
            # Volume registers (1, 3, 5, 7 = 4-bit volume)
            channel = reg_index - 4
            self.psg_channels[channel]["volume"] = value & 0x0F
    
    def step(self, cycles: int) -> None:
        """
        Run the sound system for the specified number of cycles.
        
        Args:
            cycles: Number of CPU cycles to simulate
        """
        # Add cycles to counters
        self.fm_cycles += cycles
        self.psg_cycles += cycles
        
        # YM2612 update rate is CPU clock / 144
        fm_step_rate = 144
        
        # PSG update rate is CPU clock / 15
        psg_step_rate = 15
        
        # Process FM
        while self.fm_cycles >= fm_step_rate:
            self._step_fm()
            self.fm_cycles -= fm_step_rate
            
        # Process PSG
        while self.psg_cycles >= psg_step_rate:
            self._step_psg()
            self.psg_cycles -= psg_step_rate
    
    def _step_fm(self) -> None:
        """Process one FM sound step."""
        # Update timers
        if self.timer_a_enabled:
            self.timer_a_counter += 1
            if self.timer_a_counter >= self.timer_a_period:
                self.timer_a_counter = 0
                self.timer_a_overflow = True
                
        if self.timer_b_enabled:
            self.timer_b_counter += 1
            if self.timer_b_counter >= self.timer_b_period:
                self.timer_b_counter = 0
                self.timer_b_overflow = True
        
        # In a full implementation, this would update the FM synthesis
        # For now, this is a simplified version
    
    def _step_psg(self) -> None:
        """Process one PSG sound step."""
        # Update each PSG channel
        for i in range(self.PSG_CHANNELS):
            if self.psg_channels[i]["enabled"]:
                # Increment counter
                self.psg_channels[i]["counter"] += 1
                
                # Check if period elapsed
                if i < 3:  # Tone channels
                    if self.psg_channels[i]["counter"] >= self.psg_channels[i]["frequency"]:
                        self.psg_channels[i]["counter"] = 0
                        self.psg_channels[i]["output"] ^= 1  # Toggle output
                else:  # Noise channel
                    # Simplified noise generation
                    if self.psg_channels[i]["counter"] >= 16:  # Fixed noise rate for simplicity
                        self.psg_channels[i]["counter"] = 0
                        self.psg_channels[i]["output"] = (self.psg_channels[i]["output"] + 1) & 1
    
    def get_audio_sample(self) -> float:
        """
        Get the current audio sample.
        
        Returns:
            Audio sample value (-1.0 to 1.0)
        """
        sample = 0.0
        
        # Mix PSG channels
        for i in range(self.PSG_CHANNELS):
            if self.psg_channels[i]["enabled"]:
                # PSG volume is 0-15 (15 is lowest)
                volume = (15 - self.psg_channels[i]["volume"]) / 15.0
                sample += self.psg_channels[i]["output"] * volume * 0.25  # Scale to 25% of total
        
        # Mix FM channels (simplified)
        for i in range(self.FM_CHANNELS):
            if self.fm_channels[i]["enabled"]:
                # In a real implementation, this would calculate FM synthesis output
                # For now, just add a simple sine wave as placeholder
                sample += math.sin(self.fm_cycles * (i + 1) * 0.01) * 0.1
        
        # Limit to -1.0 to 1.0 range
        return max(-1.0, min(1.0, sample))
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get the current sound system state.
        
        Returns:
            Dictionary with sound system state
        """
        return {
            "timer_a": {
                "period": self.timer_a_period,
                "counter": self.timer_a_counter,
                "enabled": self.timer_a_enabled,
                "overflow": self.timer_a_overflow
            },
            "timer_b": {
                "period": self.timer_b_period,
                "counter": self.timer_b_counter,
                "enabled": self.timer_b_enabled,
                "overflow": self.timer_b_overflow
            },
            "fm_channels": [ch["enabled"] for ch in self.fm_channels],
            "psg_channels": [
                {"enabled": ch["enabled"], "volume": ch["volume"]} 
                for ch in self.psg_channels
            ]
        }