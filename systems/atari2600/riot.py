"""
Atari 2600 RIOT (RAM, I/O, Timer) chip emulation.

The RIOT chip (6532) handles RAM, timers and I/O ports in the Atari 2600.
It contains:
- 128 bytes of RAM (handled by the memory component)
- Two 8-bit I/O ports (for controller input)
- A timer with programmable intervals
- System-level interrupts (though not used in the Atari 2600)

This module implements the timer functions and I/O functions of the RIOT chip.
"""

import logging
import time
from typing import Dict, Any, Optional

logger = logging.getLogger("QuantumSignalEmulator.Atari2600.RIOT")

class AtariRIOT:
    """
    Emulates the 6532 RIOT chip used in the Atari 2600.
    
    Provides timer functionality and I/O port access for controllers and
    switches. The Atari 2600 uses the RIOT's timer for game timing and
    the I/O ports for reading joysticks, paddles, and console switches.
    """
    
    # RIOT register addresses
    SWCHA   = 0x00  # Port A I/O (joystick input)
    SWACNT  = 0x01  # Port A I/O control
    SWCHB   = 0x02  # Port B I/O (console switches)
    SWBCNT  = 0x03  # Port B I/O control
    INTIM   = 0x04  # Timer output
    INSTAT  = 0x05  # Interrupt status
    TIM1T   = 0x14  # Set 1-cycle timer
    TIM8T   = 0x15  # Set 8-cycle timer
    TIM64T  = 0x16  # Set 64-cycle timer
    T1024T  = 0x17  # Set 1024-cycle timer
    
    # Joystick bit masks
    JOY_UP    = 0x10
    JOY_DOWN  = 0x20
    JOY_LEFT  = 0x40
    JOY_RIGHT = 0x80
    JOY_FIRE  = 0x80  # Joystick fire button (INPT4/5)
    
    # Console switch bit masks
    SWITCH_RESET   = 0x01
    SWITCH_SELECT  = 0x02
    SWITCH_COLOR   = 0x08  # 0 = color, 1 = b&w
    SWITCH_PLAYER0 = 0x10  # Difficulty switch P0
    SWITCH_PLAYER1 = 0x20  # Difficulty switch P1
    
    def __init__(self):
        """Initialize the RIOT chip."""
        # Register values
        self.registers = {
            self.SWCHA: 0xFF,   # All input lines pulled high
            self.SWACNT: 0x00,  # All pins set as input
            self.SWCHB: 0xFF,   # All switches in default position
            self.SWBCNT: 0x00,  # All pins set as input
            self.INTIM: 0x00,   # Timer value
            self.INSTAT: 0x00,  # Interrupt status
        }
        
        # Controller state
        self.joystick_p0 = {
            "up": False,
            "down": False,
            "left": False,
            "right": False,
            "fire": False
        }
        
        self.joystick_p1 = {
            "up": False,
            "down": False,
            "left": False,
            "right": False,
            "fire": False
        }
        
        # Timer state
        self.timer_value = 0
        self.timer_interval = 0
        self.timer_division = 1
        self.timer_running = False
        self.timer_last_cycle = 0
        
        # Input register for fire buttons (INPT4/5)
        self.input_registers = {
            0x04: 0x80,  # INPT4 - Player 0 fire button (active low)
            0x05: 0x80   # INPT5 - Player 1 fire button (active low)
        }
        
        logger.info("RIOT initialized")
    
    def read_register(self, address: int) -> int:
        """
        Read a RIOT register.
        
        Args:
            address: Register address within RIOT (0x00-0x1F)
            
        Returns:
            Register value
        """
        # Map actual address to RIOT register
        addr = address & 0x1F
        
        # Handle different register reads
        if addr == self.SWCHA:
            # Update joystick values before returning
            self._update_joysticks()
            return self.registers[self.SWCHA]
            
        elif addr == self.SWCHB:
            # Console switches (RESET, SELECT, etc.)
            return self.registers[self.SWCHB]
            
        elif addr == self.INTIM:
            # Timer value
            self._update_timer()
            return self.timer_value & 0xFF
            
        elif addr == self.INSTAT:
            # Timer status register
            self._update_timer()
            
            # Reading INSTAT clears the interrupt flag but keeps timer status
            status = self.registers[self.INSTAT]
            self.registers[self.INSTAT] &= 0x7F  # Clear bit 7
            return status
            
        elif addr >= 0x04 and addr <= 0x07:
            # Handle joystick trigger inputs (INPT4-INPT7)
            if addr == 0x04:  # INPT4 - Player 0 fire button
                return 0x80 if not self.joystick_p0["fire"] else 0x00
            elif addr == 0x05:  # INPT5 - Player 1 fire button
                return 0x80 if not self.joystick_p1["fire"] else 0x00
                
        # Default for other registers
        if addr in self.registers:
            return self.registers[addr]
            
        return 0
    
    def write_register(self, address: int, value: int) -> None:
        """
        Write to a RIOT register.
        
        Args:
            address: Register address within RIOT (0x00-0x1F)
            value: Value to write
        """
        # Map actual address to RIOT register
        addr = address & 0x1F
        
        # Handle different register writes
        if addr == self.SWCHA:
            # Port A data register (only bits set as outputs can be written)
            mask = self.registers[self.SWACNT]  # 1 = output, 0 = input
            self.registers[self.SWCHA] = (self.registers[self.SWCHA] & ~mask) | (value & mask)
            
        elif addr == self.SWACNT:
            # Port A data direction register
            self.registers[self.SWACNT] = value
            
        elif addr == self.SWCHB:
            # Port B data register (only bits set as outputs can be written)
            mask = self.registers[self.SWBCNT]  # 1 = output, 0 = input
            self.registers[self.SWCHB] = (self.registers[self.SWCHB] & ~mask) | (value & mask)
            
        elif addr == self.SWBCNT:
            # Port B data direction register
            self.registers[self.SWBCNT] = value
            
        elif addr == self.TIM1T:
            # Set 1-cycle interval timer
            self.timer_value = value
            self.timer_interval = value
            self.timer_division = 1
            self.timer_running = True
            self.registers[self.INSTAT] &= 0x7F  # Clear overflow flag
            
        elif addr == self.TIM8T:
            # Set 8-cycle interval timer
            self.timer_value = value
            self.timer_interval = value
            self.timer_division = 8
            self.timer_running = True
            self.registers[self.INSTAT] &= 0x7F  # Clear overflow flag
            
        elif addr == self.TIM64T:
            # Set 64-cycle interval timer
            self.timer_value = value
            self.timer_interval = value
            self.timer_division = 64
            self.timer_running = True
            self.registers[self.INSTAT] &= 0x7F  # Clear overflow flag
            
        elif addr == self.T1024T:
            # Set 1024-cycle interval timer
            self.timer_value = value
            self.timer_interval = value
            self.timer_division = 1024
            self.timer_running = True
            self.registers[self.INSTAT] &= 0x7F  # Clear overflow flag
            
        # Store in register map
        if addr in self.registers:
            self.registers[addr] = value
    
    def _update_timer(self) -> None:
        """Update the timer based on elapsed cycles."""
        if not self.timer_running:
            return
            
        # Update timer based on cycles elapsed
        elapsed_cycles = self._get_elapsed_cycles()
        
        if elapsed_cycles > 0:
            # Adjust timer value based on elapsed cycles
            decrements = elapsed_cycles // self.timer_division
            
            if decrements > 0:
                # Record the time
                self.timer_last_cycle += decrements * self.timer_division
                
                # Decrement the timer
                if decrements >= self.timer_value:
                    # Timer has expired
                    self.timer_value = 0
                    self.timer_running = False
                    self.registers[self.INSTAT] |= 0xC0  # Set bits 6 and 7
                else:
                    self.timer_value -= decrements
    
    def _get_elapsed_cycles(self) -> int:
        """
        Get the number of cycles elapsed since last timer update.
        
        Returns:
            Number of cycles elapsed
        """
        # In a real implementation, we would use CPU cycles
        # This is a simplified version that uses system time
        current_time = time.time() * 1000000  # Microseconds
        elapsed = current_time - self.timer_last_cycle
        
        # Convert to Atari 2600 cycles (1.19 MHz)
        return int(elapsed * 1.19)
    
    def _update_joysticks(self) -> None:
        """Update joystick input registers."""
        # Start with all input lines pulled high
        value = 0xFF
        
        # Apply joystick 0 inputs (bits 4-7)
        if self.joystick_p0["up"]:
            value &= ~self.JOY_UP
        if self.joystick_p0["down"]:
            value &= ~self.JOY_DOWN
        if self.joystick_p0["left"]:
            value &= ~self.JOY_LEFT
        if self.joystick_p0["right"]:
            value &= ~self.JOY_RIGHT
            
        # Apply joystick 1 inputs (bits 0-3)
        if self.joystick_p1["up"]:
            value &= ~(self.JOY_UP >> 4)
        if self.joystick_p1["down"]:
            value &= ~(self.JOY_DOWN >> 4)
        if self.joystick_p1["left"]:
            value &= ~(self.JOY_LEFT >> 4)
        if self.joystick_p1["right"]:
            value &= ~(self.JOY_RIGHT >> 4)
            
        # Update register with input-only bits preserved
        mask = ~self.registers[self.SWACNT]  # 0 = input, 1 = output
        self.registers[self.SWCHA] = (self.registers[self.SWCHA] & ~mask) | (value & mask)
        
        # Update fire button registers
        self.input_registers[0x04] = 0x00 if self.joystick_p0["fire"] else 0x80
        self.input_registers[0x05] = 0x00 if self.joystick_p1["fire"] else 0x80
    
    def set_joystick_state(self, player: int, state: Dict[str, bool]) -> None:
        """
        Set joystick state for a player.
        
        Args:
            player: Player number (0 or 1)
            state: Dictionary with joystick state
        """
        if player == 0:
            joystick = self.joystick_p0
        else:
            joystick = self.joystick_p1
            
        # Update joystick state
        for key, value in state.items():
            if key in joystick:
                joystick[key] = value
                
        logger.debug(f"Joystick P{player} state: {joystick}")
    
    def set_console_switches(self, switches: Dict[str, bool]) -> None:
        """
        Set console switch state.
        
        Args:
            switches: Dictionary with switch state
        """
        # Start with default value
        value = 0xFF
        
        # Apply switch values
        if switches.get("reset", False):
            value &= ~self.SWITCH_RESET
        if switches.get("select", False):
            value &= ~self.SWITCH_SELECT
        if switches.get("bw", False):
            value &= ~self.SWITCH_COLOR
        if switches.get("p0_diff_b", False):
            value &= ~self.SWITCH_PLAYER0
        if switches.get("p1_diff_b", False):
            value &= ~self.SWITCH_PLAYER1
            
        # Update register with input-only bits preserved
        mask = ~self.registers[self.SWBCNT]  # 0 = input, 1 = output
        self.registers[self.SWCHB] = (self.registers[self.SWCHB] & ~mask) | (value & mask)
        
        logger.debug(f"Console switches: {switches}")
    
    def step(self, cycles: int) -> None:
        """
        Step the RIOT timer for the given number of cycles.
        
        Args:
            cycles: Number of CPU cycles to step
        """
        if not self.timer_running:
            return
            
        # Update the timer
        if self.timer_division == 1:
            # 1-cycle timer
            if cycles >= self.timer_value:
                self.timer_value = 0
                self.timer_running = False
                self.registers[self.INSTAT] |= 0xC0  # Set bits 6 and 7
            else:
                self.timer_value -= cycles
                
        else:
            # Divider timers (8, 64, 1024)
            self.timer_last_cycle += cycles
            decrements = self.timer_last_cycle // self.timer_division
            
            if decrements > 0:
                self.timer_last_cycle %= self.timer_division
                
                if decrements >= self.timer_value:
                    self.timer_value = 0
                    self.timer_running = False
                    self.registers[self.INSTAT] |= 0xC0  # Set bits 6 and 7
                else:
                    self.timer_value -= decrements
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get the RIOT state.
        
        Returns:
            Dictionary with RIOT state
        """
        return {
            "registers": dict(self.registers),
            "joystick_p0": dict(self.joystick_p0),
            "joystick_p1": dict(self.joystick_p1),
            "timer_value": self.timer_value,
            "timer_interval": self.timer_interval,
            "timer_division": self.timer_division,
            "timer_running": self.timer_running
        }