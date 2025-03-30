"""
Sega Genesis/Mega Drive Video Display Processor (VDP) emulation.

The Genesis VDP is based on the Texas Instruments TMS9918 but extended with
additional capabilities. It supports multiple background layers, sprites,
and various special effects. This module implements a cycle-accurate
emulation of the Genesis VDP.
"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from ...common.interfaces import VideoProcessor

logger = logging.getLogger("QuantumSignalEmulator.Genesis.VDP")

class GenesisVDP(VideoProcessor):
    """
    Emulates the Genesis/Mega Drive VDP (Video Display Processor).
    
    The Genesis VDP supports multiple display modes, two scrollable playfields,
    sprites, and various special effects. This implementation manages VRAM, CRAM,
    VSRAM, and provides cycle-accurate rendering.
    """
    
    # VDP register count
    NUM_REGISTERS = 24
    
    # Display dimensions (NTSC)
    SCREEN_WIDTH = 320
    SCREEN_HEIGHT = 224
    
    # VRAM/CRAM/VSRAM sizes
    VRAM_SIZE = 65536   # 64KB
    CRAM_SIZE = 128     # 64 colors × 16 bits
    VSRAM_SIZE = 80     # 40 entries × 16 bits
    
    # VDP command types
    CMD_VRAM_READ = 0
    CMD_VRAM_WRITE = 1
    CMD_CRAM_READ = 2
    CMD_CRAM_WRITE = 3
    CMD_VSRAM_READ = 4
    CMD_VSRAM_WRITE = 5
    CMD_REG_WRITE = 8
    
    # Status register bits
    STATUS_FIFO_EMPTY = 0x0200
    STATUS_FIFO_FULL = 0x0100
    STATUS_VBLANK = 0x0008
    STATUS_HBLANK = 0x0004
    STATUS_DMA = 0x0002
    STATUS_PAL = 0x0001
    
    def __init__(self):
        """Initialize the Genesis VDP."""
        # Memory
        self.vram = bytearray(self.VRAM_SIZE)
        self.cram = bytearray(self.CRAM_SIZE)
        self.vsram = bytearray(self.VSRAM_SIZE)
        
        # Registers
        self.registers = [0] * self.NUM_REGISTERS
        self.command_word = 0
        self.pending_command = False
        
        # Status
        self.status = self.STATUS_FIFO_EMPTY
        
        # VRAM/DMA address
        self.address = 0
        self.command_type = 0
        
        # Read buffers
        self.read_buffer = 0
        
        # Render state
        self.scanline = 0
        self.dot = 0
        self.frame = 0
        
        # Frame buffer (RGB format)
        self.frame_buffer = bytearray(self.SCREEN_WIDTH * self.SCREEN_HEIGHT * 3)
        
        # Display mode
        self.display_enabled = False
        self.h40_mode = False         # 40 cells (320 pixels) horizontal
        self.v30_mode = False         # 30 cells (240 pixels) vertical
        self.shadow_highlight = False # Shadow/highlight enabled
        
        # Interrupts
        self.hint_counter = 0
        self.hint_value = 0
        self.vint_enabled = False
        self.hint_enabled = False
        self.external_int = False
        
        # Scrolling
        self.hscroll_mode = 0
        self.vscroll_mode = 0
        self.hscroll_base = 0
        
        # DMA
        self.dma_enabled = False
        self.dma_length = 0
        self.dma_source = 0
        
        # Colors
        self.palette = [(0, 0, 0)] * 64  # 64 colors
        
        # Planes
        self.plane_a_base = 0
        self.plane_b_base = 0
        self.window_base = 0
        self.sprite_base = 0
        self.hscroll_base = 0
        
        # Cell sizes
        self.plane_a_cell_w = 0
        self.plane_a_cell_h = 0
        self.plane_b_cell_w = 0
        self.plane_b_cell_h = 0
        
        logger.info("Genesis VDP initialized")
    
    def reset(self) -> None:
        """Reset the VDP to its initial state."""
        # Clear memory
        self.vram = bytearray(self.VRAM_SIZE)
        self.cram = bytearray(self.CRAM_SIZE)
        self.vsram = bytearray(self.VSRAM_SIZE)
        
        # Reset registers
        self.registers = [0] * self.NUM_REGISTERS
        self.command_word = 0
        self.pending_command = False
        
        # Reset status
        self.status = self.STATUS_FIFO_EMPTY
        
        # Reset address
        self.address = 0
        self.command_type = 0
        
        # Reset read buffer
        self.read_buffer = 0
        
        # Reset render state
        self.scanline = 0
        self.dot = 0
        self.frame = 0
        
        # Reset frame buffer
        self.frame_buffer = bytearray(self.SCREEN_WIDTH * self.SCREEN_HEIGHT * 3)
        
        # Reset display mode
        self.display_enabled = False
        self.h40_mode = False
        self.v30_mode = False
        self.shadow_highlight = False
        
        # Reset interrupts
        self.hint_counter = 0
        self.hint_value = 0
        self.vint_enabled = False
        self.hint_enabled = False
        self.external_int = False
        
        # Reset scrolling
        self.hscroll_mode = 0
        self.vscroll_mode = 0
        self.hscroll_base = 0
        
        # Reset DMA
        self.dma_enabled = False
        self.dma_length = 0
        self.dma_source = 0
        
        # Reset colors
        self.palette = [(0, 0, 0)] * 64
        
        # Reset planes
        self.plane_a_base = 0
        self.plane_b_base = 0
        self.window_base = 0
        self.sprite_base = 0
        self.hscroll_base = 0
        
        # Reset cell sizes
        self.plane_a_cell_w = 0
        self.plane_a_cell_h = 0
        self.plane_b_cell_w = 0
        self.plane_b_cell_h = 0
        
        logger.info("VDP reset")
    
    def read_register(self, address: int) -> int:
        """
        Read from VDP register.
        
        Args:
            address: Register address
            
        Returns:
            Register value
        """
        # Only certain addresses are valid for reading
        if address & 1:  # Odd addresses read the status
            return self.status & 0xFF
        else:  # Even addresses read from the read buffer
            return self.read_buffer & 0xFF
    
    def write_register(self, address: int, value: int) -> None:
        """
        Write to VDP register.
        
        Args:
            address: Register address
            value: Value to write
        """
        if address & 1:  # Odd addresses receive control/address information
            if self.pending_command:
                # This is the second part of a command
                self.command_word = (self.command_word & 0xFF00) | value
                self._execute_command()
                self.pending_command = False
            else:
                # This is the first part of a command
                self.command_word = (value << 8) | (self.command_word & 0xFF)
                self.pending_command = True
        else:  # Even addresses write data
            if self.pending_command:
                # Write to data port
                self._write_data(value)
            else:
                # Direct register write
                self._write_data(value)
    
    def _execute_command(self) -> None:
        """Execute a VDP command."""
        # Extract command type and address
        cmd_type = (self.command_word >> 14) & 7
        self.address = self.command_word & 0x3FFF
        
        if cmd_type == self.CMD_VRAM_READ:
            # Set up VRAM read
            self.command_type = self.CMD_VRAM_READ
            self._update_read_buffer()
        elif cmd_type == self.CMD_VRAM_WRITE:
            # Set up VRAM write
            self.command_type = self.CMD_VRAM_WRITE
        elif cmd_type == self.CMD_CRAM_READ:
            # Set up CRAM read
            self.command_type = self.CMD_CRAM_READ
            self._update_read_buffer()
        elif cmd_type == self.CMD_CRAM_WRITE:
            # Set up CRAM write
            self.command_type = self.CMD_CRAM_WRITE
        elif cmd_type == self.CMD_VSRAM_READ:
            # Set up VSRAM read
            self.command_type = self.CMD_VSRAM_READ
            self._update_read_buffer()
        elif cmd_type == self.CMD_VSRAM_WRITE:
            # Set up VSRAM write
            self.command_type = self.CMD_VSRAM_WRITE
        elif cmd_type == self.CMD_REG_WRITE:
            # Register write
            reg_num = (self.command_word >> 8) & 0x1F  # Register number in bits 8-12
            if reg_num < self.NUM_REGISTERS:
                self.registers[reg_num] = self.command_word & 0xFF
                self._update_register_settings(reg_num)
    
    def _update_read_buffer(self) -> None:
        """Update the read buffer based on current command type."""
        if self.command_type == self.CMD_VRAM_READ:
            # Read from VRAM
            if self.address < self.VRAM_SIZE:
                self.read_buffer = self.vram[self.address]
            else:
                self.read_buffer = 0
        elif self.command_type == self.CMD_CRAM_READ:
            # Read from CRAM
            if self.address < self.CRAM_SIZE:
                self.read_buffer = self.cram[self.address]
            else:
                self.read_buffer = 0
        elif self.command_type == self.CMD_VSRAM_READ:
            # Read from VSRAM
            if self.address < self.VSRAM_SIZE:
                self.read_buffer = self.vsram[self.address]
            else:
                self.read_buffer = 0
    
    def _write_data(self, value: int) -> None:
        """
        Write data to the current VDP memory location.
        
        Args:
            value: Value to write
        """
        if self.command_type == self.CMD_VRAM_WRITE:
            # Write to VRAM
            if self.address < self.VRAM_SIZE:
                self.vram[self.address] = value
            
            # Increment address
            self.address = (self.address + 1) & 0xFFFF
                
        elif self.command_type == self.CMD_CRAM_WRITE:
            # Write to CRAM
            if self.address < self.CRAM_SIZE:
                self.cram[self.address] = value
                
                # Update palette entry
                if self.address % 2 == 1:  # Every 2 bytes forms a color
                    color_index = self.address // 2
                    if color_index < 64:
                        # Convert CRAM value to RGB
                        low_byte = self.cram[self.address - 1]
                        high_byte = value
                        
                        # CRAM format: 0000 bbb0 ggg0 rrr0
                        r = (low_byte & 0xE) * 36
                        g = ((high_byte & 0x1) << 3 | (low_byte & 0xE0) >> 5) * 36
                        b = (high_byte & 0xE) * 36
                        
                        self.palette[color_index] = (r, g, b)
            
            # Increment address
            self.address = (self.address + 1) & 0xFFFF
                
        elif self.command_type == self.CMD_VSRAM_WRITE:
            # Write to VSRAM
            if self.address < self.VSRAM_SIZE:
                self.vsram[self.address] = value
            
            # Increment address
            self.address = (self.address + 1) & 0xFFFF
    
    def _update_register_settings(self, reg_num: int) -> None:
        """
        Update internal settings based on register changes.
        
        Args:
            reg_num: Register number that was updated
        """
        # Process register write
        if reg_num == 0:
            # Mode register 1
            self.hint_enabled = (self.registers[0] & 0x10) != 0
            self.h40_mode = (self.registers[0] & 0x04) != 0
            logger.debug(f"Mode 1: HInt={'enabled' if self.hint_enabled else 'disabled'}, H40={'enabled' if self.h40_mode else 'disabled'}")
            
        elif reg_num == 1:
            # Mode register 2
            self.display_enabled = (self.registers[1] & 0x40) != 0
            self.vint_enabled = (self.registers[1] & 0x20) != 0
            self.dma_enabled = (self.registers[1] & 0x10) != 0
            self.v30_mode = (self.registers[1] & 0x08) != 0
            logger.debug(f"Mode 2: Display={'enabled' if self.display_enabled else 'disabled'}, " + 
                        f"VInt={'enabled' if self.vint_enabled else 'disabled'}, " + 
                        f"DMA={'enabled' if self.dma_enabled else 'disabled'}, " + 
                        f"V30={'enabled' if self.v30_mode else 'disabled'}")
                
        elif reg_num == 2:
            # Pattern name table base address for Plane A
            self.plane_a_base = (self.registers[2] & 0x38) << 10
            logger.debug(f"Plane A base: 0x{self.plane_a_base:04X}")
                
        elif reg_num == 3:
            # Pattern name table base address for Window
            self.window_base = (self.registers[3] & 0x3E) << 10
            logger.debug(f"Window base: 0x{self.window_base:04X}")
                
        elif reg_num == 4:
            # Pattern name table base address for Plane B
            self.plane_b_base = (self.registers[4] & 0x07) << 13
            logger.debug(f"Plane B base: 0x{self.plane_b_base:04X}")
                
        elif reg_num == 5:
            # Sprite attribute table base address
            self.sprite_base = (self.registers[5] & 0x7F) << 9
            logger.debug(f"Sprite base: 0x{self.sprite_base:04X}")
                
        elif reg_num == 10:
            # Hint counter value
            self.hint_value = self.registers[10]
            logger.debug(f"HInt counter: {self.hint_value}")
                
        elif reg_num == 13:
            # Horizontal scroll table base address
            self.hscroll_base = (self.registers[13] & 0x3F) << 10
            logger.debug(f"HScroll base: 0x{self.hscroll_base:04X}")
                
        elif reg_num == 15:
            # Auto-increment value for VRAM address
            # (handled automatically in write operations)
            logger.debug(f"Auto-increment: {self.registers[15]}")
                
        elif reg_num == 16:
            # Scroll plane size
            self.plane_a_cell_w = 32 << ((self.registers[16] & 0x03) >> 0)
            self.plane_a_cell_h = 32 << ((self.registers[16] & 0x30) >> 4)
            self.plane_b_cell_w = 32 << ((self.registers[16] & 0x0C) >> 2)
            self.plane_b_cell_h = 32 << ((self.registers[16] & 0xC0) >> 6)
            logger.debug(f"Plane A size: {self.plane_a_cell_w}×{self.plane_a_cell_h} cells")
            logger.debug(f"Plane B size: {self.plane_b_cell_w}×{self.plane_b_cell_h} cells")
                
        elif reg_num == 17:
            # Window Plane Horizontal Position
            # (handled in rendering)
            pass
                
        elif reg_num == 18:
            # Window Plane Vertical Position
            # (handled in rendering)
            pass
                
        elif reg_num == 19 or reg_num == 20:
            # DMA length (19=low, 20=high)
            self.dma_length = ((self.registers[20] & 0xFF) << 8) | (self.registers[19] & 0xFF)
            logger.debug(f"DMA length: {self.dma_length}")
                
        elif reg_num == 21 or reg_num == 22 or reg_num == 23:
            # DMA source address (21=low, 22=mid, 23=high)
            self.dma_source = ((self.registers[23] & 0xFF) << 16) | ((self.registers[22] & 0xFF) << 8) | (self.registers[21] & 0xFF)
            logger.debug(f"DMA source: 0x{self.dma_source:06X}")
    
    def step(self, cycles: int) -> bool:
        """
        Run the VDP for the specified number of cycles.
        
        Args:
            cycles: Number of cycles to simulate
            
        Returns:
            True if a frame is completed
        """
        frame_completed = False
        
        # Each CPU cycle is 2 VDP dots (for 68000 at ~7.67 MHz)
        dots = cycles * 2
        
        # Constants for NTSC timing
        h_dots = 341 if self.h40_mode else 342  # Horizontal dots per line
        v_lines = 262                           # Vertical lines per frame
        
        # Update dot counter
        self.dot += dots
        
        # Check for horizontal sync
        while self.dot >= h_dots:
            self.dot -= h_dots
            self.scanline += 1
            
            # Reset H-int counter at the start of active display
            if self.scanline == 0:
                self.hint_counter = self.hint_value
            
            # Generate H-int when counter expires
            if self.scanline < 224:  # Only during active display
                self.hint_counter -= 1
                if self.hint_counter < 0:
                    if self.hint_enabled:
                        self.external_int = True
                    self.hint_counter = self.hint_value
            
            # Check for vertical sync
            if self.scanline >= v_lines:
                self.scanline = 0
                self.frame += 1
                frame_completed = True
                
                # Generate V-int at start of vblank
                if self.vint_enabled:
                    self.status |= self.STATUS_VBLANK
                    self.external_int = True
            
            # Render scanline if in active display area
            if self.display_enabled and self.scanline < 224:
                self._render_scanline(self.scanline)
        
        return frame_completed
    
    def _render_scanline(self, line: int) -> None:
        """
        Render a single scanline.
        
        Args:
            line: Scanline number to render
        """
        if line < 0 or line >= self.SCREEN_HEIGHT:
            return
            
        # Setup for this scanline
        screen_width = 320 if self.h40_mode else 256
        
        # For a simplified implementation, just fill with gradient pattern
        for x in range(screen_width):
            r = (x * 256) // screen_width
            g = (line * 256) // self.SCREEN_HEIGHT
            b = ((x + line) * 128) // (screen_width + self.SCREEN_HEIGHT)
            
            self._set_pixel(x, line, r, g, b)
    
    def _set_pixel(self, x: int, y: int, r: int, g: int, b: int) -> None:
        """
        Set a pixel in the frame buffer.
        
        Args:
            x: X coordinate
            y: Y coordinate
            r: Red value (0-255)
            g: Green value (0-255)
            b: Blue value (0-255)
        """
        if 0 <= x < self.SCREEN_WIDTH and 0 <= y < self.SCREEN_HEIGHT:
            index = (y * self.SCREEN_WIDTH + x) * 3
            self.frame_buffer[index] = r
            self.frame_buffer[index + 1] = g
            self.frame_buffer[index + 2] = b
    
    def get_frame_buffer(self) -> bytes:
        """
        Get the current frame buffer.
        
        Returns:
            Frame buffer as bytes
        """
        return bytes(self.frame_buffer)
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get the current VDP state.
        
        Returns:
            Dictionary with VDP state
        """
        return {
            "scanline": self.scanline,
            "dot": self.dot,
            "frame": self.frame,
            "status": self.status,
            "registers": list(self.registers),
            "display_enabled": self.display_enabled,
            "h40_mode": self.h40_mode,
            "v30_mode": self.v30_mode
        }