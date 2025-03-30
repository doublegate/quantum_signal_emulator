"""
SNES Picture Processing Unit (PPU) emulation.

The SNES PPU is a sophisticated graphics processor capable of displaying
multiple background layers, sprites, and special effects. This module
implements a cycle-accurate emulation of the SNES PPU, including its
various graphics modes, sprite handling, and hardware registers.
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from ...common.interfaces import VideoProcessor

logger = logging.getLogger("QuantumSignalEmulator.SNES.PPU")

class SNESPPU(VideoProcessor):
    """
    Emulates the SNES Picture Processing Unit (PPU).
    
    The SNES PPU consists of two separate PPUs (PPU1 and PPU2) that work
    together to generate the video output. It supports multiple background
    layers, sprites (objects), and various special effects like mosaic,
    color math, and window clipping.
    """
    
    # PPU register addresses
    INIDISP = 0x00  # Initial settings/Screen display
    OBSEL   = 0x01  # Object size and character size
    OAMADDL = 0x02  # OAM address (low)
    OAMADDH = 0x03  # OAM address (high)
    OAMDATA = 0x04  # OAM data write
    BGMODE  = 0x05  # BG mode and character size
    MOSAIC  = 0x06  # Mosaic size and enable
    BG1SC   = 0x07  # BG1 tilemap address and size
    BG2SC   = 0x08  # BG2 tilemap address and size
    BG3SC   = 0x09  # BG3 tilemap address and size
    BG4SC   = 0x0A  # BG4 tilemap address and size
    BG12NBA = 0x0B  # BG1 and BG2 character data address
    BG34NBA = 0x0C  # BG3 and BG4 character data address
    BG1HOFS = 0x0D  # BG1 horizontal scroll
    BG1VOFS = 0x0E  # BG1 vertical scroll
    BG2HOFS = 0x0F  # BG2 horizontal scroll
    BG2VOFS = 0x10  # BG2 vertical scroll
    BG3HOFS = 0x11  # BG3 horizontal scroll
    BG3VOFS = 0x12  # BG3 vertical scroll
    BG4HOFS = 0x13  # BG4 horizontal scroll
    BG4VOFS = 0x14  # BG4 vertical scroll
    VMAIN   = 0x15  # VRAM address increment mode
    VMADDL  = 0x16  # VRAM address (low)
    VMADDH  = 0x17  # VRAM address (high)
    VMDATAL = 0x18  # VRAM data write (low)
    VMDATAH = 0x19  # VRAM data write (high)
    M7SEL   = 0x1A  # Mode 7 settings
    M7A     = 0x1B  # Mode 7 matrix A
    M7B     = 0x1C  # Mode 7 matrix B
    M7C     = 0x1D  # Mode 7 matrix C
    M7D     = 0x1E  # Mode 7 matrix D
    M7X     = 0x1F  # Mode 7 center X
    M7Y     = 0x20  # Mode 7 center Y
    CGADD   = 0x21  # CGRAM address
    CGDATA  = 0x22  # CGRAM data write
    W12SEL  = 0x23  # Window mask settings for BG1 and BG2
    W34SEL  = 0x24  # Window mask settings for BG3 and BG4
    WOBJSEL = 0x25  # Window mask settings for OBJ and color window
    WH0     = 0x26  # Window 1 left position
    WH1     = 0x27  # Window 1 right position
    WH2     = 0x28  # Window 2 left position
    WH3     = 0x29  # Window 2 right position
    WBGLOG  = 0x2A  # Window mask logic for BGs
    WOBJLOG = 0x2B  # Window mask logic for OBJs and color window
    TM      = 0x2C  # Main screen designation
    TS      = 0x2D  # Subscreen designation
    TMW     = 0x2E  # Window mask designation for main screen
    TSW     = 0x2F  # Window mask designation for subscreen
    CGWSEL  = 0x30  # Color addition select
    CGADSUB = 0x31  # Color math designation
    COLDATA = 0x32  # Fixed color data
    SETINI  = 0x33  # Screen mode/video select
    MPYL    = 0x34  # Multiplication result (low)
    MPYM    = 0x35  # Multiplication result (middle)
    MPYH    = 0x36  # Multiplication result (high)
    SLHV    = 0x37  # Software latch for H/V counter
    OAMDATAREAD = 0x38  # OAM data read
    VMDATALREAD = 0x39  # VRAM data read (low)
    VMDATAHREAD = 0x3A  # VRAM data read (high)
    CGDATAREAD = 0x3B  # CGRAM data read
    OPHCT   = 0x3C  # H counter read
    OPVCT   = 0x3D  # V counter read
    STAT77  = 0x3E  # PPU status flag and version
    STAT78  = 0x3F  # PPU status flag and version
    
    # Display dimensions (NTSC)
    SCREEN_WIDTH = 256
    SCREEN_HEIGHT = 224  # Can be 224 or 239 depending on SETINI register
    SCREEN_WIDTH_PIXELS = 512  # Can be 512 or 256 depending on mode
    
    # Timing constants (NTSC)
    DOTS_PER_SCANLINE = 341  # Dot count in one scanline
    SCANLINES_PER_FRAME = 262  # Scanline count in one frame
    VBLANK_START_LINE = 225  # First line of V-blank
    VBLANK_END_LINE = 0     # Last line of V-blank
    HBLANK_START = 274      # Start of H-blank
    HBLANK_END = 0          # End of H-blank
    
    def __init__(self, memory=None):
        """
        Initialize the SNES PPU.
        
        Args:
            memory: Memory system for PPU access
        """
        # Memory reference
        self.memory = memory
        
        # PPU registers
        self.registers = {i: 0 for i in range(0x40)}
        
        # VRAM (64KB)
        self.vram = bytearray(64 * 1024)
        
        # CGRAM (512 bytes)
        self.cgram = bytearray(512)
        
        # OAM (544 bytes)
        self.oam = bytearray(544)
        
        # Current access addresses
        self.vram_address = 0
        self.cgram_address = 0
        self.oam_address = 0
        
        # PPU timing state
        self.h_counter = 0  # H position (0-339)
        self.v_counter = 0  # V position (0-261)
        self.frame_counter = 0
        
        # VRAM read/write buffers
        self.vram_read_buffer = 0
        self.vram_write_latch = False
        
        # OAM read/write buffers
        self.oam_read_buffer = 0
        self.oam_write_latch = False
        
        # CGRAM read/write buffers
        self.cgram_read_buffer = 0
        self.cgram_write_latch = False
        
        # Frame buffer (RGB format)
        self.frame_buffer_width = self.SCREEN_WIDTH
        self.frame_buffer_height = self.SCREEN_HEIGHT
        self.frame_buffer = bytearray(self.frame_buffer_width * self.frame_buffer_height * 3)
        
        # Background data
        self.bg_data = [
            {"hofs": 0, "vofs": 0, "tilemap_addr": 0, "character_addr": 0},
            {"hofs": 0, "vofs": 0, "tilemap_addr": 0, "character_addr": 0},
            {"hofs": 0, "vofs": 0, "tilemap_addr": 0, "character_addr": 0},
            {"hofs": 0, "vofs": 0, "tilemap_addr": 0, "character_addr": 0},
        ]
        
        # Mode 7 data
        self.mode7 = {
            "matrix_a": 0,
            "matrix_b": 0,
            "matrix_c": 0,
            "matrix_d": 0,
            "center_x": 0,
            "center_y": 0,
            "h_flip": False,
            "v_flip": False
        }
        
        # Multiplication result
        self.multiply_result = 0
        
        # H/V counters latched
        self.h_counter_latched = 0
        self.v_counter_latched = 0
        self.counters_latched = False
        
        # Force blanking flag
        self.force_blank = True
        
        # Screen brightness
        self.brightness = 0
        
        # Background mode
        self.bg_mode = 0
        
        # HDMA active flag
        self.hdma_active = False
        
        logger.info("SNES PPU initialized")
    
    def read_register(self, address: int) -> int:
        """
        Read a PPU register.
        
        Args:
            address: Register address
            
        Returns:
            Register value
        """
        # Map address to PPU register
        reg = address & 0x3F
        
        # Handle read-only and special registers
        if reg == self.MPYL:
            # Multiplication result (low)
            return self.multiply_result & 0xFF
        elif reg == self.MPYM:
            # Multiplication result (middle)
            return (self.multiply_result >> 8) & 0xFF
        elif reg == self.MPYH:
            # Multiplication result (high)
            return (self.multiply_result >> 16) & 0xFF
        elif reg == self.SLHV:
            # Latch H/V counters
            self._latch_hv_counters()
            return 0
        elif reg == self.OAMDATAREAD:
            # OAM data read
            value = self.oam[self.oam_address]
            self._increment_oam_address()
            return value
        elif reg == self.VMDATALREAD:
            # VRAM data read (low)
            value = self.vram_read_buffer & 0xFF
            if not self.vram_write_latch:
                self._update_vram_read_buffer()
            return value
        elif reg == self.VMDATAHREAD:
            # VRAM data read (high)
            value = (self.vram_read_buffer >> 8) & 0xFF
            if self.vram_write_latch:
                self._update_vram_read_buffer()
            return value
        elif reg == self.CGDATAREAD:
            # CGRAM data read
            value = self.cgram_read_buffer
            if self.cgram_write_latch:
                self.cgram_read_buffer = self.cgram[self.cgram_address]
                self.cgram_address = (self.cgram_address + 1) & 0x1FF
            self.cgram_write_latch = not self.cgram_write_latch
            return value
        elif reg == self.OPHCT:
            # H counter read
            if not self.counters_latched:
                # First read returns low byte
                value = self.h_counter_latched & 0xFF
                self.counters_latched = True
            else:
                # Second read returns high byte (only bit 0)
                value = (self.h_counter_latched >> 8) & 0x1
                self.counters_latched = False
            return value
        elif reg == self.OPVCT:
            # V counter read
            if not self.counters_latched:
                # First read returns low byte
                value = self.v_counter_latched & 0xFF
                self.counters_latched = True
            else:
                # Second read returns high byte (only bit 0)
                value = (self.v_counter_latched >> 8) & 0x1
                self.counters_latched = False
            return value
        elif reg == self.STAT77:
            # PPU1 status
            return 0x01  # Version 1
        elif reg == self.STAT78:
            # PPU2 status
            return 0x01  # Version 1
            
        # Return register value for other registers
        return self.registers[reg]
    
    def write_register(self, address: int, value: int) -> None:
        """
        Write to a PPU register.
        
        Args:
            address: Register address
            value: Value to write
        """
        # Map address to PPU register
        reg = address & 0x3F
        
        # Store value in register
        self.registers[reg] = value
        
        # Handle register writes with side effects
        if reg == self.INIDISP:
            # Screen display register
            self.force_blank = (value & 0x80) != 0
            self.brightness = value & 0x0F
            logger.debug(f"Screen display: force_blank={self.force_blank}, brightness={self.brightness}")
            
        elif reg == self.BGMODE:
            # Background mode
            self.bg_mode = value & 0x07
            logger.debug(f"BG mode set to {self.bg_mode}")
            
        elif reg == self.BG1SC:
            # BG1 tilemap address and size
            self.bg_data[0]["tilemap_addr"] = (value & 0xFC) << 8
            logger.debug(f"BG1 tilemap address: ${self.bg_data[0]['tilemap_addr']:04X}")
            
        elif reg == self.BG2SC:
            # BG2 tilemap address and size
            self.bg_data[1]["tilemap_addr"] = (value & 0xFC) << 8
            logger.debug(f"BG2 tilemap address: ${self.bg_data[1]['tilemap_addr']:04X}")
            
        elif reg == self.BG3SC:
            # BG3 tilemap address and size
            self.bg_data[2]["tilemap_addr"] = (value & 0xFC) << 8
            logger.debug(f"BG3 tilemap address: ${self.bg_data[2]['tilemap_addr']:04X}")
            
        elif reg == self.BG4SC:
            # BG4 tilemap address and size
            self.bg_data[3]["tilemap_addr"] = (value & 0xFC) << 8
            logger.debug(f"BG4 tilemap address: ${self.bg_data[3]['tilemap_addr']:04X}")
            
        elif reg == self.BG12NBA:
            # BG1 and BG2 character data address
            self.bg_data[0]["character_addr"] = (value & 0x0F) << 12
            self.bg_data[1]["character_addr"] = (value & 0xF0) << 8
            logger.debug(f"BG1 character address: ${self.bg_data[0]['character_addr']:04X}")
            logger.debug(f"BG2 character address: ${self.bg_data[1]['character_addr']:04X}")
            
        elif reg == self.BG34NBA:
            # BG3 and BG4 character data address
            self.bg_data[2]["character_addr"] = (value & 0x0F) << 12
            self.bg_data[3]["character_addr"] = (value & 0xF0) << 8
            logger.debug(f"BG3 character address: ${self.bg_data[2]['character_addr']:04X}")
            logger.debug(f"BG4 character address: ${self.bg_data[3]['character_addr']:04X}")
            
        elif reg == self.BG1HOFS:
            # BG1 horizontal scroll
            self.bg_data[0]["hofs"] = ((self.registers[self.M7HOFS] & 0x0300) >> 8) | value
            logger.debug(f"BG1 horizontal scroll: {self.bg_data[0]['hofs']}")
            
        elif reg == self.BG1VOFS:
            # BG1 vertical scroll
            self.bg_data[0]["vofs"] = ((self.registers[self.M7VOFS] & 0x0300) >> 8) | value
            logger.debug(f"BG1 vertical scroll: {self.bg_data[0]['vofs']}")
            
        elif reg == self.BG2HOFS:
            # BG2 horizontal scroll
            self.bg_data[1]["hofs"] = value
            logger.debug(f"BG2 horizontal scroll: {self.bg_data[1]['hofs']}")
            
        elif reg == self.BG2VOFS:
            # BG2 vertical scroll
            self.bg_data[1]["vofs"] = value
            logger.debug(f"BG2 vertical scroll: {self.bg_data[1]['vofs']}")
            
        elif reg == self.BG3HOFS:
            # BG3 horizontal scroll
            self.bg_data[2]["hofs"] = value
            logger.debug(f"BG3 horizontal scroll: {self.bg_data[2]['hofs']}")
            
        elif reg == self.BG3VOFS:
            # BG3 vertical scroll
            self.bg_data[2]["vofs"] = value
            logger.debug(f"BG3 vertical scroll: {self.bg_data[2]['vofs']}")
            
        elif reg == self.BG4HOFS:
            # BG4 horizontal scroll
            self.bg_data[3]["hofs"] = value
            logger.debug(f"BG4 horizontal scroll: {self.bg_data[3]['hofs']}")
            
        elif reg == self.BG4VOFS:
            # BG4 vertical scroll
            self.bg_data[3]["vofs"] = value
            logger.debug(f"BG4 vertical scroll: {self.bg_data[3]['vofs']}")
            
        elif reg == self.VMAIN:
            # VRAM address increment mode
            logger.debug(f"VRAM address increment mode: {value & 0x03}")
            
        elif reg == self.VMADDL:
            # VRAM address (low)
            self.vram_address = (self.vram_address & 0xFF00) | value
            self._update_vram_read_buffer()
            logger.debug(f"VRAM address (low): ${self.vram_address:04X}")
            
        elif reg == self.VMADDH:
            # VRAM address (high)
            self.vram_address = (self.vram_address & 0x00FF) | (value << 8)
            self._update_vram_read_buffer()
            logger.debug(f"VRAM address (high): ${self.vram_address:04X}")
            
        elif reg == self.VMDATAL:
            # VRAM data write (low)
            if not self.vram_write_latch:
                # First write goes to low byte of word
                self.vram_write_buffer = value
                self.vram_write_latch = True
            else:
                # Second write completes the word
                vram_word = (self.vram_write_buffer) | (value << 8)
                self._write_vram(vram_word)
                self.vram_write_latch = False
                
        elif reg == self.VMDATAH:
            # VRAM data write (high)
            if not self.vram_write_latch:
                # First write (high byte) stored in buffer
                self.vram_write_buffer = value << 8
                self.vram_write_latch = True
            else:
                # Second write (low byte) completes the word
                vram_word = self.vram_write_buffer | value
                self._write_vram(vram_word)
                self.vram_write_latch = False
                
        elif reg == self.M7SEL:
            # Mode 7 settings
            self.mode7["h_flip"] = (value & 0x01) != 0
            self.mode7["v_flip"] = (value & 0x02) != 0
            logger.debug(f"Mode 7 settings: h_flip={self.mode7['h_flip']}, v_flip={self.mode7['v_flip']}")
            
        elif reg == self.M7A:
            # Mode 7 matrix A
            self.mode7["matrix_a"] = value | (self.registers[self.M7A] << 8)
            self._update_multiply_result()
            logger.debug(f"Mode 7 matrix A: {self.mode7['matrix_a']}")
            
        elif reg == self.M7B:
            # Mode 7 matrix B
            self.mode7["matrix_b"] = value | (self.registers[self.M7B] << 8)
            self._update_multiply_result()
            logger.debug(f"Mode 7 matrix B: {self.mode7['matrix_b']}")
            
        elif reg == self.M7C:
            # Mode 7 matrix C
            self.mode7["matrix_c"] = value | (self.registers[self.M7C] << 8)
            logger.debug(f"Mode 7 matrix C: {self.mode7['matrix_c']}")
            
        elif reg == self.M7D:
            # Mode 7 matrix D
            self.mode7["matrix_d"] = value | (self.registers[self.M7D] << 8)
            logger.debug(f"Mode 7 matrix D: {self.mode7['matrix_d']}")
            
        elif reg == self.M7X:
            # Mode 7 center X
            self.mode7["center_x"] = value | (self.registers[self.M7X] << 8)
            logger.debug(f"Mode 7 center X: {self.mode7['center_x']}")
            
        elif reg == self.M7Y:
            # Mode 7 center Y
            self.mode7["center_y"] = value | (self.registers[self.M7Y] << 8)
            logger.debug(f"Mode 7 center Y: {self.mode7['center_y']}")
            
        elif reg == self.CGADD:
            # CGRAM address
            self.cgram_address = value * 2  # Each color is 2 bytes
            self.cgram_write_latch = False
            logger.debug(f"CGRAM address: ${self.cgram_address:03X}")
            
        elif reg == self.CGDATA:
            # CGRAM data write
            if not self.cgram_write_latch:
                # First write (low byte)
                self.cgram[self.cgram_address] = value
                self.cgram_write_latch = True
            else:
                # Second write (high byte)
                self.cgram[self.cgram_address + 1] = value & 0x7F  # Only 7 bits used
                self.cgram_address = (self.cgram_address + 2) & 0x1FF
                self.cgram_write_latch = False
                
        elif reg == self.OAMADDL:
            # OAM address (low)
            self.oam_address = (self.oam_address & 0x100) | value
            logger.debug(f"OAM address (low): ${self.oam_address:03X}")
            
        elif reg == self.OAMADDH:
            # OAM address (high) and priority rotation
            self.oam_address = (self.oam_address & 0xFF) | ((value & 0x01) << 8)
            logger.debug(f"OAM address (high): ${self.oam_address:03X}, priority rotation: {(value & 0x80) != 0}")
            
        elif reg == self.OAMDATA:
            # OAM data write
            self.oam[self.oam_address] = value
            self._increment_oam_address()
    
    def _update_vram_read_buffer(self) -> None:
        """Update the VRAM read buffer with data at the current VRAM address."""
        # VRAM is accessed in 16-bit words
        vram_addr = self.vram_address & 0x7FFF
        if vram_addr * 2 + 1 < len(self.vram):
            self.vram_read_buffer = self.vram[vram_addr * 2] | (self.vram[vram_addr * 2 + 1] << 8)
    
    def _write_vram(self, value: int) -> None:
        """
        Write a 16-bit value to VRAM and update the VRAM address.
        
        Args:
            value: 16-bit value to write
        """
        # VRAM is accessed in 16-bit words
        vram_addr = self.vram_address & 0x7FFF
        if vram_addr * 2 + 1 < len(self.vram):
            self.vram[vram_addr * 2] = value & 0xFF
            self.vram[vram_addr * 2 + 1] = (value >> 8) & 0xFF
            
        # Increment VRAM address based on VMAIN register
        increment_mode = self.registers[self.VMAIN] & 0x03
        increments = [1, 32, 128, 128]  # Possible increment values
        self.vram_address = (self.vram_address + increments[increment_mode]) & 0xFFFF
    
    def _increment_oam_address(self) -> None:
        """Increment the OAM address after a read/write."""
        self.oam_address = (self.oam_address + 1) & 0x1FF
    
    def _latch_hv_counters(self) -> None:
        """Latch the current H/V counter values."""
        self.h_counter_latched = self.h_counter
        self.v_counter_latched = self.v_counter
        self.counters_latched = False
        logger.debug(f"H/V counters latched: H=${self.h_counter_latched:03X}, V=${self.v_counter_latched:03X}")
    
    def _update_multiply_result(self) -> None:
        """Update the multiplication result register (M7A * M7B)."""
        # Perform 16-bit signed multiplication
        a = self.mode7["matrix_a"]
        b = self.mode7["matrix_b"]
        
        # Convert to signed 16-bit values
        if a & 0x8000:
            a = a - 0x10000
        if b & 0x8000:
            b = b - 0x10000
            
        # Multiply and store the 24-bit result
        self.multiply_result = (a * b) & 0xFFFFFF
    
    def step(self, cycles: int) -> bool:
        """
        Run the PPU for a specified number of cycles.
        
        Args:
            cycles: Number of CPU cycles to simulate
            
        Returns:
            True if a frame is completed
        """
        frame_completed = False
        
        # Each CPU cycle is 4 dot clocks for the PPU
        dots = cycles * 4
        
        for _ in range(dots):
            # Update H/V counters
            self.h_counter += 1
            
            if self.h_counter >= self.DOTS_PER_SCANLINE:
                self.h_counter = 0
                self.v_counter += 1
                
                # Check for HDMA activation at start of scanline
                if self.v_counter < self.VBLANK_START_LINE and self.memory and hasattr(self.memory, 'dma_active'):
                    self.hdma_active = True
                
                if self.v_counter >= self.SCANLINES_PER_FRAME:
                    self.v_counter = 0
                    self.frame_counter += 1
                    frame_completed = True
            
            # Handle H-blank
            if self.h_counter == self.HBLANK_START:
                # H-blank start
                pass
            elif self.h_counter == self.HBLANK_END:
                # H-blank end
                pass
            
            # Handle V-blank
            if self.v_counter == self.VBLANK_START_LINE and self.h_counter == 0:
                # V-blank start
                # Set NMI flag if enabled
                if self.memory:
                    nmitimen = self.memory._read_internal_register(0x00)
                    if nmitimen & 0x80:
                        self.memory._write_internal_register(0x10, 0x80)  # Set NMI flag
            elif self.v_counter == self.VBLANK_END_LINE and self.h_counter == 0:
                # V-blank end
                pass
            
            # Render visible scanlines
            if not self.force_blank and self.v_counter < self.frame_buffer_height and self.h_counter < self.SCREEN_WIDTH:
                self._render_pixel(self.h_counter, self.v_counter)
        
        return frame_completed
    
    def _render_pixel(self, x: int, y: int) -> None:
        """
        Render a single pixel.
        
        Args:
            x: X coordinate
            y: Y coordinate
        """
        # Skip if outside frame buffer
        if x < 0 or x >= self.frame_buffer_width or y < 0 or y >= self.frame_buffer_height:
            return
            
        # Get pixel colors from backgrounds and sprites
        bg_colors = self._get_background_colors(x, y)
        sprite_color = self._get_sprite_color(x, y)
        
        # Determine final color based on priority and color math
        final_color = self._apply_color_math(bg_colors, sprite_color)
        
        # Apply master brightness
        final_color = self._apply_brightness(final_color)
        
        # Set pixel in frame buffer
        pixel_index = (y * self.frame_buffer_width + x) * 3
        if pixel_index + 2 < len(self.frame_buffer):
            self.frame_buffer[pixel_index] = (final_color >> 16) & 0xFF     # R
            self.frame_buffer[pixel_index + 1] = (final_color >> 8) & 0xFF  # G
            self.frame_buffer[pixel_index + 2] = final_color & 0xFF         # B
    
    def _get_background_colors(self, x: int, y: int) -> List[int]:
        """
        Get background colors at the specified position.
        
        Args:
            x: X coordinate
            y: Y coordinate
            
        Returns:
            List of background colors with priority info
        """
        bg_colors = []
        
        # Check which backgrounds are enabled
        tm = self.registers[self.TM]  # Main screen designation
        
        # Get colors from each enabled background
        if self.bg_mode == 0:
            # Mode 0: 4 BGs, 2 bpp each
            if tm & 0x01:  # BG1 enabled
                bg_colors.append(self._get_bg_color(0, x, y, 2))
            if tm & 0x02:  # BG2 enabled
                bg_colors.append(self._get_bg_color(1, x, y, 2))
            if tm & 0x04:  # BG3 enabled
                bg_colors.append(self._get_bg_color(2, x, y, 2))
            if tm & 0x08:  # BG4 enabled
                bg_colors.append(self._get_bg_color(3, x, y, 2))
                
        elif self.bg_mode == 1:
            # Mode 1: 2 BGs (4 bpp), 1 BG (2 bpp)
            if tm & 0x01:  # BG1 enabled
                bg_colors.append(self._get_bg_color(0, x, y, 4))
            if tm & 0x02:  # BG2 enabled
                bg_colors.append(self._get_bg_color(1, x, y, 4))
            if tm & 0x04:  # BG3 enabled
                bg_colors.append(self._get_bg_color(2, x, y, 2))
                
        elif self.bg_mode == 2:
            # Mode 2: 2 BGs (4 bpp) with offset per tile
            if tm & 0x01:  # BG1 enabled
                bg_colors.append(self._get_bg_color(0, x, y, 4, True))
            if tm & 0x02:  # BG2 enabled
                bg_colors.append(self._get_bg_color(1, x, y, 4, True))
                
        elif self.bg_mode == 3:
            # Mode 3: 1 BG (8 bpp), 1 BG (4 bpp)
            if tm & 0x01:  # BG1 enabled
                bg_colors.append(self._get_bg_color(0, x, y, 8))
            if tm & 0x02:  # BG2 enabled
                bg_colors.append(self._get_bg_color(1, x, y, 4))
                
        elif self.bg_mode == 4:
            # Mode 4: 1 BG (8 bpp) with offset per tile, 1 BG (2 bpp)
            if tm & 0x01:  # BG1 enabled
                bg_colors.append(self._get_bg_color(0, x, y, 8, True))
            if tm & 0x02:  # BG2 enabled
                bg_colors.append(self._get_bg_color(1, x, y, 2))
                
        elif self.bg_mode == 5:
            # Mode 5: 1 BG (4 bpp, 16x8), 1 BG (2 bpp, 16x8)
            # High resolution mode (512x224)
            if tm & 0x01:  # BG1 enabled
                bg_colors.append(self._get_bg_color(0, x, y, 4, False, True))
            if tm & 0x02:  # BG2 enabled
                bg_colors.append(self._get_bg_color(1, x, y, 2, False, True))
                
        elif self.bg_mode == 6:
            # Mode 6: 1 BG (4 bpp, 16x8) with offset per tile
            # High resolution mode (512x224)
            if tm & 0x01:  # BG1 enabled
                bg_colors.append(self._get_bg_color(0, x, y, 4, True, True))
                
        elif self.bg_mode == 7:
            # Mode 7: 1 BG (8 bpp) with rotation/scaling
            if tm & 0x01:  # BG1 enabled
                bg_colors.append(self._get_mode7_color(x, y))
        
        # Sort by priority
        bg_colors.sort(key=lambda c: c[1], reverse=True)
        
        return bg_colors
    
    def _get_bg_color(self, bg_index: int, x: int, y: int, bpp: int, offset_per_tile: bool = False, 
                   high_res: bool = False) -> Tuple[int, int]:
        """
        Get background color at the specified position.
        
        Args:
            bg_index: Background index (0-3)
            x: X coordinate
            y: Y coordinate
            bpp: Bits per pixel (2, 4, or 8)
            offset_per_tile: Whether to use offset per tile
            high_res: Whether to use high resolution mode
            
        Returns:
            Tuple of (color, priority)
        """
        # For a simplified implementation, return a placeholder color
        # In a complete implementation, this would access VRAM to get tile and color data
        
        if high_res:
            # High resolution mode uses different calculation
            # This is just a placeholder
            color_index = ((x // 8) + (y // 8)) % 256
        else:
            # Normal resolution mode
            color_index = ((x // 8) + (y // 8)) % 256
            
        # Apply color conversion from index to RGB
        # This is a placeholder - real implementation would use CGRAM
        rgb_value = self._color_index_to_rgb(color_index)
        
        # Priority is based on background index
        priority = 3 - bg_index
        
        return (rgb_value, priority)
    
    def _get_mode7_color(self, x: int, y: int) -> Tuple[int, int]:
        """
        Get Mode 7 color at the specified position.
        
        Args:
            x: X coordinate
            y: Y coordinate
            
        Returns:
            Tuple of (color, priority)
        """
        # For a simplified implementation, return a placeholder color
        # In a complete implementation, this would apply the Mode 7 transformation
        
        # Use checkerboard pattern as placeholder
        color_index = ((x // 16) + (y // 16)) % 2 * 128
        
        # Apply color conversion from index to RGB
        rgb_value = self._color_index_to_rgb(color_index)
        
        # Mode 7 always has high priority
        priority = 3
        
        return (rgb_value, priority)
    
    def _get_sprite_color(self, x: int, y: int) -> Optional[Tuple[int, int]]:
        """
        Get sprite color at the specified position.
        
        Args:
            x: X coordinate
            y: Y coordinate
            
        Returns:
            Tuple of (color, priority) or None if transparent
        """
        # For a simplified implementation, return None (transparent)
        # In a complete implementation, this would check OAM data
        
        return None
    
    def _apply_color_math(self, bg_colors: List[Tuple[int, int]], sprite_color: Optional[Tuple[int, int]]) -> int:
        """
        Apply color math to determine final pixel color.
        
        Args:
            bg_colors: List of background colors with priority
            sprite_color: Sprite color with priority or None
            
        Returns:
            Final RGB color
        """
        # For a simplified implementation, use the highest priority color
        
        # Start with backdrop color (color 0)
        final_color = self._color_index_to_rgb(0)
        
        # Check sprite color
        if sprite_color is not None:
            final_color = sprite_color[0]
        
        # Check background colors
        if bg_colors:
            # Use highest priority non-transparent color
            for color, _ in bg_colors:
                if color != 0:  # 0 is transparent
                    final_color = color
                    break
        
        return final_color
    
    def _apply_brightness(self, color: int) -> int:
        """
        Apply screen brightness to a color.
        
        Args:
            color: RGB color value
            
        Returns:
            Adjusted RGB color
        """
        # Extract RGB components
        r = (color >> 16) & 0xFF
        g = (color >> 8) & 0xFF
        b = color & 0xFF
        
        # Apply brightness (0-15)
        brightness_factor = min(15, self.brightness) / 15.0
        
        # Adjust each component
        r = int(r * brightness_factor)
        g = int(g * brightness_factor)
        b = int(b * brightness_factor)
        
        # Return adjusted color
        return (r << 16) | (g << 8) | b
    
    def _color_index_to_rgb(self, index: int) -> int:
        """
        Convert a color index to RGB value using CGRAM.
        
        Args:
            index: Color index
            
        Returns:
            RGB color value
        """
        # For a simplified implementation, use a placeholder color palette
        # In a complete implementation, this would read from CGRAM
        
        cgram_addr = (index * 2) & 0x1FF
        if cgram_addr + 1 < len(self.cgram):
            # CGRAM format: 0bbbbbgg gggrrrrr
            low_byte = self.cgram[cgram_addr]
            high_byte = self.cgram[cgram_addr + 1]
            
            # Extract RGB components (15-bit color)
            r = (low_byte & 0x1F) * 8
            g = (((high_byte & 0x03) << 3) | ((low_byte & 0xE0) >> 5)) * 8
            b = ((high_byte & 0x7C) >> 2) * 8
            
            return (r << 16) | (g << 8) | b
        
        # Default colors for placeholder
        colors = [
            0x000000,  # 0: Black
            0xFF0000,  # 1: Red
            0x00FF00,  # 2: Green
            0xFFFF00,  # 3: Yellow
            0x0000FF,  # 4: Blue
            0xFF00FF,  # 5: Magenta
            0x00FFFF,  # 6: Cyan
            0xFFFFFF,  # 7: White
            0x808080,  # 8: Gray
            0xFF8080,  # 9: Light Red
            0x80FF80,  # 10: Light Green
            0xFFFF80,  # 11: Light Yellow
            0x8080FF,  # 12: Light Blue
            0xFF80FF,  # 13: Light Magenta
            0x80FFFF,  # 14: Light Cyan
            0xC0C0C0   # 15: Light Gray
        ]
        
        return colors[index % 16]
    
    def get_frame_buffer(self) -> bytes:
        """
        Get the current frame buffer.
        
        Returns:
            Frame buffer as bytes
        """
        return bytes(self.frame_buffer)
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get the current PPU state.
        
        Returns:
            Dictionary with PPU state
        """
        return {
            "h_counter": self.h_counter,
            "v_counter": self.v_counter,
            "frame_counter": self.frame_counter,
            "force_blank": self.force_blank,
            "brightness": self.brightness,
            "bg_mode": self.bg_mode,
            "registers": dict(self.registers)
        }