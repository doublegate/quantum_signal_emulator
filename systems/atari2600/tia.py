"""
Atari 2600 TIA (Television Interface Adapter) emulation.

The TIA is the custom chip responsible for generating video and audio output
in the Atari 2600. It has highly unusual video timing characteristics that are
directly controlled by the CPU, requiring cycle-perfect emulation for proper
operation. The TIA generates the video signal in real time as the TV beam
scans across the screen.

This module implements a cycle-accurate emulation of the TIA with all its
objects (players, missiles, ball, playfield) and color capabilities.
"""

from ...common.interfaces import VideoProcessor, Memory
import numpy as np
import logging
from typing import List, Dict, Any

logger = logging.getLogger("QuantumSignalEmulator.Atari2600.TIA")

class AtariTIA(VideoProcessor):
    """
    Emulates the Atari 2600 TIA (Television Interface Adapter).
    
    The TIA creates a video signal by directly controlling the electron beam
    position of a CRT TV. Unlike modern video systems, scanlines can be
    of variable length due to the WSYNC register, which halts the CPU
    until the horizontal blank period. This allows for interesting
    visual effects but requires precise cycle-accurate emulation.
    """
    
    # TIA color palette (NTSC)
    NTSC_PALETTE = [
        0x000000, 0x404040, 0x6C6C6C, 0x909090, 0xB0B0B0, 0xC8C8C8, 0xDCDCDC, 0xECECEC,
        0x444400, 0x646410, 0x848424, 0xA0A034, 0xB8B840, 0xD0D050, 0xE8E85C, 0xFCFC68,
        0x702800, 0x844414, 0x985C28, 0xAC783C, 0xBC8C4C, 0xCCA05C, 0xDCB468, 0xECC878,
        0x841800, 0x983418, 0xAC5030, 0xC06848, 0xD0805C, 0xE09470, 0xECA880, 0xFCBC94,
        0x880000, 0x9C2020, 0xB03C3C, 0xC05858, 0xD07070, 0xE08888, 0xECA0A0, 0xFCB4B4,
        0x78005C, 0x8C2074, 0xA03C88, 0xB0589C, 0xC070B0, 0xD084C0, 0xDC9CD0, 0xECB0E0,
        0x480078, 0x602090, 0x783CA4, 0x8C58B8, 0xA070CC, 0xB484DC, 0xC49CEC, 0xD4B0FC,
        0x140084, 0x302098, 0x4C3CAC, 0x6858C0, 0x7C70D0, 0x9488E0, 0xA8A0EC, 0xBCB4FC,
        0x000088, 0x1C209C, 0x3840B0, 0x505CC0, 0x6874D0, 0x7C8CE0, 0x90A4EC, 0xA4B8FC,
        0x00187C, 0x1C3890, 0x3854A8, 0x5070BC, 0x6888CC, 0x7CA0DC, 0x90B4EC, 0xA4C8FC,
        0x002C5C, 0x1C4C78, 0x386890, 0x5084AC, 0x689CC0, 0x7CB4D4, 0x90CCE8, 0xA4E0FC,
        0x003C2C, 0x1C5C48, 0x387C64, 0x509C80, 0x68B494, 0x7CD0AC, 0x90E4C0, 0xA4FCD4,
        0x003C00, 0x205C20, 0x407C40, 0x5C9C5C, 0x74B474, 0x8CD08C, 0xA4E4A4, 0xB8FCB8,
        0x143800, 0x345C1C, 0x507C38, 0x6C9850, 0x84B468, 0x9CCC7C, 0xB4E490, 0xC8FCA4,
        0x2C3000, 0x4C501C, 0x687034, 0x848C4C, 0x9CA864, 0xB4C078, 0xCCD488, 0xE0EC9C,
        0x442800, 0x644818, 0x846830, 0xA08444, 0xB89C58, 0xD0B46C, 0xE8CC7C, 0xFCE08C
    ]
    
    # Sprite width
    PLAYER_WIDTH = 8
    MISSILE_WIDTH = 1
    BALL_WIDTH = 1
    
    # TIA display dimensions
    DISPLAY_WIDTH = 160
    DISPLAY_HEIGHT = 192
    
    # TIA timing constants
    HBLANK_CLOCKS = 68  # Clock cycles in horizontal blank
    SCANLINE_CLOCKS = 228  # Clock cycles in a scanline (76 color clocks * 3)
    VSYNC_SCANLINES = 3
    VBLANK_SCANLINES = 37
    OVERSCAN_SCANLINES = 30
    
    # TIA register addresses
    VSYNC   = 0x00  # Vertical Sync Set-Clear
    VBLANK  = 0x01  # Vertical Blank Set-Clear
    WSYNC   = 0x02  # Wait for Horizontal Blank
    RSYNC   = 0x03  # Reset Horizontal Sync Counter
    NUSIZ0  = 0x04  # Number/Size player/missile 0
    NUSIZ1  = 0x05  # Number/Size player/missile 1
    COLUP0  = 0x06  # Color-Luminance Player 0
    COLUP1  = 0x07  # Color-Luminance Player 1
    COLUPF  = 0x08  # Color-Luminance Playfield
    COLUBK  = 0x09  # Color-Luminance Background
    CTRLPF  = 0x0A  # Control Playfield, Ball, Collisions
    REFP0   = 0x0B  # Reflection Player 0
    REFP1   = 0x0C  # Reflection Player 1
    PF0     = 0x0D  # Playfield Register 0
    PF1     = 0x0E  # Playfield Register 1
    PF2     = 0x0F  # Playfield Register 2
    RESP0   = 0x10  # Reset Player 0
    RESP1   = 0x11  # Reset Player 1
    RESM0   = 0x12  # Reset Missile 0
    RESM1   = 0x13  # Reset Missile 1
    RESBL   = 0x14  # Reset Ball
    AUDC0   = 0x15  # Audio Control 0
    AUDC1   = 0x16  # Audio Control 1
    AUDF0   = 0x17  # Audio Frequency 0
    AUDF1   = 0x18  # Audio Frequency 1
    AUDV0   = 0x19  # Audio Volume 0
    AUDV1   = 0x1A  # Audio Volume 1
    GRP0    = 0x1B  # Graphics Player 0
    GRP1    = 0x1C  # Graphics Player 1
    ENAM0   = 0x1D  # Enable Missile 0
    ENAM1   = 0x1E  # Enable Missile 1
    ENABL   = 0x1F  # Enable Ball
    HMP0    = 0x20  # Horizontal Motion Player 0
    HMP1    = 0x21  # Horizontal Motion Player 1
    HMM0    = 0x22  # Horizontal Motion Missile 0
    HMM1    = 0x23  # Horizontal Motion Missile 1
    HMBL    = 0x24  # Horizontal Motion Ball
    VDELP0  = 0x25  # Vertical Delay Player 0
    VDELP1  = 0x26  # Vertical Delay Player 1
    VDELBL  = 0x27  # Vertical Delay Ball
    RESMP0  = 0x28  # Reset Missile 0 to Player 0
    RESMP1  = 0x29  # Reset Missile 1 to Player 1
    HMOVE   = 0x2A  # Apply Horizontal Motion
    HMCLR   = 0x2B  # Clear Horizontal Move Registers
    CXCLR   = 0x2C  # Clear Collision Latches
    
    # Collision detection registers (read-only)
    CXM0P   = 0x00  # Collision Missile 0 and Player
    CXM1P   = 0x01  # Collision Missile 1 and Player
    CXP0FB  = 0x02  # Collision Player 0 and Playfield/Ball
    CXP1FB  = 0x03  # Collision Player 1 and Playfield/Ball
    CXM0FB  = 0x04  # Collision Missile 0 and Playfield/Ball
    CXM1FB  = 0x05  # Collision Missile 1 and Playfield/Ball
    CXBLPF  = 0x06  # Collision Ball and Playfield
    CXPPMM  = 0x07  # Collision Player-Player, Missile-Missile
    
    def __init__(self, memory: Memory = None):
        """
        Initialize the TIA.
        
        Args:
            memory: Memory system for DMA
        """
        # TIA registers
        self.registers = {
            self.VSYNC: 0,
            self.VBLANK: 0,
            self.WSYNC: 0,
            self.RSYNC: 0,
            self.NUSIZ0: 0,
            self.NUSIZ1: 0,
            self.COLUP0: 0,
            self.COLUP1: 0,
            self.COLUPF: 0,
            self.COLUBK: 0,
            self.CTRLPF: 0,
            self.REFP0: 0,
            self.REFP1: 0,
            self.PF0: 0,
            self.PF1: 0,
            self.PF2: 0,
            self.RESP0: 0,
            self.RESP1: 0,
            self.RESM0: 0,
            self.RESM1: 0,
            self.RESBL: 0,
            self.AUDC0: 0,
            self.AUDC1: 0,
            self.AUDF0: 0,
            self.AUDF1: 0,
            self.AUDV0: 0,
            self.AUDV1: 0,
            self.GRP0: 0,
            self.GRP1: 0,
            self.ENAM0: 0,
            self.ENAM1: 0,
            self.ENABL: 0,
            self.HMP0: 0,
            self.HMP1: 0,
            self.HMM0: 0,
            self.HMM1: 0,
            self.HMBL: 0,
            self.VDELP0: 0,
            self.VDELP1: 0,
            self.VDELBL: 0,
            self.RESMP0: 0,
            self.RESMP1: 0,
            self.HMOVE: 0,
            self.HMCLR: 0,
            self.CXCLR: 0
        }
        
        # Collision registers (read-only)
        self.collisions = {
            self.CXM0P: 0,
            self.CXM1P: 0,
            self.CXP0FB: 0,
            self.CXP1FB: 0,
            self.CXM0FB: 0,
            self.CXM1FB: 0,
            self.CXBLPF: 0,
            self.CXPPMM: 0
        }
        
        # Graphics objects state
        self.player0 = {
            "graphics": 0,
            "position": 0,
            "reflected": False,
            "size": 1,
            "copies": 1,
            "vdelay": False,
            "vdelaydata": 0,
            "motion": 0,
            "width": self.PLAYER_WIDTH
        }
        
        self.player1 = {
            "graphics": 0,
            "position": 0,
            "reflected": False,
            "size": 1,
            "copies": 1,
            "vdelay": False,
            "vdelaydata": 0,
            "motion": 0,
            "width": self.PLAYER_WIDTH
        }
        
        self.missile0 = {
            "enabled": False,
            "position": 0,
            "size": 1,
            "copies": 1,
            "motion": 0,
            "width": self.MISSILE_WIDTH,
            "resetToPlayer": False
        }
        
        self.missile1 = {
            "enabled": False,
            "position": 0,
            "size": 1,
            "copies": 1,
            "motion": 0,
            "width": self.MISSILE_WIDTH,
            "resetToPlayer": False
        }
        
        self.ball = {
            "enabled": False,
            "position": 0,
            "size": 1,
            "motion": 0,
            "width": self.BALL_WIDTH,
            "vdelay": False,
            "vdelaydata": 0
        }
        
        self.playfield = {
            "reflected": False,
            "score": False,
            "priority": False,
            "pf0": 0,
            "pf1": 0,
            "pf2": 0
        }
        
        # Timing state
        self.cycle = 0  # Current clock cycle in scanline
        self.scanline = 0  # Current scanline
        self.frame = 0  # Frame counter
        
        # Frame buffer (160x192 pixels, RGB format)
        self.frame_buffer = bytearray(self.DISPLAY_WIDTH * self.DISPLAY_HEIGHT * 3)
        
        # WSYNC flag (CPU halted)
        self.wsync_halt = False
        
        # Indicate if we are in VSYNC, VBLANK, VISIBLE, or OVERSCAN section
        self.section = "VSYNC"
        
        # Horizontal blank state
        self.hblank = True
        
        # Audio state
        self.audio_channels = [{"freq": 0, "control": 0, "volume": 0, "divider": 0, "output": 0} for _ in range(2)]
        
        # Reference to memory system
        self.memory = memory
        
        # Lookup tables
        self._generate_lookup_tables()
        
        logger.info("TIA initialized")
    
    def _generate_lookup_tables(self):
        """Generate lookup tables for efficient rendering."""
        # Player sizing lookup table
        # For each NUSIZ value, store: [width multiplier, number of copies, distance between copies]
        self.player_sizes = [
            [1, 1, 0],   # 0: one copy
            [2, 1, 0],   # 1: two copies close
            [1, 2, 16],  # 2: two copies medium
            [1, 3, 32],  # 3: three copies close
            [2, 2, 16],  # 4: two copies medium width
            [1, 2, 32],  # 5: two copies far
            [1, 3, 8],   # 6: three copies medium
            [1, 1, 0]    # 7: one copy (invalid)
        ]
        
        # Missile sizing lookup table (indexed by bits 4-6 of NUSIZ)
        self.missile_sizes = [1, 2, 4, 8]
        
        # Ball sizing lookup table (indexed by bits 4-5 of CTRLPF)
        self.ball_sizes = [1, 2, 4, 8]
        
        # Horizontal motion lookup table (convert register value to pixels)
        self.hmove_values = [0] * 16
        for i in range(16):
            if i < 8:
                self.hmove_values[i] = i
            else:
                self.hmove_values[i] = i - 16
        
        # Playfield pattern lookup for efficient drawing
        self.playfield_pixels = [0] * 160
    
    def read_register(self, address: int) -> int:
        """
        Read a TIA register (including collision detection).
        
        Args:
            address: Register address
            
        Returns:
            Register value
        """
        address &= 0x3F  # Only 6 bits are used for TIA addressing
        
        # Handle read-only collision registers
        if address <= 0x0D:
            if address <= 0x07:
                return self.collisions[address]
            if address == 0x0D:
                return self.registers[self.INPT0]
        
        return 0  # TIA registers are generally write-only
    
    def write_register(self, address: int, value: int) -> None:
        """
        Write to a TIA register.
        
        Args:
            address: Register address
            value: Value to write
        """
        address &= 0x3F  # Only 6 bits are used for TIA addressing
        value &= 0xFF  # Ensure value is a byte
        
        # Store value in register
        if address in self.registers:
            self.registers[address] = value
        
        # Process special register writes with side effects
        if address == self.WSYNC:
            self.wsync_halt = True
        elif address == self.RSYNC:
            self.cycle = 0
        elif address == self.PF0:
            self.playfield["pf0"] = value >> 4  # Only upper 4 bits are used
            self._update_playfield_pattern()
        elif address == self.PF1:
            self.playfield["pf1"] = value
            self._update_playfield_pattern()
        elif address == self.PF2:
            self.playfield["pf2"] = value
            self._update_playfield_pattern()
        elif address == self.RESP0:
            # Position is related to current cycle
            self.player0["position"] = (self.cycle - self.HBLANK_CLOCKS) // 3
            if self.player0["position"] < 0:
                self.player0["position"] += self.DISPLAY_WIDTH
        elif address == self.RESP1:
            self.player1["position"] = (self.cycle - self.HBLANK_CLOCKS) // 3
            if self.player1["position"] < 0:
                self.player1["position"] += self.DISPLAY_WIDTH
        elif address == self.RESM0:
            self.missile0["position"] = (self.cycle - self.HBLANK_CLOCKS) // 3
            if self.missile0["position"] < 0:
                self.missile0["position"] += self.DISPLAY_WIDTH
        elif address == self.RESM1:
            self.missile1["position"] = (self.cycle - self.HBLANK_CLOCKS) // 3
            if self.missile1["position"] < 0:
                self.missile1["position"] += self.DISPLAY_WIDTH
        elif address == self.RESBL:
            self.ball["position"] = (self.cycle - self.HBLANK_CLOCKS) // 3
            if self.ball["position"] < 0:
                self.ball["position"] += self.DISPLAY_WIDTH
        elif address == self.GRP0:
            if self.player0["vdelay"]:
                self.player0["vdelaydata"] = value
            else:
                self.player0["graphics"] = value
        elif address == self.GRP1:
            if self.player1["vdelay"]:
                self.player1["vdelaydata"] = value
            else:
                self.player1["graphics"] = value
                
            # GRP1 triggers the delayed data load of GRP0
            if self.player0["vdelay"]:
                self.player0["graphics"] = self.player0["vdelaydata"]
        elif address == self.ENAM0:
            self.missile0["enabled"] = (value & 0x02) != 0
        elif address == self.ENAM1:
            self.missile1["enabled"] = (value & 0x02) != 0
        elif address == self.ENABL:
            if self.ball["vdelay"]:
                self.ball["vdelaydata"] = (value & 0x02) != 0
            else:
                self.ball["enabled"] = (value & 0x02) != 0
        elif address == self.HMP0:
            self.player0["motion"] = self.hmove_values[(value >> 4) & 0x0F]
        elif address == self.HMP1:
            self.player1["motion"] = self.hmove_values[(value >> 4) & 0x0F]
        elif address == self.HMM0:
            self.missile0["motion"] = self.hmove_values[(value >> 4) & 0x0F]
        elif address == self.HMM1:
            self.missile1["motion"] = self.hmove_values[(value >> 4) & 0x0F]
        elif address == self.HMBL:
            self.ball["motion"] = self.hmove_values[(value >> 4) & 0x0F]
        elif address == self.VDELP0:
            self.player0["vdelay"] = (value & 0x01) != 0
        elif address == self.VDELP1:
            self.player1["vdelay"] = (value & 0x01) != 0
        elif address == self.VDELBL:
            self.ball["vdelay"] = (value & 0x01) != 0
            
            # VDELBL triggers the delayed ball enable
            if self.ball["vdelay"]:
                self.ball["enabled"] = self.ball["vdelaydata"]
        elif address == self.RESMP0:
            self.missile0["resetToPlayer"] = (value & 0x02) != 0
            if self.missile0["resetToPlayer"]:
                self.missile0["position"] = self.player0["position"]
        elif address == self.RESMP1:
            self.missile1["resetToPlayer"] = (value & 0x02) != 0
            if self.missile1["resetToPlayer"]:
                self.missile1["position"] = self.player1["position"]
        elif address == self.HMOVE:
            # Apply horizontal motion
            self._apply_horizontal_motion()
        elif address == self.HMCLR:
            # Clear all horizontal motion registers
            self.player0["motion"] = 0
            self.player1["motion"] = 0
            self.missile0["motion"] = 0
            self.missile1["motion"] = 0
            self.ball["motion"] = 0
        elif address == self.CXCLR:
            # Clear all collision latches
            for reg in self.collisions:
                self.collisions[reg] = 0
        elif address == self.CTRLPF:
            # Control playfield
            self.playfield["reflected"] = (value & 0x01) != 0
            self.playfield["score"] = (value & 0x02) != 0
            self.playfield["priority"] = (value & 0x04) != 0
            self.ball["size"] = self.ball_sizes[(value >> 4) & 0x03]
            self._update_playfield_pattern()
        elif address == self.NUSIZ0:
            # Number-size player-missile 0
            size_index = value & 0x07
            self.player0["size"] = self.player_sizes[size_index][0]
            self.player0["copies"] = self.player_sizes[size_index][1]
            self.missile0["size"] = self.missile_sizes[(value >> 4) & 0x03]
            self.missile0["copies"] = self.player0["copies"]
        elif address == self.NUSIZ1:
            # Number-size player-missile 1
            size_index = value & 0x07
            self.player1["size"] = self.player_sizes[size_index][0]
            self.player1["copies"] = self.player_sizes[size_index][1]
            self.missile1["size"] = self.missile_sizes[(value >> 4) & 0x03]
            self.missile1["copies"] = self.player1["copies"]
        elif address == self.REFP0:
            # Reflection player 0
            self.player0["reflected"] = (value & 0x08) != 0
        elif address == self.REFP1:
            # Reflection player 1
            self.player1["reflected"] = (value & 0x08) != 0
        elif address == self.VSYNC:
            # Vertical sync - start/end frame
            if value & 0x02:
                # Start VSYNC
                if self.section != "VSYNC":
                    self.section = "VSYNC"
                    self.scanline = 0
            else:
                # End VSYNC - move to VBLANK
                if self.section == "VSYNC" and self.scanline >= self.VSYNC_SCANLINES:
                    self.section = "VBLANK"
                    self.scanline = 0
        elif address == self.VBLANK:
            # Vertical blank
            if value & 0x02:
                # Start VBLANK
                if self.section == "VSYNC" and self.scanline >= self.VSYNC_SCANLINES:
                    self.section = "VBLANK"
                    self.scanline = 0
            else:
                # End VBLANK - move to visible area
                if self.section == "VBLANK" and self.scanline >= self.VBLANK_SCANLINES:
                    self.section = "VISIBLE"
                    self.scanline = 0
        elif address in [self.AUDC0, self.AUDC1, self.AUDF0, self.AUDF1, self.AUDV0, self.AUDV1]:
            # Handle audio registers
            channel = 0 if address in [self.AUDC0, self.AUDF0, self.AUDV0] else 1
            
            if address in [self.AUDC0, self.AUDC1]:
                self.audio_channels[channel]["control"] = value & 0x0F
            elif address in [self.AUDF0, self.AUDF1]:
                self.audio_channels[channel]["freq"] = value & 0x1F
            elif address in [self.AUDV0, self.AUDV1]:
                self.audio_channels[channel]["volume"] = value & 0x0F
    
    def step(self, cycles: int) -> bool:
        """
        Run the TIA for specified number of CPU cycles.
        Each CPU cycle is 3 TIA color clocks.
        
        Args:
            cycles: Number of CPU cycles to simulate
            
        Returns:
            True if a frame is completed
        """
        frame_completed = False
        wsync_used = False
        
        for _ in range(cycles):
            # Process 3 color clocks per CPU cycle
            for _ in range(3):
                # Check if we're in horizontal blank
                if self.cycle < self.HBLANK_CLOCKS:
                    self.hblank = True
                else:
                    if self.hblank:
                        # Leaving HBLANK
                        self.hblank = False
                        
                        # Release CPU from WSYNC
                        if self.wsync_halt:
                            self.wsync_halt = False
                            wsync_used = True
                    
                    # Draw pixel if we're in visible area
                    if (not self.hblank and 
                        self.section == "VISIBLE" and 
                        self.scanline < self.DISPLAY_HEIGHT):
                        self._draw_pixel()
                
                # Process collisions
                self._check_collisions()
                
                # Update audio
                self._update_audio()
                
                # Advance clock
                self.cycle += 1
                
                # End of scanline
                if self.cycle >= self.SCANLINE_CLOCKS:
                    self.cycle = 0
                    self.scanline += 1
                    
                    # Handle section changes
                    if self.section == "VSYNC" and self.scanline >= self.VSYNC_SCANLINES:
                        self.section = "VBLANK"
                        self.scanline = 0
                    elif self.section == "VBLANK" and self.scanline >= self.VBLANK_SCANLINES:
                        self.section = "VISIBLE"
                        self.scanline = 0
                    elif self.section == "VISIBLE" and self.scanline >= self.DISPLAY_HEIGHT:
                        self.section = "OVERSCAN"
                        self.scanline = 0
                    elif self.section == "OVERSCAN" and self.scanline >= self.OVERSCAN_SCANLINES:
                        # End of frame
                        self.section = "VSYNC"
                        self.scanline = 0
                        self.frame += 1
                        frame_completed = True
                        
            # If we hit a WSYNC, stop processing more CPU cycles
            if wsync_used and self.wsync_halt:
                break
        
        return frame_completed
    
    def _apply_horizontal_motion(self):
        """Apply horizontal motion to all objects."""
        # Apply motion to players
        self.player0["position"] = (self.player0["position"] - self.player0["motion"]) % self.DISPLAY_WIDTH
        self.player1["position"] = (self.player1["position"] - self.player1["motion"]) % self.DISPLAY_WIDTH
        
        # Apply motion to missiles
        self.missile0["position"] = (self.missile0["position"] - self.missile0["motion"]) % self.DISPLAY_WIDTH
        self.missile1["position"] = (self.missile1["position"] - self.missile1["motion"]) % self.DISPLAY_WIDTH
        
        # Apply motion to ball
        self.ball["position"] = (self.ball["position"] - self.ball["motion"]) % self.DISPLAY_WIDTH
    
    def _update_playfield_pattern(self):
        """Update the playfield pattern for efficient drawing."""
        pf0 = self.playfield["pf0"]
        pf1 = self.playfield["pf1"]
        pf2 = self.playfield["pf2"]
        
        # Clear playfield pixels
        self.playfield_pixels = [0] * 160
        
        # Left half of playfield (0-79)
        # PF0 (4 bits, reversed)
        for i in range(4):
            bit = (pf0 >> i) & 0x01
            for j in range(4):
                self.playfield_pixels[i * 4 + j] = bit
                
        # PF1 (8 bits)
        for i in range(8):
            bit = (pf1 >> (7 - i)) & 0x01  # PF1 is reversed
            for j in range(4):
                self.playfield_pixels[16 + i * 4 + j] = bit
                
        # PF2 (8 bits, normal order)
        for i in range(8):
            bit = (pf2 >> i) & 0x01
            for j in range(4):
                self.playfield_pixels[48 + i * 4 + j] = bit
        
        # Right half of playfield (80-159)
        if self.playfield["reflected"]:
            # Mirror left half for right half
            for i in range(80):
                self.playfield_pixels[80 + i] = self.playfield_pixels[79 - i]
        else:
            # Repeat left half pattern
            # PF0 (4 bits, reversed)
            for i in range(4):
                bit = (pf0 >> i) & 0x01
                for j in range(4):
                    self.playfield_pixels[80 + i * 4 + j] = bit
                    
            # PF1 (8 bits)
            for i in range(8):
                bit = (pf1 >> (7 - i)) & 0x01  # PF1 is reversed
                for j in range(4):
                    self.playfield_pixels[96 + i * 4 + j] = bit
                    
            # PF2 (8 bits, normal order)
            for i in range(8):
                bit = (pf2 >> i) & 0x01
                for j in range(4):
                    self.playfield_pixels[128 + i * 4 + j] = bit
    
    def _draw_pixel(self):
        """Draw a pixel at the current beam position."""
        # Calculate x coordinate (relative to left of visible screen)
        x = self.cycle - self.HBLANK_CLOCKS
        
        # Skip if outside visible area
        if x < 0 or x >= self.DISPLAY_WIDTH:
            return
            
        # Calculate pixel index in frame buffer
        pixel_index = (self.scanline * self.DISPLAY_WIDTH + x) * 3
        
        # Determine which color to use based on playfield, players, missiles, and ball
        # Start with background color
        color_index = self.registers[self.COLUBK] & 0xFF
        
        # Check playfield
        pf_pixel = self.playfield_pixels[x]
        if pf_pixel:
            if self.playfield["score"]:
                # Score mode: use player colors for playfield
                if x < self.DISPLAY_WIDTH // 2:
                    color_index = self.registers[self.COLUP0] & 0xFF
                else:
                    color_index = self.registers[self.COLUP1] & 0xFF
            else:
                color_index = self.registers[self.COLUPF] & 0xFF
        
        # Check if player/missile/ball pixels are drawn
        player0_pixel = self._is_player_pixel(self.player0, x)
        player1_pixel = self._is_player_pixel(self.player1, x)
        missile0_pixel = self._is_missile_pixel(self.missile0, x)
        missile1_pixel = self._is_missile_pixel(self.missile1, x)
        ball_pixel = self._is_ball_pixel(self.ball, x)
        
        # Apply playfield priority if enabled
        if self.playfield["priority"]:
            if pf_pixel or ball_pixel:
                color_index = self.registers[self.COLUPF] & 0xFF
            elif player0_pixel or missile0_pixel:
                color_index = self.registers[self.COLUP0] & 0xFF
            elif player1_pixel or missile1_pixel:
                color_index = self.registers[self.COLUP1] & 0xFF
        else:
            # Normal priority
            if player0_pixel or missile0_pixel:
                color_index = self.registers[self.COLUP0] & 0xFF
            elif player1_pixel or missile1_pixel:
                color_index = self.registers[self.COLUP1] & 0xFF
            elif ball_pixel:
                color_index = self.registers[self.COLUPF] & 0xFF
        
        # Map color index to RGB
        rgb = self.NTSC_PALETTE[color_index & 0x7F]  # Limit to 128 colors
        
        # Set pixel in frame buffer
        if pixel_index + 2 < len(self.frame_buffer):
            self.frame_buffer[pixel_index] = (rgb >> 16) & 0xFF  # R
            self.frame_buffer[pixel_index + 1] = (rgb >> 8) & 0xFF  # G
            self.frame_buffer[pixel_index + 2] = rgb & 0xFF  # B
    
    def _is_player_pixel(self, player, x):
        """
        Check if a player sprite is drawn at the given X coordinate.
        
        Args:
            player: Player object
            x: X coordinate
            
        Returns:
            True if player pixel should be drawn
        """
        # Handle multiple copies
        for copy in range(player["copies"]):
            # Calculate position of this copy
            copy_offset = copy * 16  # Default offset
            if player["copies"] > 1:
                if copy == 1:
                    copy_offset = 16
                elif copy == 2:
                    copy_offset = 32
            
            pos = (player["position"] + copy_offset) % self.DISPLAY_WIDTH
            
            # Check if x is within range of the player
            relative_x = (x - pos) % self.DISPLAY_WIDTH
            if relative_x >= 0 and relative_x < player["width"] * player["size"]:
                # Determine which bit to check in the graphics register
                bit_pos = relative_x // player["size"]
                if player["reflected"]:
                    bit_pos = 7 - bit_pos
                
                # Check if the bit is set
                if (player["graphics"] >> bit_pos) & 0x01:
                    return True
                    
        return False
    
    def _is_missile_pixel(self, missile, x):
        """
        Check if a missile is drawn at the given X coordinate.
        
        Args:
            missile: Missile object
            x: X coordinate
            
        Returns:
            True if missile pixel should be drawn
        """
        if not missile["enabled"]:
            return False
            
        # Handle multiple copies (based on player copy settings)
        for copy in range(missile["copies"]):
            # Calculate position of this copy
            copy_offset = copy * 16  # Default offset
            if missile["copies"] > 1:
                if copy == 1:
                    copy_offset = 16
                elif copy == 2:
                    copy_offset = 32
            
            pos = (missile["position"] + copy_offset) % self.DISPLAY_WIDTH
            
            # Check if x is within range of the missile
            relative_x = (x - pos) % self.DISPLAY_WIDTH
            if relative_x >= 0 and relative_x < missile["width"] * missile["size"]:
                return True
                
        return False
    
    def _is_ball_pixel(self, ball, x):
        """
        Check if the ball is drawn at the given X coordinate.
        
        Args:
            ball: Ball object
            x: X coordinate
            
        Returns:
            True if ball pixel should be drawn
        """
        if not ball["enabled"]:
            return False
            
        pos = ball["position"]
        
        # Check if x is within range of the ball
        relative_x = (x - pos) % self.DISPLAY_WIDTH
        if relative_x >= 0 and relative_x < ball["width"] * ball["size"]:
            return True
            
        return False
    
    def _check_collisions(self):
        """Update collision registers based on current pixel state."""
        # Only check collisions in visible part of the screen
        if self.hblank or self.section != "VISIBLE":
            return
            
        # Calculate x coordinate (relative to left of visible screen)
        x = self.cycle - self.HBLANK_CLOCKS
        
        # Skip if outside visible area
        if x < 0 or x >= self.DISPLAY_WIDTH:
            return
            
        # Check if objects are drawn at this position
        pf_pixel = self.playfield_pixels[x]
        player0_pixel = self._is_player_pixel(self.player0, x)
        player1_pixel = self._is_player_pixel(self.player1, x)
        missile0_pixel = self._is_missile_pixel(self.missile0, x)
        missile1_pixel = self._is_missile_pixel(self.missile1, x)
        ball_pixel = self._is_ball_pixel(self.ball, x)
        
        # Update collision registers
        # M0-P1, M0-P0
        if missile0_pixel:
            if player0_pixel:
                self.collisions[self.CXM0P] |= 0x40
            if player1_pixel:
                self.collisions[self.CXM0P] |= 0x80
        
        # M1-P0, M1-P1
        if missile1_pixel:
            if player0_pixel:
                self.collisions[self.CXM1P] |= 0x80
            if player1_pixel:
                self.collisions[self.CXM1P] |= 0x40
        
        # P0-PF, P0-BL
        if player0_pixel:
            if pf_pixel:
                self.collisions[self.CXP0FB] |= 0x80
            if ball_pixel:
                self.collisions[self.CXP0FB] |= 0x40
        
        # P1-PF, P1-BL
        if player1_pixel:
            if pf_pixel:
                self.collisions[self.CXP1FB] |= 0x80
            if ball_pixel:
                self.collisions[self.CXP1FB] |= 0x40
        
        # M0-PF, M0-BL
        if missile0_pixel:
            if pf_pixel:
                self.collisions[self.CXM0FB] |= 0x80
            if ball_pixel:
                self.collisions[self.CXM0FB] |= 0x40
        
        # M1-PF, M1-BL
        if missile1_pixel:
            if pf_pixel:
                self.collisions[self.CXM1FB] |= 0x80
            if ball_pixel:
                self.collisions[self.CXM1FB] |= 0x40
        
        # BL-PF
        if ball_pixel and pf_pixel:
            self.collisions[self.CXBLPF] |= 0x80
        
        # P0-P1, M0-M1
        if player0_pixel and player1_pixel:
            self.collisions[self.CXPPMM] |= 0x80
        if missile0_pixel and missile1_pixel:
            self.collisions[self.CXPPMM] |= 0x40
    
    def _update_audio(self):
        """Update audio channels."""
        # Process each audio channel
        for i, channel in enumerate(self.audio_channels):
            # Skip if volume is 0
            if channel["volume"] == 0:
                continue
                
            # Update divider
            channel["divider"] -= 1
            if channel["divider"] <= 0:
                # Reset divider based on frequency
                channel["divider"] = channel["freq"] + 1
                
                # Update output based on control type
                control = channel["control"]
                if control == 0:
                    # Set to 1
                    channel["output"] = 1
                elif control == 1:
                    # 4-bit poly (divide by 15)
                    channel["output"] = (channel["output"] >> 1) ^ (-(channel["output"] & 1) & 0xB400)
                elif control == 2:
                    # 4-bit poly (divide by 15) but only for values equal to div 15
                    channel["output"] = (channel["output"] >> 1) ^ (-(channel["output"] & 1) & 0xB400)
                elif control == 3:
                    # 5-bit poly sequence
                    channel["output"] = (channel["output"] >> 1) ^ (-(channel["output"] & 1) & 0x12000)
                elif control == 4:
                    # Pure tone (divide by 2)
                    channel["output"] = channel["output"] ^ 1
                elif control == 5:
                    # Pure tone (divide by 2), and then div by 15 on 4-bit poly
                    channel["output"] = channel["output"] ^ 1
                elif control == 6:
                    # Pure tone (divide by 31)
                    channel["output"] = (channel["output"] + 1) % 31
                elif control == 7:
                    # 5-bit poly sequence with pure tone
                    channel["output"] = ((channel["output"] >> 1) ^ (-(channel["output"] & 1) & 0x12000)) | 1
                elif control == 8:
                    # 9-bit poly sequence
                    channel["output"] = (channel["output"] >> 1) ^ (-(channel["output"] & 1) & 0x10004)
                elif control == 9:
                    # 5-bit poly sequence
                    channel["output"] = (channel["output"] >> 1) ^ (-(channel["output"] & 1) & 0x12000)
                elif control == 10:
                    # Pure tone div by 2, and then combined with 5-bit poly (no poly)
                    channel["output"] = channel["output"] ^ 1
                elif control == 11:
                    # Pure tone (divide by 2)
                    channel["output"] = channel["output"] ^ 1
                elif control == 12:
                    # Pure tone (divide by 2)
                    channel["output"] = channel["output"] ^ 1
                elif control == 13:
                    # Pure tone (divide by 2), and then div by 15 on 4-bit poly
                    channel["output"] = channel["output"] ^ 1
                elif control == 14:
                    # Pure tone (divide by 2), and then combined with 5-bit poly
                    channel["output"] = channel["output"] ^ 1
                elif control == 15:
                    # 5-bit poly sequence with pure tone
                    channel["output"] = ((channel["output"] >> 1) ^ (-(channel["output"] & 1) & 0x12000)) | 1
    
    def get_frame_buffer(self) -> bytes:
        """
        Get the current frame buffer.
        
        Returns:
            Bytes object containing the frame buffer data
        """
        return bytes(self.frame_buffer)
    
    def get_state(self) -> dict:
        """
        Get the current TIA state.
        
        Returns:
            Dictionary containing TIA state
        """
        return {
            "cycle": self.cycle,
            "scanline": self.scanline,
            "frame": self.frame,
            "section": self.section,
            "hblank": self.hblank,
            "wsync_halt": self.wsync_halt,
            "player0": {
                "position": self.player0["position"],
                "graphics": self.player0["graphics"]
            },
            "player1": {
                "position": self.player1["position"],
                "graphics": self.player1["graphics"]
            },
            "playfield": {
                "pf0": self.playfield["pf0"],
                "pf1": self.playfield["pf1"],
                "pf2": self.playfield["pf2"]
            },
            "collisions": dict(self.collisions),
            "registers": {
                "VSYNC": self.registers[self.VSYNC],
                "VBLANK": self.registers[self.VBLANK],
                "COLUBK": self.registers[self.COLUBK],
                "COLUPF": self.registers[self.COLUPF],
                "COLUP0": self.registers[self.COLUP0],
                "COLUP1": self.registers[self.COLUP1]
            }
        }