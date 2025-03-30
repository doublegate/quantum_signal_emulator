"""
Sega Genesis/Mega Drive memory system implementation.

The Genesis has a complex memory map with various regions for the 68000 CPU,
Z80 CPU, VDP, and cartridge ROM/RAM. This module implements the Genesis memory
system with proper interaction between all components.
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Any, Union

logger = logging.getLogger("QuantumSignalEmulator.Genesis.Memory")

class GenesisMemory:
    """
    Implements the Genesis/Mega Drive memory system.
    
    The Genesis has two CPUs (68000 and Z80) with different address spaces and
    memory maps. This implementation manages both memory spaces and handles
    the appropriate routing of memory access to cartridge ROM, RAM, VRAM,
    and hardware registers.
    """
    
    # Memory size constants
    M68K_RAM_SIZE = 64 * 1024    # 64KB of 68000 RAM
    Z80_RAM_SIZE = 8 * 1024      # 8KB of Z80 RAM
    VRAM_SIZE = 64 * 1024        # 64KB of Video RAM
    CRAM_SIZE = 128              # 128 bytes of Color RAM (64 entries of 9-bit color)
    VSRAM_SIZE = 80              # 80 bytes of Vertical Scroll RAM
    ROM_MAX_SIZE = 4 * 1024 * 1024  # 4MB maximum ROM size
    SRAM_SIZE = 32 * 1024        # 32KB of cartridge SRAM
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Genesis memory system.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # Initialize memory arrays
        self.m68k_ram = bytearray(self.M68K_RAM_SIZE)
        self.z80_ram = bytearray(self.Z80_RAM_SIZE)
        self.sram = bytearray(self.SRAM_SIZE)
        
        # ROM data
        self.rom = None
        self.rom_size = 0
        
        # SRAM-related flags
        self.sram_enabled = False
        self.sram_write_enabled = False
        
        # Z80 control flags
        self.z80_reset = True
        self.z80_busreq = True
        
        # Z80 address space access from 68000
        self.z80_bank_address = 0
        
        # Connected components
        self.vdp = None
        self.fm_sound = None
        
        logger.info("Genesis memory system initialized")
    
    def connect_vdp(self, vdp) -> None:
        """
        Connect the VDP component.
        
        Args:
            vdp: VDP component
        """
        self.vdp = vdp
        logger.debug("VDP connected to memory system")
    
    def connect_fm_sound(self, fm_sound) -> None:
        """
        Connect the FM sound component.
        
        Args:
            fm_sound: FM sound component
        """
        self.fm_sound = fm_sound
        logger.debug("FM sound connected to memory system")
    
    def m68k_read(self, address: int) -> int:
        """
        Read a byte from the 68000 address space.
        
        Args:
            address: 24-bit address
            
        Returns:
            Byte value at the address
        """
        # Mask address to 24 bits
        address &= 0xFFFFFF
        
        # ROM: 0x000000-0x3FFFFF
        if address < 0x400000:
            if self.rom and address < self.rom_size:
                return self.rom[address]
            return 0
            
        # SRAM: 0x400000-0x4FFFFF (if enabled)
        elif address < 0x500000 and self.sram_enabled:
            sram_addr = address - 0x400000
            if sram_addr < self.SRAM_SIZE:
                return self.sram[sram_addr]
            return 0
            
        # Reserved/Unused: 0x500000-0x9FFFFF
        elif address < 0xA00000:
            return 0
            
        # Z80 address space: 0xA00000-0xA0FFFF
        elif address < 0xA10000:
            if not self.z80_busreq:
                # Z80 has bus, 68000 cannot access
                return 0xFF
                
            z80_addr = address & 0xFFFF
            
            # Z80 RAM: 0x0000-0x1FFF
            if z80_addr < 0x2000:
                return self.z80_ram[z80_addr]
                
            # YM2612 (FM): 0x4000-0x4003
            elif 0x4000 <= z80_addr <= 0x4003 and self.fm_sound:
                return self.fm_sound.read_register(z80_addr - 0x4000)
                
            return 0
            
        # 68000 RAM: 0xC00000-0xCFFFFF (mirrored every 64KB)
        elif 0xE00000 <= address < 0xF00000:
            ram_addr = address & 0xFFFF  # 64KB mirror
            return self.m68k_ram[ram_addr]
            
        # VDP: 0xC00000-0xC0001F
        elif 0xC00000 <= address < 0xC00020:
            if self.vdp:
                return self.vdp.read_register(address & 0x1F)
            return 0
            
        # Misc hardware registers
        elif 0xA10000 <= address < 0xA10100:
            # Handle control port, I/O ports, etc.
            if address == 0xA10000:
                # Version register
                return 0x00  # NTSC 
            elif address == 0xA10001:
                # Version register
                return 0x00  # NTSC overseas
            elif address == 0xA11100:
                # Z80 BUSREQ
                return 0x01 if self.z80_busreq else 0x00
            elif address == 0xA11200:
                # Z80 RESET
                return 0x01 if self.z80_reset else 0x00
            return 0
        
        # Default for unmapped addresses
        return 0
    
    def m68k_write(self, address: int, value: int) -> None:
        """
        Write a byte to the 68000 address space.
        
        Args:
            address: 24-bit address
            value: Byte value to write
        """
        # Mask address to 24 bits and value to 8 bits
        address &= 0xFFFFFF
        value &= 0xFF
        
        # SRAM: 0x400000-0x4FFFFF (if enabled and writable)
        if address < 0x500000 and self.sram_enabled and self.sram_write_enabled:
            sram_addr = address - 0x400000
            if sram_addr < self.SRAM_SIZE:
                self.sram[sram_addr] = value
                
        # Z80 address space: 0xA00000-0xA0FFFF
        elif address < 0xA10000:
            if not self.z80_busreq:
                # Z80 has bus, 68000 cannot access
                return
                
            z80_addr = address & 0xFFFF
            
            # Z80 RAM: 0x0000-0x1FFF
            if z80_addr < 0x2000:
                self.z80_ram[z80_addr] = value
                
            # YM2612 (FM): 0x4000-0x4003
            elif 0x4000 <= z80_addr <= 0x4003 and self.fm_sound:
                self.fm_sound.write_register(z80_addr - 0x4000, value)
                
        # 68000 RAM: 0xE00000-0xEFFFFF (mirrored every 64KB)
        elif 0xE00000 <= address < 0xF00000:
            ram_addr = address & 0xFFFF  # 64KB mirror
            self.m68k_ram[ram_addr] = value
            
        # VDP: 0xC00000-0xC0001F
        elif 0xC00000 <= address < 0xC00020:
            if self.vdp:
                self.vdp.write_register(address & 0x1F, value)
                
        # Misc hardware registers
        elif 0xA10000 <= address < 0xA10100:
            # Handle control port, I/O ports, etc.
            if address == 0xA11100:
                # Z80 BUSREQ (0=request bus, 1=release bus)
                self.z80_busreq = (value & 0x01) != 0
            elif address == 0xA11200:
                # Z80 RESET (0=assert reset, 1=release reset)
                self.z80_reset = (value & 0x01) != 0
    
    def z80_read(self, address: int) -> int:
        """
        Read a byte from the Z80 address space.
        
        Args:
            address: 16-bit address
            
        Returns:
            Byte value at the address
        """
        # Mask address to 16 bits
        address &= 0xFFFF
        
        # Z80 RAM: 0x0000-0x1FFF
        if address < 0x2000:
            return self.z80_ram[address]
            
        # YM2612 (FM): 0x4000-0x4003
        elif 0x4000 <= address <= 0x4003 and self.fm_sound:
            return self.fm_sound.read_register(address - 0x4000)
            
        # 68000 bank address space: 0x8000-0xFFFF
        elif address >= 0x8000:
            # Access through bank window to 68000 memory
            m68k_addr = self.z80_bank_address + (address - 0x8000)
            return self.m68k_read(m68k_addr)
            
        # Default for unmapped addresses
        return 0xFF
    
    def z80_write(self, address: int, value: int) -> None:
        """
        Write a byte to the Z80 address space.
        
        Args:
            address: 16-bit address
            value: Byte value to write
        """
        # Mask address to 16 bits and value to 8 bits
        address &= 0xFFFF
        value &= 0xFF
        
        # Z80 RAM: 0x0000-0x1FFF
        if address < 0x2000:
            self.z80_ram[address] = value
            
        # YM2612 (FM): 0x4000-0x4003
        elif 0x4000 <= address <= 0x4003 and self.fm_sound:
            self.fm_sound.write_register(address - 0x4000, value)
            
        # Bank register: 0x6000-0x60FF
        elif 0x6000 <= address <= 0x60FF:
            # Set bank register (shifted up by 15 bits, 9 bits used)
            bank_bits = (value & 0x1F)  # Only 5 lower bits used
            self.z80_bank_address = (bank_bits << 15) & 0x0F8000
            
        # 68000 bank address space: 0x8000-0xFFFF
        elif address >= 0x8000:
            # Access through bank window to 68000 memory
            m68k_addr = self.z80_bank_address + (address - 0x8000)
            self.m68k_write(m68k_addr, value)
    
    def load_rom(self, rom_data: bytes) -> None:
        """
        Load ROM data.
        
        Args:
            rom_data: ROM data as bytes
        """
        if not rom_data:
            logger.error("No ROM data provided")
            return
            
        # Store ROM data
        self.rom = bytearray(rom_data)
        self.rom_size = len(rom_data)
        
        logger.info(f"ROM loaded: {self.rom_size} bytes")
    
    def reset(self) -> None:
        """Reset the memory system."""
        # Clear RAM
        self.m68k_ram = bytearray(self.M68K_RAM_SIZE)
        self.z80_ram = bytearray(self.Z80_RAM_SIZE)
        
        # Reset control flags
        self.z80_reset = True
        self.z80_busreq = True
        self.z80_bank_address = 0
        
        # Reset SRAM flags
        self.sram_enabled = False
        self.sram_write_enabled = False
        
        logger.info("Memory system reset")