"""
Atari 2600 memory system implementation.

The Atari 2600 has a very limited memory map:
- 128 bytes of RAM (addresses 0x80-0xFF)
- TIA registers (addresses 0x00-0x7F, mirrored)
- RIOT registers (addresses 0x280-0x29F, mirrored)
- Cartridge ROM (addresses 0x1000-0x1FFF)

This module provides the memory implementation that connects all components
and maps them to their respective address ranges.
"""

from ...common.interfaces import Memory
import numpy as np
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger("QuantumSignalEmulator.Atari2600.Memory")

class AtariMemory(Memory):
    """
    Emulates the Atari 2600 memory system.
    
    Provides access to RAM, TIA registers, RIOT registers, and cartridge ROM
    according to the Atari 2600 memory map. All components are connected through
    this central memory interface.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the memory system.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # RAM (128 bytes)
        self.ram = bytearray(128)
        
        # ROM (up to 4KB, but can be larger with bankswitching)
        self.rom = None
        self.rom_size = 0
        self.cart_type = "2K"  # Default is 2K cart
        
        # Connected components
        self.tia = None
        self.riot = None
        
        logger.info("Atari 2600 memory system initialized")
    
    def connect_tia(self, tia) -> None:
        """
        Connect the TIA component.
        
        Args:
            tia: TIA component
        """
        self.tia = tia
        logger.debug("TIA connected to memory system")
    
    def connect_riot(self, riot) -> None:
        """
        Connect the RIOT component.
        
        Args:
            riot: RIOT component
        """
        self.riot = riot
        logger.debug("RIOT connected to memory system")
    
    def read(self, address: int) -> int:
        """
        Read a byte from the specified address.
        
        Args:
            address: Memory address
            
        Returns:
            Byte value at address
        """
        # Mask address to 13 bits (6507 has a 13-bit address bus)
        address &= 0x1FFF
        
        # TIA registers (0x00-0x7F, mirrored throughout the first 1K)
        if address < 0x80 or (address & 0x1080) == 0x0000:
            if self.tia:
                return self.tia.read_register(address & 0x3F)
            return 0
            
        # RAM (0x80-0xFF, mirrored several times)
        elif (address & 0x1080) == 0x0080:
            return self.ram[address & 0x7F]
            
        # RIOT registers (0x280-0x29F, mirrored)
        elif (address & 0x1080) == 0x0280:
            if self.riot:
                return self.riot.read_register(address & 0x1F)
            return 0
            
        # Cartridge ROM (0x1000-0x1FFF)
        elif address >= 0x1000:
            if self.rom:
                # Handle different cartridge types and bankswitching
                if self.cart_type == "2K":
                    # 2K cart, ROM is mirrored throughout 0x1000-0x1FFF
                    return self.rom[(address & 0x07FF)]
                elif self.cart_type == "4K":
                    # 4K cart, no mirroring needed
                    return self.rom[(address & 0x0FFF)]
                elif self.cart_type == "F8":
                    # F8 bankswitching (two 4K banks)
                    # Bank selection is done by accessing 0x1FF8/0x1FF9
                    return self.rom[(address & 0x0FFF) + self.current_bank * 0x1000]
                else:
                    # Default case for unsupported cart types
                    return self.rom[address & (self.rom_size - 1)]
            return 0
            
        # Unmapped memory
        return 0
    
    def write(self, address: int, value: int) -> None:
        """
        Write a byte to the specified address.
        
        Args:
            address: Memory address
            value: Byte value to write
        """
        # Mask address to 13 bits (6507 has a 13-bit address bus)
        address &= 0x1FFF
        value &= 0xFF  # Ensure value is a byte
        
        # TIA registers (0x00-0x7F, mirrored throughout the first 1K)
        if address < 0x80 or (address & 0x1080) == 0x0000:
            if self.tia:
                self.tia.write_register(address & 0x3F, value)
                
        # RAM (0x80-0xFF, mirrored several times)
        elif (address & 0x1080) == 0x0080:
            self.ram[address & 0x7F] = value
            
        # RIOT registers (0x280-0x29F, mirrored)
        elif (address & 0x1080) == 0x0280:
            if self.riot:
                self.riot.write_register(address & 0x1F, value)
                
        # Cartridge ROM (0x1000-0x1FFF)
        elif address >= 0x1000:
            # Most writes to ROM space do nothing, but some cart types use
            # writes to certain addresses to trigger bankswitching
            if self.cart_type == "F8":
                if address == 0x1FF8:
                    self.current_bank = 0
                    logger.debug("Switched to ROM bank 0")
                elif address == 0x1FF9:
                    self.current_bank = 1
                    logger.debug("Switched to ROM bank 1")
            elif self.cart_type == "F6":
                if address == 0x1FF6:
                    self.current_bank = 0
                    logger.debug("Switched to ROM bank 0")
                elif address == 0x1FF7:
                    self.current_bank = 1
                    logger.debug("Switched to ROM bank 1")
                elif address == 0x1FF8:
                    self.current_bank = 2
                    logger.debug("Switched to ROM bank 2")
                elif address == 0x1FF9:
                    self.current_bank = 3
                    logger.debug("Switched to ROM bank 3")
            # Add more bankswitching schemes as needed
    
    def load_rom(self, rom_data: bytes) -> None:
        """
        Load a ROM into memory.
        
        Args:
            rom_data: ROM data as bytes
        """
        if not rom_data:
            logger.error("No ROM data provided")
            return
            
        self.rom_size = len(rom_data)
        self.rom = bytearray(rom_data)
        
        # Determine cartridge type based on ROM size
        if self.rom_size <= 2048:
            self.cart_type = "2K"
            logger.info("Loaded 2K cartridge")
        elif self.rom_size <= 4096:
            self.cart_type = "4K"
            logger.info("Loaded 4K cartridge")
        elif self.rom_size <= 8192:
            self.cart_type = "F8"  # Assume F8 bankswitching for 8K ROMs
            self.current_bank = 0
            logger.info("Loaded 8K cartridge with F8 bankswitching")
        elif self.rom_size <= 16384:
            self.cart_type = "F6"  # Assume F6 bankswitching for 16K ROMs
            self.current_bank = 0
            logger.info("Loaded 16K cartridge with F6 bankswitching")
        else:
            logger.warning(f"ROM size {self.rom_size} is unusually large for Atari 2600")
            self.cart_type = "CUSTOM"
            self.current_bank = 0
            
        logger.info(f"ROM loaded: {self.rom_size} bytes, type: {self.cart_type}")
    
    def reset(self) -> None:
        """Reset the memory to initial state."""
        # Clear RAM
        self.ram = bytearray(128)
        
        # Reset bank selection for bankswitched ROMs
        if self.cart_type in ["F8", "F6", "F4", "FA"]:
            self.current_bank = 0
            
        logger.info("Memory system reset")