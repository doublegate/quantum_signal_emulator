"""
Atari 2600 cartridge handling module.

The Atari 2600 used various cartridge formats with different bankswitching
schemes to overcome the 4KB ROM limitation of the 6507 CPU. This module
provides cartridge detection and handling for various bankswitching formats
including standard 2K/4K carts and advanced formats like F8, F6, F4, etc.
"""

import logging
from typing import Dict, List, Tuple, Optional, Any, Union, BinaryIO
import os

logger = logging.getLogger("QuantumSignalEmulator.Atari2600.Cartridge")

# Cartridge types and their ROM sizes
CART_TYPES = {
    "2K": 2048,        # 2K carts, no bankswitching
    "4K": 4096,        # 4K carts, no bankswitching
    "F8": 8192,        # 8K carts, 2 banks of 4K each
    "F6": 16384,       # 16K carts, 4 banks of 4K each
    "F4": 32768,       # 32K carts, 8 banks of 4K each
    "FA": 12288,       # 12K carts, 3 banks of 4K each
    "FE": 8192,        # 8K carts, different bankswitching than F8
    "E0": 8192,        # 8K carts, 8 banks of 1K each
    "3F": 8192,        # 8K carts, 2 banks of 4K each (different switching than F8)
    "DPC": 8192,       # 8K + custom DPC chip
    "AR": 6144,        # 6K + 2K RAM Superchip
    "SC": 8192,        # 8K + 128 bytes onboard RAM
    "CV": 2048,        # 2K Commavid cart with RAM
    "UA": 8192,        # 8K UA type
    "CUSTOM": 0        # Custom/unknown mapping
}

def detect_cartridge_type(rom_data: bytes) -> str:
    """
    Detect the cartridge type based on the ROM data.
    
    Args:
        rom_data: ROM data as bytes
        
    Returns:
        String identifier for the cartridge type
    """
    rom_size = len(rom_data)
    
    # First, check standard sizes
    if rom_size <= 2048:
        return "2K"
    elif rom_size <= 4096:
        return "4K"
    
    # Check for special cart types based on ROM size
    for cart_type, size in CART_TYPES.items():
        if rom_size == size and cart_type != "2K" and cart_type != "4K":
            # Now we need to further differentiate between types with the same size
            if cart_type == "F8" and rom_size == 8192:
                # Check for markers in the ROM that indicate bankswitching type
                # F8 has JMP instructions at addresses 0x1FF8 and 0x1FF9
                if _check_jmp_vectors(rom_data, 0x1FF8) and _check_jmp_vectors(rom_data, 0x1FF9):
                    return "F8"
                
                # 3F bankswitching check
                # Look for bank switching code patterns
                if _check_3f_bankswitching(rom_data):
                    return "3F"
                
                # Default to F8 for 8K ROMs if no better match
                return "F8"
                
            elif cart_type == "F6" and rom_size == 16384:
                # F6 has JMP instructions at addresses 0x1FF6-0x1FF9
                if (_check_jmp_vectors(rom_data, 0x1FF6) and 
                    _check_jmp_vectors(rom_data, 0x1FF7) and
                    _check_jmp_vectors(rom_data, 0x1FF8) and
                    _check_jmp_vectors(rom_data, 0x1FF9)):
                    return "F6"
                
                # Default to F6 for 16K ROMs
                return "F6"
                
            elif cart_type == "F4" and rom_size == 32768:
                # F4 has JMP instructions at addresses 0x1FF4-0x1FFB
                return "F4"
                
            # Add more detection for other types
            
    # If no special type detected, use defaults based on size
    if rom_size == 8192:
        return "F8"  # Assume F8 for 8K ROMs
    elif rom_size == 16384:
        return "F6"  # Assume F6 for 16K ROMs
    elif rom_size == 32768:
        return "F4"  # Assume F4 for 32K ROMs
    elif rom_size == 12288:
        return "FA"  # Assume FA for 12K ROMs
    
    # Default for unknown sizes
    return "CUSTOM"

def _check_jmp_vectors(rom_data: bytes, address: int) -> bool:
    """
    Check if there's a JMP instruction at the given address.
    
    Args:
        rom_data: ROM data as bytes
        address: Address to check
        
    Returns:
        True if a JMP instruction exists at the address
    """
    # JMP absolute is 0x4C in 6502
    rel_addr = address & 0x0FFF  # Map to 4K bank
    if len(rom_data) > rel_addr:
        return rom_data[rel_addr] == 0x4C
    return False

def _check_3f_bankswitching(rom_data: bytes) -> bool:
    """
    Check for 3F bankswitching patterns.
    
    Args:
        rom_data: ROM data as bytes
        
    Returns:
        True if 3F bankswitching is detected
    """
    # Look for characteristic 3F bankswitching code
    # This is a simplified version, real detection would be more complex
    pattern = bytes([0xA9, 0x3F, 0x8D])  # LDA #$3F, STA ...
    return pattern in rom_data

class CartridgeFactory:
    """Factory class for creating cartridge handlers."""
    
    @staticmethod
    def create_cartridge(rom_data: bytes) -> 'Cartridge':
        """
        Create a cartridge handler for the ROM data.
        
        Args:
            rom_data: ROM data as bytes
            
        Returns:
            Cartridge handler instance
        """
        cart_type = detect_cartridge_type(rom_data)
        
        if cart_type == "2K" or cart_type == "4K":
            return StandardCartridge(rom_data, cart_type)
        elif cart_type == "F8":
            return F8Cartridge(rom_data)
        elif cart_type == "F6":
            return F6Cartridge(rom_data)
        elif cart_type == "F4":
            return F4Cartridge(rom_data)
        elif cart_type == "FA":
            return FACartridge(rom_data)
        # Add more cartridge types as needed
        
        # Default to standard cartridge
        return StandardCartridge(rom_data, cart_type)

class Cartridge:
    """Base class for Atari 2600 cartridges."""
    
    def __init__(self, rom_data: bytes, cart_type: str):
        """
        Initialize the cartridge.
        
        Args:
            rom_data: ROM data as bytes
            cart_type: Cartridge type identifier
        """
        self.rom_data = rom_data
        self.cart_type = cart_type
        self.rom_size = len(rom_data)
        
        # For bankswitched cartridges
        self.current_bank = 0
        self.banks = []
        
        logger.info(f"Created {cart_type} cartridge, {self.rom_size} bytes")
    
    def read(self, address: int) -> int:
        """
        Read a byte from the cartridge.
        
        Args:
            address: Address to read (0x1000-0x1FFF)
            
        Returns:
            Byte value from ROM
        """
        # Default implementation, override in subclasses
        addr = address & 0x0FFF  # Map to 4K ROM space
        if addr < self.rom_size:
            return self.rom_data[addr]
        return 0
    
    def write(self, address: int, value: int) -> None:
        """
        Write a byte to the cartridge (used for bankswitching).
        
        Args:
            address: Address to write
            value: Value to write
        """
        # Default implementation (most carts are read-only)
        pass
    
    def reset(self) -> None:
        """Reset the cartridge state."""
        self.current_bank = 0

class StandardCartridge(Cartridge):
    """Standard 2K or 4K cartridge with no bankswitching."""
    
    def read(self, address: int) -> int:
        """
        Read a byte from the cartridge.
        
        Args:
            address: Address to read (0x1000-0x1FFF)
            
        Returns:
            Byte value from ROM
        """
        addr = address & 0x0FFF  # Map to 4K ROM space
        
        # For 2K cartridges, mirror the ROM
        if self.cart_type == "2K":
            addr &= 0x07FF  # Mask to 2K
        
        if addr < self.rom_size:
            return self.rom_data[addr]
        return 0

class F8Cartridge(Cartridge):
    """F8 bankswitched cartridge (8K with 2 banks of 4K each)."""
    
    def __init__(self, rom_data: bytes):
        """
        Initialize the F8 cartridge.
        
        Args:
            rom_data: ROM data as bytes
        """
        super().__init__(rom_data, "F8")
        
        # Create 4K banks
        self.bank_size = 4096
        self.num_banks = self.rom_size // self.bank_size
        
        self.banks = []
        for i in range(self.num_banks):
            start = i * self.bank_size
            end = start + self.bank_size
            self.banks.append(self.rom_data[start:end])
    
    def read(self, address: int) -> int:
        """
        Read a byte from the cartridge.
        
        Args:
            address: Address to read (0x1000-0x1FFF)
            
        Returns:
            Byte value from ROM
        """
        # Check for bankswitching hotspots
        if address == 0x1FF8:
            self.current_bank = 0
            logger.debug("Switched to bank 0")
        elif address == 0x1FF9:
            self.current_bank = 1
            logger.debug("Switched to bank 1")
            
        # Access current bank
        addr = address & 0x0FFF  # Map to 4K ROM space
        if self.current_bank < len(self.banks) and addr < self.bank_size:
            return self.banks[self.current_bank][addr]
        return 0
    
    def write(self, address: int, value: int) -> None:
        """
        Write a byte to the cartridge (used for bankswitching).
        
        Args:
            address: Address to write
            value: Value to write
        """
        # F8 bankswitching occurs on writes to specific addresses
        if address == 0x1FF8:
            self.current_bank = 0
            logger.debug("Switched to bank 0")
        elif address == 0x1FF9:
            self.current_bank = 1
            logger.debug("Switched to bank 1")

class F6Cartridge(Cartridge):
    """F6 bankswitched cartridge (16K with 4 banks of 4K each)."""
    
    def __init__(self, rom_data: bytes):
        """
        Initialize the F6 cartridge.
        
        Args:
            rom_data: ROM data as bytes
        """
        super().__init__(rom_data, "F6")
        
        # Create 4K banks
        self.bank_size = 4096
        self.num_banks = self.rom_size // self.bank_size
        
        self.banks = []
        for i in range(self.num_banks):
            start = i * self.bank_size
            end = start + self.bank_size
            self.banks.append(self.rom_data[start:end])
    
    def read(self, address: int) -> int:
        """
        Read a byte from the cartridge.
        
        Args:
            address: Address to read (0x1000-0x1FFF)
            
        Returns:
            Byte value from ROM
        """
        # Check for bankswitching hotspots
        if address == 0x1FF6:
            self.current_bank = 0
        elif address == 0x1FF7:
            self.current_bank = 1
        elif address == 0x1FF8:
            self.current_bank = 2
        elif address == 0x1FF9:
            self.current_bank = 3
            
        # Access current bank
        addr = address & 0x0FFF  # Map to 4K ROM space
        if self.current_bank < len(self.banks) and addr < self.bank_size:
            return self.banks[self.current_bank][addr]
        return 0
    
    def write(self, address: int, value: int) -> None:
        """
        Write a byte to the cartridge (used for bankswitching).
        
        Args:
            address: Address to write
            value: Value to write
        """
        # F6 bankswitching occurs on writes to specific addresses
        if address == 0x1FF6:
            self.current_bank = 0
            logger.debug("Switched to bank 0")
        elif address == 0x1FF7:
            self.current_bank = 1
            logger.debug("Switched to bank 1")
        elif address == 0x1FF8:
            self.current_bank = 2
            logger.debug("Switched to bank 2")
        elif address == 0x1FF9:
            self.current_bank = 3
            logger.debug("Switched to bank 3")

class F4Cartridge(Cartridge):
    """F4 bankswitched cartridge (32K with 8 banks of 4K each)."""
    
    def __init__(self, rom_data: bytes):
        """
        Initialize the F4 cartridge.
        
        Args:
            rom_data: ROM data as bytes
        """
        super().__init__(rom_data, "F4")
        
        # Create 4K banks
        self.bank_size = 4096
        self.num_banks = self.rom_size // self.bank_size
        
        self.banks = []
        for i in range(self.num_banks):
            start = i * self.bank_size
            end = start + self.bank_size
            self.banks.append(self.rom_data[start:end])
    
    def read(self, address: int) -> int:
        """
        Read a byte from the cartridge.
        
        Args:
            address: Address to read (0x1000-0x1FFF)
            
        Returns:
            Byte value from ROM
        """
        # Check for bankswitching hotspots
        if address >= 0x1FF4 and address <= 0x1FFB:
            self.current_bank = address - 0x1FF4
            logger.debug(f"Switched to bank {self.current_bank}")
            
        # Access current bank
        addr = address & 0x0FFF  # Map to 4K ROM space
        if self.current_bank < len(self.banks) and addr < self.bank_size:
            return self.banks[self.current_bank][addr]
        return 0
    
    def write(self, address: int, value: int) -> None:
        """
        Write a byte to the cartridge (used for bankswitching).
        
        Args:
            address: Address to write
            value: Value to write
        """
        # F4 bankswitching occurs on writes to specific addresses
        if address >= 0x1FF4 and address <= 0x1FFB:
            self.current_bank = address - 0x1FF4
            logger.debug(f"Switched to bank {self.current_bank}")

class FACartridge(Cartridge):
    """FA bankswitched cartridge (12K with 3 banks of 4K each)."""
    
    def __init__(self, rom_data: bytes):
        """
        Initialize the FA cartridge.
        
        Args:
            rom_data: ROM data as bytes
        """
        super().__init__(rom_data, "FA")
        
        # Create 4K banks
        self.bank_size = 4096
        self.num_banks = self.rom_size // self.bank_size
        
        self.banks = []
        for i in range(self.num_banks):
            start = i * self.bank_size
            end = start + self.bank_size
            self.banks.append(self.rom_data[start:end])
    
    def read(self, address: int) -> int:
        """
        Read a byte from the cartridge.
        
        Args:
            address: Address to read (0x1000-0x1FFF)
            
        Returns:
            Byte value from ROM
        """
        # Check for bankswitching hotspots
        if address == 0x1FF8:
            self.current_bank = 0
        elif address == 0x1FF9:
            self.current_bank = 1
        elif address == 0x1FFA:
            self.current_bank = 2
            
        # Access current bank
        addr = address & 0x0FFF  # Map to 4K ROM space
        if self.current_bank < len(self.banks) and addr < self.bank_size:
            return self.banks[self.current_bank][addr]
        return 0
    
    def write(self, address: int, value: int) -> None:
        """
        Write a byte to the cartridge (used for bankswitching).
        
        Args:
            address: Address to write
            value: Value to write
        """
        # FA bankswitching occurs on writes to specific addresses
        if address == 0x1FF8:
            self.current_bank = 0
            logger.debug("Switched to bank 0")
        elif address == 0x1FF9:
            self.current_bank = 1
            logger.debug("Switched to bank 1")
        elif address == 0x1FFA:
            self.current_bank = 2
            logger.debug("Switched to bank 2")