"""
Sega Genesis/Mega Drive cartridge handling module.

The Genesis/Mega Drive cartridge formats vary in mapping and features.
This module provides cartridge detection, ROM loading, and memory mapping
for different cartridge types including standard, SRAM, and mappers.
"""

import logging
from typing import Dict, List, Tuple, Optional, Any, Union, BinaryIO
import os

logger = logging.getLogger("QuantumSignalEmulator.Genesis.Cartridge")

# Genesis cartridge types
CART_TYPES = {
    "ROM": 0,            # Standard ROM
    "ROM_SRAM": 1,       # ROM with battery-backed SRAM
    "ROM_EEPROM": 2,     # ROM with EEPROM
    "SSFF2": 3,          # Super Street Fighter II mapper
    "BANKSWITCH": 4,     # ROM with bankswitch
    "ROM_SVP": 5,        # ROM with SVP (Virtua Racing)
    "ROM_MAPPER": 6,     # ROM with generic mapper
}

class GenesisCartridge:
    """
    Handles Sega Genesis/Mega Drive cartridge operations.
    
    Provides cartridge type detection, ROM loading, memory mapping,
    and SRAM handling for Genesis/Mega Drive games.
    """
    
    def __init__(self):
        """Initialize the cartridge handler."""
        # ROM data
        self.rom = None
        self.rom_size = 0
        
        # SRAM data
        self.sram = bytearray(32 * 1024)  # 32KB SRAM
        self.sram_size = 0
        self.sram_start = 0x200000  # Default SRAM start address
        self.sram_end = 0x20FFFF    # Default SRAM end address
        
        # Cartridge type
        self.cart_type = CART_TYPES["ROM"]
        
        # Header information
        self.header = {}
        
        # Bank switching
        self.current_bank = 0
        self.bank_registers = [0] * 8  # Up to 8 bank registers
        
        # Flags
        self.sram_enabled = False
        self.sram_write_enabled = False
        
        logger.info("Genesis cartridge handler initialized")
    
    def load_rom(self, rom_data: bytes) -> bool:
        """
        Load ROM data and detect cartridge type.
        
        Args:
            rom_data: ROM data as bytes
            
        Returns:
            True if ROM loaded successfully
        """
        if not rom_data:
            logger.error("No ROM data provided")
            return False
            
        self.rom = bytearray(rom_data)
        self.rom_size = len(rom_data)
        
        # Parse ROM header
        if not self._parse_header():
            logger.warning("Could not parse ROM header, using default settings")
            
        # Detect cartridge type
        self._detect_cart_type()
        
        # Set up SRAM if present
        if self.cart_type in [CART_TYPES["ROM_SRAM"], CART_TYPES["ROM_EEPROM"]]:
            self._setup_sram()
        
        logger.info(f"ROM loaded: {self.rom_size} bytes, type: {self._get_cart_type_name()}")
        
        return True
    
    def _parse_header(self) -> bool:
        """
        Parse the ROM header at offset 0x100-0x1FF.
        
        Returns:
            True if header parsed successfully
        """
        if self.rom_size < 0x200:
            return False
            
        # Extract system and copyright info
        system = self._read_string(0x100, 16)
        copyright = self._read_string(0x110, 16)
        
        # Extract domestic and international names
        domestic_name = self._read_string(0x120, 48)
        int_name = self._read_string(0x150, 48)
        
        # Extract serial number and checksum
        serial = self._read_string(0x180, 14)
        checksum = (self.rom[0x18E] << 8) | self.rom[0x18F]
        
        # Extract device support and ROM address range
        device = self._read_string(0x190, 16)
        rom_start = (self.rom[0x1A0] << 24) | (self.rom[0x1A1] << 16) | (self.rom[0x1A2] << 8) | self.rom[0x1A3]
        rom_end = (self.rom[0x1A4] << 24) | (self.rom[0x1A5] << 16) | (self.rom[0x1A6] << 8) | self.rom[0x1A7]
        
        # Extract RAM address range
        ram_start = (self.rom[0x1A8] << 24) | (self.rom[0x1A9] << 16) | (self.rom[0x1AA] << 8) | self.rom[0x1AB]
        ram_end = (self.rom[0x1AC] << 24) | (self.rom[0x1AD] << 16) | (self.rom[0x1AE] << 8) | self.rom[0x1AF]
        
        # Store header information
        self.header = {
            "system": system,
            "copyright": copyright,
            "domestic_name": domestic_name,
            "international_name": int_name,
            "serial": serial,
            "checksum": checksum,
            "device": device,
            "rom_start": rom_start,
            "rom_end": rom_end,
            "ram_start": ram_start,
            "ram_end": ram_end
        }
        
        # Check if header looks valid
        if not (system.startswith("SEGA") or int_name.strip() or domestic_name.strip()):
            logger.warning("ROM header does not contain valid SEGA identifier")
            return False
            
        # If RAM range is specified, update SRAM settings
        if ram_start != 0 and ram_end != 0 and ram_end > ram_start:
            self.sram_start = ram_start
            self.sram_end = ram_end
            self.sram_size = ram_end - ram_start + 1
            
        logger.debug(f"ROM header parsed: {domestic_name}")
        
        return True
    
    def _detect_cart_type(self) -> None:
        """Detect cartridge type based on ROM content and header."""
        # Default to standard ROM
        self.cart_type = CART_TYPES["ROM"]
        
        # Check for SRAM
        if self.header.get("ram_start") and self.header.get("ram_end"):
            self.cart_type = CART_TYPES["ROM_SRAM"]
            
        # Check for specific mappers
        if self._search_string("SUPER STREET FIGHTER2"):
            self.cart_type = CART_TYPES["SSFF2"]
            
        # Check for SVP (Virtua Racing)
        if self._search_string("VIRTUA RACING") or self._search_string("VIRTUARACING"):
            self.cart_type = CART_TYPES["ROM_SVP"]
            
        # Check for generic bankswitching
        if self.rom_size > 4 * 1024 * 1024:  # Larger than 4MB
            self.cart_type = CART_TYPES["BANKSWITCH"]
            
        # Check for games known to use EEPROM
        if self._search_string("NBA JAM") or self._search_string("NFL QUARTERBACK CLUB"):
            self.cart_type = CART_TYPES["ROM_EEPROM"]
            
        logger.info(f"Detected cartridge type: {self._get_cart_type_name()}")
    
    def _setup_sram(self) -> None:
        """Set up SRAM based on cartridge type."""
        # Limit SRAM size to reasonable values
        if self.sram_size > 32 * 1024:
            self.sram_size = 32 * 1024
            
        # If SRAM size is still 0, use default 8KB
        if self.sram_size == 0:
            self.sram_size = 8 * 1024
            
        # Create SRAM buffer
        self.sram = bytearray(self.sram_size)
        
        # EEPROM type uses different addressing
        if self.cart_type == CART_TYPES["ROM_EEPROM"]:
            self.sram = bytearray(2 * 1024)  # 2KB for most EEPROM games
            self.sram_size = 2 * 1024
            
        logger.debug(f"SRAM setup: {self.sram_size} bytes at ${self.sram_start:06X}-${self.sram_end:06X}")
    
    def read(self, address: int) -> int:
        """
        Read a byte from cartridge ROM or SRAM.
        
        Args:
            address: 24-bit address
            
        Returns:
            Byte value
        """
        # Check if address is in SRAM range
        if self.sram_enabled and self._is_sram_address(address):
            return self._read_sram(address)
            
        # ROM access
        return self._read_rom(address)
    
    def write(self, address: int, value: int) -> None:
        """
        Write a byte to cartridge SRAM or handle mapper register writes.
        
        Args:
            address: 24-bit address
            value: Byte value to write
        """
        # Check if address is in SRAM range and SRAM writes are enabled
        if self.sram_enabled and self.sram_write_enabled and self._is_sram_address(address):
            self._write_sram(address, value)
            return
            
        # Check if this is a mapper register write
        if self._is_mapper_address(address):
            self._write_mapper_register(address, value)
    
    def _read_rom(self, address: int) -> int:
        """
        Read from ROM with mapping applied.
        
        Args:
            address: 24-bit address
            
        Returns:
            Byte value from ROM
        """
        # Handle different cartridge types
        if self.cart_type == CART_TYPES["BANKSWITCH"]:
            return self._read_rom_bankswitched(address)
        elif self.cart_type == CART_TYPES["SSFF2"]:
            return self._read_rom_ssf2(address)
        elif self.cart_type == CART_TYPES["ROM_SVP"]:
            return self._read_rom_svp(address)
            
        # Standard ROM mapping (no banking)
        rom_addr = address & 0x3FFFFF  # 4MB address space
        
        # Check if address is in range
        if rom_addr < self.rom_size:
            return self.rom[rom_addr]
            
        # For addresses beyond ROM size, mirror or return 0
        if rom_addr < 0x400000:  # Within standard 4MB space
            return self.rom[rom_addr % self.rom_size]
            
        return 0
    
    def _read_rom_bankswitched(self, address: int) -> int:
        """
        Read from bankswitched ROM.
        
        Args:
            address: 24-bit address
            
        Returns:
            Byte value from ROM
        """
        # Simple banking: 2MB fixed + 2MB bankswitched
        if address < 0x200000:
            # First 2MB are fixed
            if address < self.rom_size:
                return self.rom[address]
            else:
                return 0
        elif address < 0x400000:
            # Upper 2MB are bankswitched
            bank_addr = 0x200000 + ((address - 0x200000) | (self.current_bank << 19))
            if bank_addr < self.rom_size:
                return self.rom[bank_addr]
            else:
                return 0
        
        return 0
    
    def _read_rom_ssf2(self, address: int) -> int:
        """
        Read from Super Street Fighter 2 mapper.
        
        Args:
            address: 24-bit address
            
        Returns:
            Byte value from ROM
        """
        # SSF2 uses 8 512KB banks
        bank = (address >> 19) & 7
        offset = address & 0x7FFFF
        
        # Get bank number from registers
        bank_number = self.bank_registers[bank]
        
        # Calculate ROM address
        rom_addr = (bank_number << 19) | offset
        
        if rom_addr < self.rom_size:
            return self.rom[rom_addr]
        else:
            return 0
    
    def _read_rom_svp(self, address: int) -> int:
        """
        Read from SVP (Virtua Racing) mapper.
        
        Args:
            address: 24-bit address
            
        Returns:
            Byte value from ROM or SVP chip
        """
        # SVP chip is not fully implemented
        # For now, just do standard ROM mapping
        rom_addr = address & 0x3FFFFF
        
        if rom_addr < self.rom_size:
            return self.rom[rom_addr]
        else:
            return 0
    
    def _read_sram(self, address: int) -> int:
        """
        Read from SRAM.
        
        Args:
            address: 24-bit address
            
        Returns:
            Byte value from SRAM
        """
        # Calculate SRAM address
        sram_addr = address - self.sram_start
        
        # Check if address is in range
        if 0 <= sram_addr < self.sram_size:
            return self.sram[sram_addr]
        
        return 0
    
    def _write_sram(self, address: int, value: int) -> None:
        """
        Write to SRAM.
        
        Args:
            address: 24-bit address
            value: Byte value to write
        """
        # Calculate SRAM address
        sram_addr = address - self.sram_start
        
        # Check if address is in range
        if 0 <= sram_addr < self.sram_size:
            self.sram[sram_addr] = value
    
    def _write_mapper_register(self, address: int, value: int) -> None:
        """
        Write to mapper registers.
        
        Args:
            address: 24-bit address
            value: Byte value to write
        """
        # Handle different mapper types
        if self.cart_type == CART_TYPES["BANKSWITCH"]:
            # Simple bankswitch register at 0xA13000-0xA13FFF
            if 0xA13000 <= address <= 0xA13FFF:
                self.current_bank = value & 0x0F
                logger.debug(f"Bank switched to {self.current_bank}")
                
        elif self.cart_type == CART_TYPES["SSFF2"]:
            # SSF2 mapper has registers at 0xA130xx
            if 0xA13000 <= address <= 0xA13020:
                bank = (address - 0xA13000) & 7
                self.bank_registers[bank] = value & 0x0F
                logger.debug(f"SSF2 bank {bank} set to {value & 0x0F}")
                
        elif self.cart_type == CART_TYPES["ROM_SRAM"] or self.cart_type == CART_TYPES["ROM_EEPROM"]:
            # SRAM control register at 0xA130F0-0xA130FF
            if 0xA130F0 <= address <= 0xA130FF:
                self.sram_enabled = (value & 0x01) != 0
                self.sram_write_enabled = (value & 0x02) != 0
                logger.debug(f"SRAM access: enabled={self.sram_enabled}, write={self.sram_write_enabled}")
    
    def _is_sram_address(self, address: int) -> bool:
        """
        Check if an address is in SRAM range.
        
        Args:
            address: 24-bit address
            
        Returns:
            True if address is in SRAM range
        """
        return self.sram_start <= address <= self.sram_end
    
    def _is_mapper_address(self, address: int) -> bool:
        """
        Check if an address is a mapper register.
        
        Args:
            address: 24-bit address
            
        Returns:
            True if address is a mapper register
        """
        # Common mapper register areas
        if 0xA13000 <= address <= 0xA13FFF:
            return True
            
        return False
    
    def _read_string(self, offset: int, length: int) -> str:
        """
        Read a string from ROM.
        
        Args:
            offset: Start offset
            length: String length
            
        Returns:
            String read from ROM
        """
        if offset + length > self.rom_size:
            return ""
            
        # Read bytes and convert to string
        try:
            return bytes(self.rom[offset:offset+length]).decode('ascii', errors='ignore').strip('\x00')
        except Exception:
            return ""
    
    def _search_string(self, search: str) -> bool:
        """
        Search for a string in ROM header.
        
        Args:
            search: String to search for
            
        Returns:
            True if string is found
        """
        # Search in domestic and international names
        if search in self.header.get("domestic_name", "") or search in self.header.get("international_name", ""):
            return True
            
        # Search in first 8KB of ROM
        if self.rom_size >= 8192:
            rom_str = bytes(self.rom[0:8192]).decode('ascii', errors='ignore')
            if search in rom_str:
                return True
                
        return False
    
    def _get_cart_type_name(self) -> str:
        """
        Get the name of the current cartridge type.
        
        Returns:
            Cartridge type name
        """
        for name, value in CART_TYPES.items():
            if value == self.cart_type:
                return name
                
        return "UNKNOWN"
    
    def save_sram(self, filename: str) -> bool:
        """
        Save SRAM to a file.
        
        Args:
            filename: Output filename
            
        Returns:
            True if successful
        """
        if self.cart_type not in [CART_TYPES["ROM_SRAM"], CART_TYPES["ROM_EEPROM"]]:
            logger.warning("No SRAM to save")
            return False
            
        try:
            with open(filename, 'wb') as f:
                f.write(self.sram)
                
            logger.info(f"Saved {self.sram_size} bytes of SRAM to {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save SRAM: {e}")
            return False
    
    def load_sram(self, filename: str) -> bool:
        """
        Load SRAM from a file.
        
        Args:
            filename: Input filename
            
        Returns:
            True if successful
        """
        if self.cart_type not in [CART_TYPES["ROM_SRAM"], CART_TYPES["ROM_EEPROM"]]:
            logger.warning("No SRAM to load")
            return False
            
        try:
            with open(filename, 'rb') as f:
                data = f.read(self.sram_size)
                
            # Copy data to SRAM
            for i in range(min(len(data), self.sram_size)):
                self.sram[i] = data[i]
                
            logger.info(f"Loaded {len(data)} bytes of SRAM from {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load SRAM: {e}")
            return False