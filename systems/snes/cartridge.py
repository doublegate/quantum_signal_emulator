"""
SNES cartridge handling module.

The SNES supports various cartridge mapper types including LoROM, HiROM, and
enhancement chips like SuperFX, DSP, and SA-1. This module provides cartridge
detection and handling for different ROM formats and memory mapping schemes.
"""

import logging
from typing import Dict, List, Tuple, Optional, Any, Union, BinaryIO
import os

logger = logging.getLogger("QuantumSignalEmulator.SNES.Cartridge")

# SNES ROM mapping modes
ROM_MAPPING = {
    "LOROM": 0,
    "HIROM": 1,
    "EXLOROM": 2,
    "EXHIROM": 3,
    "SA1ROM": 4,
    "SFXROM": 5
}

# Enhancement chips
ENHANCEMENT_CHIPS = {
    "DSP": 0x01,
    "SuperFX": 0x02,
    "SA1": 0x04,
    "S-DD1": 0x08,
    "S-RTC": 0x10,
    "CX4": 0x20,
    "SPC7110": 0x40
}

class SNESCartridge:
    """
    SNES cartridge handling.
    
    Provides cartridge type detection, ROM header parsing, and memory mapping
    for different SNES cartridge types. Handles enhancement chips and
    special memory access patterns.
    """
    
    def __init__(self):
        """Initialize the cartridge handler."""
        # ROM data
        self.rom = None
        self.rom_size = 0
        self.sram = None
        self.sram_size = 0
        
        # ROM header information
        self.title = ""
        self.rom_mapping = ROM_MAPPING["LOROM"]
        self.rom_type = 0
        self.rom_size_code = 0
        self.sram_size_code = 0
        self.country_code = 0
        self.developer_id = 0
        self.version = 0
        self.checksum = 0
        self.complement_check = 0
        
        # Enhancement chip
        self.enhancement_chip = None
        
        # ROM banks
        self.banks = []
        
        logger.info("SNES cartridge handler initialized")
    
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
            
        # Store ROM data
        self.rom = bytearray(rom_data)
        self.rom_size = len(rom_data)
        
        # Parse ROM header
        if not self._parse_header():
            logger.error("Failed to parse ROM header")
            return False
            
        # Initialize SRAM
        if self.sram_size_code > 0:
            self.sram_size = 1024 << self.sram_size_code
            self.sram = bytearray(self.sram_size)
            logger.info(f"Initialized {self.sram_size} bytes of SRAM")
        else:
            self.sram = bytearray(0)
            self.sram_size = 0
            logger.info("No SRAM in cartridge")
        
        # Set up ROM banks based on mapping
        self._setup_banks()
        
        logger.info(f"ROM loaded: {self.title}, {self.rom_size} bytes, mapping: {self._get_mapping_name()}")
        
        return True
    
    def _parse_header(self) -> bool:
        """
        Parse the ROM header to detect cartridge type.
        
        Returns:
            True if header parsed successfully
        """
        # Try to detect header location
        return self._try_parse_lorom_header() or self._try_parse_hirom_header()
    
    def _try_parse_lorom_header(self) -> bool:
        """
        Try to parse header assuming LoROM mapping.
        
        Returns:
            True if header seems valid
        """
        # LoROM header location
        if self.rom_size < 0x8000:
            return False
            
        # Header is at 0x7FB0-0x7FFF
        header_start = 0x7FB0
        
        # Read title (21 bytes)
        self.title = self._read_string(header_start, 21)
        
        # Read mapping type (at 0x7FD5)
        mapping_byte = self.rom[header_start + 0x25]
        rom_mapping = mapping_byte & 0x01
        
        # Read ROM type (at 0x7FD6)
        self.rom_type = self.rom[header_start + 0x26]
        
        # Read ROM size (at 0x7FD7)
        self.rom_size_code = self.rom[header_start + 0x27]
        
        # Read SRAM size (at 0x7FD8)
        self.sram_size_code = self.rom[header_start + 0x28]
        
        # Read country code (at 0x7FD9)
        self.country_code = self.rom[header_start + 0x29]
        
        # Read developer ID (at 0x7FDA)
        self.developer_id = self.rom[header_start + 0x2A]
        
        # Read version (at 0x7FDB)
        self.version = self.rom[header_start + 0x2B]
        
        # Read checksum complement (at 0x7FDC-0x7FDD)
        self.complement_check = self.rom[header_start + 0x2C] | (self.rom[header_start + 0x2D] << 8)
        
        # Read checksum (at 0x7FDE-0x7FDF)
        self.checksum = self.rom[header_start + 0x2E] | (self.rom[header_start + 0x2F] << 8)
        
        # Check if checksum and its complement make sense
        if (self.checksum ^ self.complement_check) != 0xFFFF:
            # Not a valid LoROM header
            return False
            
        # Detect enhancement chips
        self._detect_enhancement_chip()
        
        # Set mapping type
        self.rom_mapping = ROM_MAPPING["LOROM"]
        
        return True
    
    def _try_parse_hirom_header(self) -> bool:
        """
        Try to parse header assuming HiROM mapping.
        
        Returns:
            True if header seems valid
        """
        # HiROM header location
        if self.rom_size < 0x10000:
            return False
            
        # Header is at 0xFFB0-0xFFFF
        header_start = 0xFFB0
        
        # Read title (21 bytes)
        self.title = self._read_string(header_start, 21)
        
        # Read mapping type (at 0xFFD5)
        mapping_byte = self.rom[header_start + 0x25]
        rom_mapping = mapping_byte & 0x01
        
        # Read ROM type (at 0xFFD6)
        self.rom_type = self.rom[header_start + 0x26]
        
        # Read ROM size (at 0xFFD7)
        self.rom_size_code = self.rom[header_start + 0x27]
        
        # Read SRAM size (at 0xFFD8)
        self.sram_size_code = self.rom[header_start + 0x28]
        
        # Read country code (at 0xFFD9)
        self.country_code = self.rom[header_start + 0x29]
        
        # Read developer ID (at 0xFFDA)
        self.developer_id = self.rom[header_start + 0x2A]
        
        # Read version (at 0xFFDB)
        self.version = self.rom[header_start + 0x2B]
        
        # Read checksum complement (at 0xFFDC-0xFFDD)
        self.complement_check = self.rom[header_start + 0x2C] | (self.rom[header_start + 0x2D] << 8)
        
        # Read checksum (at 0xFFDE-0xFFDF)
        self.checksum = self.rom[header_start + 0x2E] | (self.rom[header_start + 0x2F] << 8)
        
        # Check if checksum and its complement make sense
        if (self.checksum ^ self.complement_check) != 0xFFFF:
            # Not a valid HiROM header
            return False
            
        # Detect enhancement chips
        self._detect_enhancement_chip()
        
        # Set mapping type
        self.rom_mapping = ROM_MAPPING["HIROM"]
        
        return True
    
    def _detect_enhancement_chip(self) -> None:
        """Detect enhancement chips based on ROM type."""
        # Check ROM type for enhancement chips
        if (self.rom_type & 0xF0) == 0x10:
            self.enhancement_chip = "DSP"
            logger.info("Enhancement chip detected: DSP")
        elif (self.rom_type & 0xF0) == 0x20:
            self.enhancement_chip = "SuperFX"
            logger.info("Enhancement chip detected: SuperFX")
        elif (self.rom_type & 0xF0) == 0x30:
            self.enhancement_chip = "SA1"
            logger.info("Enhancement chip detected: SA1")
        elif (self.rom_type & 0xF0) == 0x40:
            self.enhancement_chip = "S-DD1"
            logger.info("Enhancement chip detected: S-DD1")
        elif (self.rom_type & 0xF0) == 0xF0:
            if (self.rom_type & 0x0F) == 0x05:
                self.enhancement_chip = "S-RTC"
                logger.info("Enhancement chip detected: S-RTC")
            elif (self.rom_type & 0x0F) == 0x07:
                self.enhancement_chip = "SPC7110"
                logger.info("Enhancement chip detected: SPC7110")
            elif (self.rom_type & 0x0F) == 0x09:
                self.enhancement_chip = "CX4"
                logger.info("Enhancement chip detected: CX4")
    
    def _setup_banks(self) -> None:
        """Set up ROM banks based on mapping type."""
        if self.rom_mapping == ROM_MAPPING["LOROM"]:
            # For LoROM, banks are 32KB
            self.banks = []
            for i in range(0, self.rom_size, 0x8000):
                if i + 0x8000 <= self.rom_size:
                    self.banks.append(self.rom[i:i+0x8000])
                else:
                    # Pad last bank if needed
                    bank_data = self.rom[i:] + bytearray(0x8000 - (self.rom_size - i))
                    self.banks.append(bank_data)
                    
            logger.info(f"LoROM mapping: {len(self.banks)} banks of 32KB")
            
        elif self.rom_mapping == ROM_MAPPING["HIROM"]:
            # For HiROM, banks are 64KB
            self.banks = []
            for i in range(0, self.rom_size, 0x10000):
                if i + 0x10000 <= self.rom_size:
                    self.banks.append(self.rom[i:i+0x10000])
                else:
                    # Pad last bank if needed
                    bank_data = self.rom[i:] + bytearray(0x10000 - (self.rom_size - i))
                    self.banks.append(bank_data)
                    
            logger.info(f"HiROM mapping: {len(self.banks)} banks of 64KB")
        
        # For other mapping types, additional setup would be needed
    
    def read(self, address: int) -> int:
        """
        Read a byte from the cartridge ROM or SRAM.
        
        Args:
            address: 24-bit address
            
        Returns:
            Byte value at address
        """
        # Split address into bank and offset
        bank = (address >> 16) & 0xFF
        addr = address & 0xFFFF
        
        # Handle different mapping types
        if self.rom_mapping == ROM_MAPPING["LOROM"]:
            return self._read_lorom(bank, addr)
        elif self.rom_mapping == ROM_MAPPING["HIROM"]:
            return self._read_hirom(bank, addr)
        else:
            # Default to LoROM
            return self._read_lorom(bank, addr)
    
    def write(self, address: int, value: int) -> None:
        """
        Write a byte to the cartridge SRAM.
        
        Args:
            address: 24-bit address
            value: Byte value to write
        """
        # Split address into bank and offset
        bank = (address >> 16) & 0xFF
        addr = address & 0xFFFF
        
        # Handle different mapping types
        if self.rom_mapping == ROM_MAPPING["LOROM"]:
            self._write_lorom(bank, addr, value)
        elif self.rom_mapping == ROM_MAPPING["HIROM"]:
            self._write_hirom(bank, addr, value)
        else:
            # Default to LoROM
            self._write_lorom(bank, addr, value)
    
    def _read_lorom(self, bank: int, addr: int) -> int:
        """
        Read a byte using LoROM mapping.
        
        Args:
            bank: Bank number
            addr: Address within bank
            
        Returns:
            Byte value
        """
        # ROM access (banks 00-7D, 80-FF, addr 8000-FFFF)
        if addr >= 0x8000:
            bank_index = bank & 0x7F
            if bank_index < len(self.banks):
                return self.banks[bank_index][addr - 0x8000]
        
        # SRAM access (banks 70-7D, addr 0000-7FFF)
        elif bank >= 0x70 and bank <= 0x7D and addr < 0x8000:
            sram_addr = ((bank - 0x70) * 0x8000) + addr
            if sram_addr < self.sram_size:
                return self.sram[sram_addr]
        
        # Default return
        return 0
    
    def _write_lorom(self, bank: int, addr: int, value: int) -> None:
        """
        Write a byte using LoROM mapping.
        
        Args:
            bank: Bank number
            addr: Address within bank
            value: Byte value to write
        """
        # SRAM access (banks 70-7D, addr 0000-7FFF)
        if bank >= 0x70 and bank <= 0x7D and addr < 0x8000:
            sram_addr = ((bank - 0x70) * 0x8000) + addr
            if sram_addr < self.sram_size:
                self.sram[sram_addr] = value
    
    def _read_hirom(self, bank: int, addr: int) -> int:
        """
        Read a byte using HiROM mapping.
        
        Args:
            bank: Bank number
            addr: Address within bank
            
        Returns:
            Byte value
        """
        # ROM access (banks 00-3F, 80-BF addr 8000-FFFF and banks 40-7D, C0-FF addr 0000-FFFF)
        if ((bank < 0x40 or (bank >= 0x80 and bank < 0xC0)) and addr >= 0x8000) or \
           ((bank >= 0x40 and bank < 0x7E) or bank >= 0xC0):
            bank_index = bank & 0x3F
            if bank_index < len(self.banks):
                if addr >= 0x8000 or bank >= 0x40:
                    offset = addr
                    if bank >= 0x40:
                        offset += 0x8000
                    return self.banks[bank_index][offset & 0xFFFF]
        
        # SRAM access (banks 30-3F, addr 6000-7FFF)
        elif (bank >= 0x30 and bank < 0x40) and (addr >= 0x6000 and addr < 0x8000):
            sram_addr = ((bank - 0x30) * 0x2000) + (addr - 0x6000)
            if sram_addr < self.sram_size:
                return self.sram[sram_addr]
        
        # Default return
        return 0
    
    def _write_hirom(self, bank: int, addr: int, value: int) -> None:
        """
        Write a byte using HiROM mapping.
        
        Args:
            bank: Bank number
            addr: Address within bank
            value: Byte value to write
        """
        # SRAM access (banks 30-3F, addr 6000-7FFF)
        if (bank >= 0x30 and bank < 0x40) and (addr >= 0x6000 and addr < 0x8000):
            sram_addr = ((bank - 0x30) * 0x2000) + (addr - 0x6000)
            if sram_addr < self.sram_size:
                self.sram[sram_addr] = value
    
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
    
    def _get_mapping_name(self) -> str:
        """
        Get the name of the current ROM mapping.
        
        Returns:
            ROM mapping name
        """
        for name, value in ROM_MAPPING.items():
            if value == self.rom_mapping:
                return name
        return "Unknown"
    
    def save_sram(self, filename: str) -> bool:
        """
        Save SRAM data to a file.
        
        Args:
            filename: Output filename
            
        Returns:
            True if successful
        """
        if self.sram_size == 0:
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
        Load SRAM data from a file.
        
        Args:
            filename: Input filename
            
        Returns:
            True if successful
        """
        if self.sram_size == 0:
            logger.warning("No SRAM to load")
            return False
            
        try:
            with open(filename, 'rb') as f:
                data = f.read(self.sram_size)
                
            # Copy data to SRAM
            for i in range(len(data)):
                self.sram[i] = data[i]
                
            logger.info(f"Loaded {len(data)} bytes of SRAM from {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load SRAM: {e}")
            return False