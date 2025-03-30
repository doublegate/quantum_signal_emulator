"""
Nintendo Entertainment System (NES) cartridge handling module.

This module handles different NES cartridge formats, mappers, and bankswitching
schemas. It provides support for ROM loading, memory mapping, and various
mapper implementations for accurate emulation.
"""

import logging
from typing import Dict, List, Tuple, Optional, Any, BinaryIO
import os

logger = logging.getLogger("QuantumSignalEmulator.NES.Cartridge")

# iNES mapper numbers for common mappers
MAPPER_NROM = 0      # No mapper (32KB ROM + 8KB VRAM)
MAPPER_MMC1 = 1      # Nintendo MMC1
MAPPER_UNROM = 2     # UxROM
MAPPER_CNROM = 3     # CNROM
MAPPER_MMC3 = 4      # Nintendo MMC3
MAPPER_MMC5 = 5      # Nintendo MMC5
MAPPER_AOROM = 7     # AxROM

class Mapper:
    """Base class for NES mappers."""
    def __init__(self, rom_data: bytes, chr_data: bytes, prg_ram_size: int = 8192):
        """
        Initialize the mapper.
        
        Args:
            rom_data: PRG ROM data
            chr_data: CHR ROM data
            prg_ram_size: Size of PRG RAM in bytes
        """
        self.rom = rom_data
        self.chr = chr_data
        self.rom_size = len(rom_data)
        self.chr_size = len(chr_data)
        
        # PRG RAM (battery-backed or not)
        self.prg_ram = bytearray(prg_ram_size)
        self.prg_ram_size = prg_ram_size
        
        # CHR RAM (if CHR ROM is empty)
        if self.chr_size == 0:
            self.chr_ram = bytearray(8192)  # 8KB CHR RAM
            self.uses_chr_ram = True
        else:
            self.chr_ram = bytearray(0)
            self.uses_chr_ram = False
    
    def read_prg(self, address: int) -> int:
        """
        Read from PRG memory.
        
        Args:
            address: CPU address (0x8000-0xFFFF)
            
        Returns:
            Byte at mapped address
        """
        # Default implementation maps linearly
        rom_addr = address - 0x8000
        if rom_addr < self.rom_size:
            return self.rom[rom_addr]
        
        # Mirroring for smaller ROMs
        rom_addr %= self.rom_size
        return self.rom[rom_addr]
    
    def write_prg(self, address: int, value: int) -> None:
        """
        Write to PRG memory (used for mapper registers).
        
        Args:
            address: CPU address (0x8000-0xFFFF)
            value: Byte to write
        """
        # By default, ROM writes are ignored
        pass
    
    def read_chr(self, address: int) -> int:
        """
        Read from CHR memory.
        
        Args:
            address: PPU address (0x0000-0x1FFF)
            
        Returns:
            Byte at mapped address
        """
        if self.uses_chr_ram:
            # Using CHR RAM
            return self.chr_ram[address % len(self.chr_ram)]
        else:
            # Using CHR ROM
            return self.chr[address % self.chr_size]
    
    def write_chr(self, address: int, value: int) -> None:
        """
        Write to CHR memory.
        
        Args:
            address: PPU address (0x0000-0x1FFF)
            value: Byte to write
        """
        if self.uses_chr_ram:
            # CHR RAM is writable
            self.chr_ram[address % len(self.chr_ram)] = value
    
    def read_sram(self, address: int) -> int:
        """
        Read from cartridge SRAM (0x6000-0x7FFF).
        
        Args:
            address: CPU address
            
        Returns:
            Byte at mapped address
        """
        sram_addr = address - 0x6000
        if sram_addr < self.prg_ram_size:
            return self.prg_ram[sram_addr]
        return 0
    
    def write_sram(self, address: int, value: int) -> None:
        """
        Write to cartridge SRAM (0x6000-0x7FFF).
        
        Args:
            address: CPU address
            value: Byte to write
        """
        sram_addr = address - 0x6000
        if sram_addr < self.prg_ram_size:
            self.prg_ram[sram_addr] = value
    
    def get_mirroring(self) -> str:
        """
        Get the current mirroring mode.
        
        Returns:
            Mirroring mode: 'horizontal', 'vertical', 'single0', 'single1', or 'four'
        """
        # Default to horizontal mirroring
        return 'horizontal'

class MapperNROM(Mapper):
    """
    NROM mapper (Mapper 0).
    
    The simplest mapper with no bankswitching. 16KB or 32KB PRG ROM
    and 8KB CHR ROM.
    """
    
    def __init__(self, rom_data: bytes, chr_data: bytes, prg_ram_size: int = 8192, mirror_mode: str = 'horizontal'):
        """
        Initialize the NROM mapper.
        
        Args:
            rom_data: PRG ROM data
            chr_data: CHR ROM data
            prg_ram_size: Size of PRG RAM in bytes
            mirror_mode: Mirroring mode ('horizontal' or 'vertical')
        """
        super().__init__(rom_data, chr_data, prg_ram_size)
        self.mirror_mode = mirror_mode
    
    def read_prg(self, address: int) -> int:
        """
        Read from PRG memory.
        
        Args:
            address: CPU address (0x8000-0xFFFF)
            
        Returns:
            Byte at mapped address
        """
        # NROM has 16KB or 32KB of PRG ROM
        if self.rom_size <= 16384:
            # 16KB ROM is mirrored in both banks
            rom_addr = (address - 0x8000) % 16384
        else:
            # 32KB ROM uses full space
            rom_addr = address - 0x8000
            
        return self.rom[rom_addr % self.rom_size]
    
    def get_mirroring(self) -> str:
        """
        Get the current mirroring mode.
        
        Returns:
            Mirroring mode: 'horizontal' or 'vertical'
        """
        return self.mirror_mode

class MapperMMC1(Mapper):
    """
    MMC1 mapper (Mapper 1).
    
    Features:
    - PRG ROM bank switching (16KB + 16KB or 32KB)
    - CHR ROM bank switching (4KB + 4KB or 8KB)
    - Controllable mirroring
    """
    
    def __init__(self, rom_data: bytes, chr_data: bytes, prg_ram_size: int = 8192):
        """
        Initialize the MMC1 mapper.
        
        Args:
            rom_data: PRG ROM data
            chr_data: CHR ROM data
            prg_ram_size: Size of PRG RAM in bytes
        """
        super().__init__(rom_data, chr_data, prg_ram_size)
        
        # MMC1 registers
        self.control = 0x0C      # Default: PRG ROM mode 3, horizontal mirroring
        self.chr_bank_0 = 0
        self.chr_bank_1 = 0
        self.prg_bank = 0
        
        # Shift register for serial writes
        self.shift_register = 0x10
        self.shift_count = 0
        
        # Calculate number of PRG ROM and CHR ROM banks
        self.prg_banks = self.rom_size // 16384  # 16KB banks
        self.chr_banks = self.chr_size // 4096   # 4KB banks
    
    def write_prg(self, address: int, value: int) -> None:
        """
        Write to PRG memory (used for mapper registers).
        
        Args:
            address: CPU address (0x8000-0xFFFF)
            value: Byte to write
        """
        # Check if reset bit (bit 7) is set
        if value & 0x80:
            # Reset shift register and control register
            self.shift_register = 0x10
            self.shift_count = 0
            self.control |= 0x0C  # Set bits 2 and 3 for fixed PRG mode
            return
            
        # Serial data loading
        # Each write provides 1 bit of data
        bit = value & 0x01
        self.shift_register = (self.shift_register >> 1) | (bit << 4)
        self.shift_count += 1
        
        # After 5 bits are written, update the appropriate register
        if self.shift_count == 5:
            # Determine which register to update based on address
            if 0x8000 <= address <= 0x9FFF:
                # Control register
                self.control = self.shift_register
            elif 0xA000 <= address <= 0xBFFF:
                # CHR bank 0 register
                self.chr_bank_0 = self.shift_register
            elif 0xC000 <= address <= 0xDFFF:
                # CHR bank 1 register
                self.chr_bank_1 = self.shift_register
            elif 0xE000 <= address <= 0xFFFF:
                # PRG bank register
                self.prg_bank = self.shift_register & 0x0F
            
            # Reset shift register
            self.shift_register = 0x10
            self.shift_count = 0
    
    def read_prg(self, address: int) -> int:
        """
        Read from PRG memory.
        
        Args:
            address: CPU address (0x8000-0xFFFF)
            
        Returns:
            Byte at mapped address
        """
        # Get PRG ROM mode (bits 2-3 of control register)
        prg_mode = (self.control >> 2) & 0x03
        
        # Calculate bank number and offset
        if 0x8000 <= address <= 0x9FFF:
            if prg_mode in [0, 1]:
                # 32KB mode: use bits 0-3 of PRG bank
                bank = self.prg_bank & 0x0E
            elif prg_mode == 2:
                # Fix first bank to bank 0
                bank = 0
            else:  # prg_mode == 3
                # Fix first bank to bank (last bank - 1)
                bank = (self.prg_banks - 1) & 0x0E
                
            offset = address - 0x8000
                
        elif 0xC000 <= address <= 0xFFFF:
            if prg_mode in [0, 1]:
                # 32KB mode: use bits 0-3 of PRG bank
                bank = (self.prg_bank & 0x0E) + 1
            elif prg_mode == 2:
                # Fix last bank to last bank
                bank = self.prg_banks - 1
            else:  # prg_mode == 3
                # Fix last bank to last bank
                bank = self.prg_banks - 1
                
            offset = address - 0xC000
        
        # Calculate final ROM address
        rom_addr = (bank * 16384) + offset
        
        # Handle out of range
        rom_addr %= self.rom_size
        
        return self.rom[rom_addr]
    
    def read_chr(self, address: int) -> int:
        """
        Read from CHR memory.
        
        Args:
            address: PPU address (0x0000-0x1FFF)
            
        Returns:
            Byte at mapped address
        """
        if self.uses_chr_ram:
            # Using CHR RAM
            return self.chr_ram[address % len(self.chr_ram)]
        
        # Get CHR ROM mode (bit 4 of control register)
        chr_mode = (self.control >> 4) & 0x01
        
        # Calculate bank number and offset
        if address < 0x1000:
            if chr_mode == 0:
                # 8KB mode: use bits 0-3 of CHR bank 0, ignore bit 0
                bank = (self.chr_bank_0 & 0x1E) // 2
                offset = address
            else:
                # 4KB mode: use bits 0-4 of CHR bank 0
                bank = self.chr_bank_0
                offset = address
        else:
            if chr_mode == 0:
                # 8KB mode: use bits 0-3 of CHR bank 0, ignore bit 0
                bank = (self.chr_bank_0 & 0x1E) // 2
                offset = address
            else:
                # 4KB mode: use bits 0-4 of CHR bank 1
                bank = self.chr_bank_1
                offset = address - 0x1000
        
        # Calculate final CHR address
        chr_addr = (bank * 4096) + offset
        
        # Handle out of range
        chr_addr %= self.chr_size
        
        return self.chr[chr_addr]
    
    def write_chr(self, address: int, value: int) -> None:
        """
        Write to CHR memory.
        
        Args:
            address: PPU address (0x0000-0x1FFF)
            value: Byte to write
        """
        if self.uses_chr_ram:
            # CHR RAM is writable
            self.chr_ram[address % len(self.chr_ram)] = value
    
    def get_mirroring(self) -> str:
        """
        Get the current mirroring mode.
        
        Returns:
            Mirroring mode
        """
        # Get mirroring from bits 0-1 of control register
        mirroring = self.control & 0x03
        
        if mirroring == 0:
            return 'single0'  # One-screen mirroring (lower bank)
        elif mirroring == 1:
            return 'single1'  # One-screen mirroring (upper bank)
        elif mirroring == 2:
            return 'vertical'
        else:  # mirroring == 3
            return 'horizontal'

class MapperUxROM(Mapper):
    """
    UxROM mapper (Mapper 2).
    
    Features:
    - PRG ROM bank switching (16KB switchable + 16KB fixed to last bank)
    - CHR ROM is not bankswitched
    """
    
    def __init__(self, rom_data: bytes, chr_data: bytes, prg_ram_size: int = 8192, mirror_mode: str = 'horizontal'):
        """
        Initialize the UxROM mapper.
        
        Args:
            rom_data: PRG ROM data
            chr_data: CHR ROM data
            prg_ram_size: Size of PRG RAM in bytes
            mirror_mode: Mirroring mode ('horizontal' or 'vertical')
        """
        super().__init__(rom_data, chr_data, prg_ram_size)
        self.mirror_mode = mirror_mode
        
        # Current PRG ROM bank
        self.prg_bank = 0
        
        # Calculate number of PRG ROM banks
        self.prg_banks = self.rom_size // 16384  # 16KB banks
    
    def write_prg(self, address: int, value: int) -> None:
        """
        Write to PRG memory (used for mapper registers).
        
        Args:
            address: CPU address (0x8000-0xFFFF)
            value: Byte to write
        """
        # Bank select register
        self.prg_bank = value & 0x0F
    
    def read_prg(self, address: int) -> int:
        """
        Read from PRG memory.
        
        Args:
            address: CPU address (0x8000-0xFFFF)
            
        Returns:
            Byte at mapped address
        """
        # Calculate bank number and offset
        if 0x8000 <= address <= 0xBFFF:
            # Switchable bank
            bank = self.prg_bank
            offset = address - 0x8000
        else:  # 0xC000 <= address <= 0xFFFF
            # Fixed to last bank
            bank = self.prg_banks - 1
            offset = address - 0xC000
        
        # Calculate final ROM address
        rom_addr = (bank * 16384) + offset
        
        # Handle out of range
        rom_addr %= self.rom_size
        
        return self.rom[rom_addr]
    
    def get_mirroring(self) -> str:
        """
        Get the current mirroring mode.
        
        Returns:
            Mirroring mode: 'horizontal' or 'vertical'
        """
        return self.mirror_mode

def create_mapper(mapper_num: int, rom_data: bytes, chr_data: bytes, mirroring: str = 'horizontal') -> Mapper:
    """
    Create a mapper instance based on mapper number.
    
    Args:
        mapper_num: Mapper number from iNES header
        rom_data: PRG ROM data
        chr_data: CHR ROM data
        mirroring: Nametable mirroring mode
        
    Returns:
        Mapper instance
    """
    if mapper_num == MAPPER_NROM:
        return MapperNROM(rom_data, chr_data, mirror_mode=mirroring)
    elif mapper_num == MAPPER_MMC1:
        return MapperMMC1(rom_data, chr_data)
    elif mapper_num == MAPPER_UNROM:
        return MapperUxROM(rom_data, chr_data, mirror_mode=mirroring)
    else:
        # Default to NROM for unsupported mappers
        logger.warning(f"Mapper {mapper_num} not implemented, falling back to NROM")
        return MapperNROM(rom_data, chr_data, mirror_mode=mirroring)