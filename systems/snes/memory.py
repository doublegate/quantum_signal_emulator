"""
SNES memory system implementation.

The SNES has a complex memory map with various regions for RAM, ROM, and
hardware registers. This module implements the SNES memory system with
support for different memory types, including Work RAM (WRAM), Save RAM (SRAM),
Video RAM (VRAM), and ROM mapping through various mappers (LoROM, HiROM, etc.).
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple

logger = logging.getLogger("QuantumSignalEmulator.SNES.Memory")

class SNESMemory:
    """
    Implements the SNES memory system.
    
    The SNES has a complex 24-bit address space with various memory regions:
    - WRAM: 128KB of system RAM
    - SRAM: Up to 64KB of save RAM
    - ROM: Up to 4MB of game ROM
    - VRAM: 64KB of video RAM
    - OAM: 544 bytes of sprite data
    - CGRAM: 512 bytes of color palette data
    - Hardware registers: Various I/O and control registers
    
    This implementation supports different memory access patterns and mappings.
    """
    
    # Memory size constants
    WRAM_SIZE = 128 * 1024  # 128KB Work RAM
    SRAM_SIZE = 64 * 1024   # 64KB Save RAM (maximum)
    VRAM_SIZE = 64 * 1024   # 64KB Video RAM
    OAM_SIZE = 544          # 544 bytes Object Attribute Memory
    CGRAM_SIZE = 512        # 512 bytes Color Generator RAM
    ROM_MAX_SIZE = 4 * 1024 * 1024  # 4MB maximum ROM size
    
    # Memory map constants
    LOROM = 0
    HIROM = 1
    EXLOROM = 2
    EXHIROM = 3
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the SNES memory system.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # Initialize memory arrays
        self.wram = bytearray(self.WRAM_SIZE)
        self.sram = bytearray(self.SRAM_SIZE)
        self.vram = bytearray(self.VRAM_SIZE)
        self.oam = bytearray(self.OAM_SIZE)
        self.cgram = bytearray(self.CGRAM_SIZE)
        
        # ROM data
        self.rom = None
        self.rom_size = 0
        self.rom_mask = 0
        
        # ROM mapping mode
        self.rom_mapping = self.LOROM
        
        # Hardware registers
        self.registers = {}
        
        # Connected components
        self.ppu = None
        self.apu = None
        self.dma = None
        
        # DMA active flag
        self.dma_active = False
        
        logger.info("SNES memory system initialized")
    
    def connect_ppu(self, ppu) -> None:
        """
        Connect the PPU component.
        
        Args:
            ppu: PPU component
        """
        self.ppu = ppu
        logger.debug("PPU connected to memory system")
    
    def connect_apu(self, apu) -> None:
        """
        Connect the APU component.
        
        Args:
            apu: APU component
        """
        self.apu = apu
        logger.debug("APU connected to memory system")
    
    def connect_dma(self, dma) -> None:
        """
        Connect the DMA component.
        
        Args:
            dma: DMA component
        """
        self.dma = dma
        logger.debug("DMA connected to memory system")
    
    def read(self, address: int) -> int:
        """
        Read a byte from memory.
        
        Args:
            address: 24-bit address
            
        Returns:
            Byte value at the address
        """
        # Split address into bank and offset
        bank = (address >> 16) & 0xFF
        addr = address & 0xFFFF
        
        # Handle different memory regions based on bank
        
        # 00-3F, 80-BF: System area (WRAM, hardware registers, etc.)
        if bank < 0x40 or (bank >= 0x80 and bank < 0xC0):
            return self._read_system_area(bank, addr)
            
        # 40-7D, C0-FF: ROM access varies by mapping mode
        elif (bank >= 0x40 and bank < 0x7E) or bank >= 0xC0:
            return self._read_rom(bank, addr)
            
        # 7E-7F: WRAM
        elif bank == 0x7E:
            return self.wram[addr]
        elif bank == 0x7F:
            return self.wram[0x10000 + addr]
            
        # Default case
        logger.warning(f"Unhandled memory read at ${address:06X}")
        return 0
    
    def write(self, address: int, value: int) -> None:
        """
        Write a byte to memory.
        
        Args:
            address: 24-bit address
            value: Byte value to write
        """
        # Ensure value is a byte
        value &= 0xFF
        
        # Split address into bank and offset
        bank = (address >> 16) & 0xFF
        addr = address & 0xFFFF
        
        # Handle different memory regions based on bank
        
        # 00-3F, 80-BF: System area (WRAM, hardware registers, etc.)
        if bank < 0x40 or (bank >= 0x80 and bank < 0xC0):
            self._write_system_area(bank, addr, value)
            
        # 40-7D, C0-FF: ROM access (some areas may be writable for SRAM)
        elif (bank >= 0x40 and bank < 0x7E) or bank >= 0xC0:
            self._write_rom(bank, addr, value)
            
        # 7E-7F: WRAM
        elif bank == 0x7E:
            self.wram[addr] = value
        elif bank == 0x7F:
            self.wram[0x10000 + addr] = value
            
        # Default case
        else:
            logger.warning(f"Unhandled memory write at ${address:06X} = ${value:02X}")
    
    def _read_system_area(self, bank: int, addr: int) -> int:
        """
        Read from system area (banks 00-3F, 80-BF).
        
        Args:
            bank: Bank number
            addr: Address offset within bank
            
        Returns:
            Byte value
        """
        # Simplify bank handling
        # Banks 80-BF are mirrors of 00-3F
        bank &= 0x3F
        
        # Handle different address ranges
        if addr < 0x2000:
            # Lower WRAM (mirror of 7E:0000-7E:1FFF)
            return self.wram[addr]
            
        elif addr < 0x4000:
            # PPU registers
            if self.ppu:
                return self.ppu.read_register(addr & 0x3F)
            return 0
            
        elif addr < 0x4400:
            # CPU/APU registers
            if addr < 0x4200:
                # Internal CPU registers
                return self._read_internal_register(addr & 0xFF)
            elif addr < 0x4300:
                # APU registers
                if self.apu:
                    return self.apu.read_register(addr & 0xFF)
                return 0
            else:
                # DMA registers
                if self.dma:
                    return self.dma.read_register(addr & 0xFF)
                return 0
                
        elif addr >= 0x6000 and addr < 0x8000:
            # Expansion memory or SRAM depending on mapping
            if self.rom_mapping in [self.LOROM, self.EXLOROM]:
                # In LoROM, this is often expansion memory, but use SRAM for simplicity
                sram_addr = ((bank & 0x3F) << 13) | (addr & 0x1FFF)
                if sram_addr < self.SRAM_SIZE:
                    return self.sram[sram_addr]
            return 0
            
        elif addr >= 0x8000:
            # ROM access in system banks
            return self._read_rom(bank, addr)
            
        # Default case
        return 0
    
    def _write_system_area(self, bank: int, addr: int, value: int) -> None:
        """
        Write to system area (banks 00-3F, 80-BF).
        
        Args:
            bank: Bank number
            addr: Address offset within bank
            value: Byte value to write
        """
        # Simplify bank handling
        # Banks 80-BF are mirrors of 00-3F
        bank &= 0x3F
        
        # Handle different address ranges
        if addr < 0x2000:
            # Lower WRAM (mirror of 7E:0000-7E:1FFF)
            self.wram[addr] = value
            
        elif addr < 0x4000:
            # PPU registers
            if self.ppu:
                self.ppu.write_register(addr & 0x3F, value)
                
        elif addr < 0x4400:
            # CPU/APU registers
            if addr < 0x4200:
                # Internal CPU registers
                self._write_internal_register(addr & 0xFF, value)
            elif addr < 0x4300:
                # APU registers
                if self.apu:
                    self.apu.write_register(addr & 0xFF, value)
            else:
                # DMA registers
                if self.dma:
                    self.dma.write_register(addr & 0xFF, value)
                    # Check if DMA should be triggered
                    if addr == 0x420B and value > 0:
                        self._trigger_dma(value)
                        
        elif addr >= 0x6000 and addr < 0x8000:
            # Expansion memory or SRAM depending on mapping
            if self.rom_mapping in [self.LOROM, self.EXLOROM]:
                # In LoROM, this is often expansion memory, but use SRAM for simplicity
                sram_addr = ((bank & 0x3F) << 13) | (addr & 0x1FFF)
                if sram_addr < self.SRAM_SIZE:
                    self.sram[sram_addr] = value
    
    def _read_internal_register(self, addr: int) -> int:
        """
        Read internal CPU register.
        
        Args:
            addr: Register address
            
        Returns:
            Register value
        """
        # Handle specific registers
        if addr in self.registers:
            return self.registers[addr]
            
        # Default value
        return 0
    
    def _write_internal_register(self, addr: int, value: int) -> None:
        """
        Write to internal CPU register.
        
        Args:
            addr: Register address
            value: Value to write
        """
        # Handle specific registers
        self.registers[addr] = value
        
        # Special handling for some registers
        if addr == 0x0D:
            # Memory mapping mode register
            if value & 0x01:
                self.rom_mapping = self.HIROM
            else:
                self.rom_mapping = self.LOROM
                
            logger.debug(f"ROM mapping set to {'HiROM' if self.rom_mapping == self.HIROM else 'LoROM'}")
    
    def _read_rom(self, bank: int, addr: int) -> int:
        """
        Read from ROM.
        
        Args:
            bank: Bank number
            addr: Address offset within bank
            
        Returns:
            Byte value
        """
        if not self.rom:
            return 0
            
        # Calculate ROM address based on mapping mode
        rom_addr = self._map_rom_address(bank, addr)
        
        # Check if address is within ROM bounds
        if rom_addr is not None and rom_addr < self.rom_size:
            return self.rom[rom_addr]
            
        return 0
    
    def _write_rom(self, bank: int, addr: int, value: int) -> None:
        """
        Write to ROM area (might be SRAM in some cases).
        
        Args:
            bank: Bank number
            addr: Address offset within bank
            value: Byte value to write
        """
        # In LoROM mapping, banks 70-7D and F0-FF can contain SRAM
        if self.rom_mapping == self.LOROM and ((bank >= 0x70 and bank < 0x7E) or bank >= 0xF0):
            sram_addr = ((bank & 0x0F) << 13) | (addr & 0x1FFF)
            if sram_addr < self.SRAM_SIZE:
                self.sram[sram_addr] = value
                
        # In HiROM mapping, banks 30-3F can contain SRAM
        elif self.rom_mapping == self.HIROM and (bank >= 0x30 and bank < 0x40):
            sram_addr = ((bank & 0x0F) << 16) | addr
            if sram_addr < self.SRAM_SIZE:
                self.sram[sram_addr] = value
    
    def _map_rom_address(self, bank: int, addr: int) -> Optional[int]:
        """
        Map SNES address to ROM address based on mapping mode.
        
        Args:
            bank: Bank number
            addr: Address offset within bank
            
        Returns:
            ROM address or None if not mapped
        """
        if self.rom_mapping == self.LOROM:
            # LoROM mapping
            if addr >= 0x8000:
                # Calculate ROM address
                rom_addr = ((bank & 0x7F) << 15) | (addr & 0x7FFF)
                return rom_addr & self.rom_mask
        elif self.rom_mapping == self.HIROM:
            # HiROM mapping
            if bank >= 0x40 and addr >= 0x0000:
                # Calculate ROM address
                rom_addr = ((bank & 0x3F) << 16) | addr
                return rom_addr & self.rom_mask
            elif bank < 0x40 and addr >= 0x8000:
                # Calculate ROM address
                rom_addr = ((bank & 0x3F) << 16) | (addr & 0x7FFF) | 0x8000
                return rom_addr & self.rom_mask
        
        # Not mapped to ROM
        return None
    
    def _trigger_dma(self, channels: int) -> None:
        """
        Trigger DMA transfer on specified channels.
        
        Args:
            channels: Bit mask of channels to activate
        """
        if self.dma:
            # Set DMA active flag
            self.dma_active = True
            
            # Perform DMA transfers
            self.dma.trigger_transfer(channels)
            
            # Clear DMA active flag
            self.dma_active = False
    
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
        
        # Calculate ROM mask for address wrapping
        # Find the smallest power of 2 that's >= rom_size
        mask = 1
        while mask < self.rom_size:
            mask *= 2
        self.rom_mask = mask - 1
        
        # Determine mapping mode from ROM header
        self._detect_mapping_mode()
        
        logger.info(f"ROM loaded: {self.rom_size} bytes, mapping: {'HiROM' if self.rom_mapping == self.HIROM else 'LoROM'}")
    
    def _detect_mapping_mode(self) -> None:
        """Detect ROM mapping mode based on header information."""
        if not self.rom or self.rom_size < 0x8000:
            # Default to LoROM for small or invalid ROMs
            self.rom_mapping = self.LOROM
            return
            
        # Check for header at both LoROM and HiROM locations
        lorom_score = 0
        hirom_score = 0
        
        # LoROM header location
        lorom_header = 0x7FD5
        if lorom_header < self.rom_size:
            mapping_byte = self.rom[lorom_header]
            if mapping_byte & 0x01 == 0:
                lorom_score += 10  # Consistent with LoROM
                
            # Check checksum
            if lorom_header + 3 < self.rom_size:
                checksum = self.rom[lorom_header + 1] | (self.rom[lorom_header + 2] << 8)
                checksum_complement = self.rom[lorom_header + 3] | (self.rom[lorom_header + 4] << 8)
                if (checksum ^ checksum_complement) == 0xFFFF:
                    lorom_score += 5  # Valid checksum
                    
            # Check title (should be ASCII)
            if lorom_header - 21 >= 0:
                title_start = lorom_header - 21
                title_end = lorom_header - 1
                ascii_count = 0
                for i in range(title_start, title_end):
                    if 0x20 <= self.rom[i] <= 0x7E:
                        ascii_count += 1
                if ascii_count > 15:
                    lorom_score += 5  # Title looks valid
        
        # HiROM header location
        hirom_header = 0xFFD5
        if hirom_header < self.rom_size:
            mapping_byte = self.rom[hirom_header]
            if mapping_byte & 0x01 == 1:
                hirom_score += 10  # Consistent with HiROM
                
            # Check checksum
            if hirom_header + 3 < self.rom_size:
                checksum = self.rom[hirom_header + 1] | (self.rom[hirom_header + 2] << 8)
                checksum_complement = self.rom[hirom_header + 3] | (self.rom[hirom_header + 4] << 8)
                if (checksum ^ checksum_complement) == 0xFFFF:
                    hirom_score += 5  # Valid checksum
                    
            # Check title (should be ASCII)
            if hirom_header - 21 >= 0:
                title_start = hirom_header - 21
                title_end = hirom_header - 1
                ascii_count = 0
                for i in range(title_start, title_end):
                    if 0x20 <= self.rom[i] <= 0x7E:
                        ascii_count += 1
                if ascii_count > 15:
                    hirom_score += 5  # Title looks valid
        
        # Determine mapping mode based on scores
        if hirom_score > lorom_score:
            self.rom_mapping = self.HIROM
        else:
            self.rom_mapping = self.LOROM
            
        logger.debug(f"ROM mapping detection: LoROM score {lorom_score}, HiROM score {hirom_score}")
    
    def reset(self) -> None:
        """Reset the memory system."""
        # Clear RAM
        self.wram = bytearray(self.WRAM_SIZE)
        
        # Clear registers
        self.registers = {}
        
        # Reset DMA state
        self.dma_active = False
        
        logger.info("Memory system reset")