# systems/nes/memory.py
from ...common.interfaces import Memory

class NESMemory(Memory):
    def __init__(self, config: dict):
        self.ram = bytearray(2048)  # 2KB internal RAM
        self.ppu_registers = bytearray(8)  # 8 PPU registers
        self.apu_io_registers = bytearray(24)  # 24 APU and IO registers
        
        self.cartridge_prg_rom = None
        self.cartridge_chr_rom = None
        self.mapper = None
        self.mirroring = 'horizontal'
        
        self.config = config
    
    def read(self, address: int) -> int:
        # Memory map implementation
        if address < 0x2000:  # Internal RAM
            return self.ram[address & 0x07FF]  # Mirrored every 2KB
        elif address < 0x4000:  # PPU registers
            return self._read_ppu_register(address & 0x2007)  # Mirrored every 8 bytes
        elif address < 0x4020:  # APU and IO registers
            return self._read_apu_io_register(address & 0x401F)
        else:  # Cartridge space
            return self._read_cartridge(address)
    
    def write(self, address: int, value: int) -> None:
        # Memory map implementation
        if address < 0x2000:  # Internal RAM
            self.ram[address & 0x07FF] = value & 0xFF
        elif address < 0x4000:  # PPU registers
            self._write_ppu_register(address & 0x2007, value & 0xFF)
        elif address < 0x4020:  # APU and IO registers
            self._write_apu_io_register(address & 0x401F, value & 0xFF)
        else:  # Cartridge space
            self._write_cartridge(address, value & 0xFF)
    
    def load_rom(self, rom_data: bytes) -> None:
        # Parse iNES format
        if rom_data[0:4] != b'NES\x1a':
            raise ValueError("Not a valid iNES ROM file")
        
        prg_rom_size = rom_data[4] * 16384  # 16KB units
        chr_rom_size = rom_data[5] * 8192   # 8KB units
        flags6 = rom_data[6]
        flags7 = rom_data[7]
        
        # Get mapper number
        mapper_number = (flags7 & 0xF0) | (flags6 >> 4)
        
        # Set mirroring
        self.mirroring = 'vertical' if (flags6 & 1) else 'horizontal'
        
        # Extract ROM data
        header_size = 16
        self.cartridge_prg_rom = rom_data[header_size:header_size+prg_rom_size]
        self.cartridge_chr_rom = rom_data[header_size+prg_rom_size:header_size+prg_rom_size+chr_rom_size]
        
        # Initialize mapper
        from .cartridge import create_mapper
        self.mapper = create_mapper(mapper_number, self.cartridge_prg_rom, self.cartridge_chr_rom)
    
    # Helper methods for memory access
    def _read_ppu_register(self, address: int) -> int:
        reg_index = address - 0x2000
        # PPU register read side effects would be implemented here
        return self.ppu_registers[reg_index]
    
    def _write_ppu_register(self, address: int, value: int) -> None:
        reg_index = address - 0x2000
        # PPU register write side effects would be implemented here
        self.ppu_registers[reg_index] = value
    
    def _read_apu_io_register(self, address: int) -> int:
        # APU register handling
        return self.apu_io_registers[address - 0x4000]
    
    def _write_apu_io_register(self, address: int, value: int) -> None:
        # APU register handling with side effects
        self.apu_io_registers[address - 0x4000] = value
    
    def _read_cartridge(self, address: int) -> int:
        if self.mapper:
            return self.mapper.read(address)
        return 0
    
    def _write_cartridge(self, address: int, value: int) -> None:
        if self.mapper:
            self.mapper.write(address, value)