# systems/nes/cpu.py
from ...common.interfaces import CPU, Memory
import typing as t

class CPU6502(CPU):
    def __init__(self):
        # CPU registers
        self.A = 0x00  # Accumulator
        self.X = 0x00  # X index
        self.Y = 0x00  # Y index
        self.SP = 0xFD  # Stack pointer
        self.PC = 0x0000  # Program counter
        self.P = 0x24  # Status register
        
        # Cycle counting
        self.cycles = 0
        self.memory = None
        
        # Instruction table
        self._build_instruction_table()
        
    def _build_instruction_table(self):
        # Create lookup table for all 6502 instructions
        self.instructions = {}
        
        # Addressing modes
        self.addr_immediate = lambda: self.PC + 1
        # ... more addressing modes ...
        
        # Instructions (just a few examples)
        self.instructions[0xA9] = (self._lda, self.addr_immediate, 2)  # LDA Immediate
        self.instructions[0x8D] = (self._sta, self.addr_absolute, 4)   # STA Absolute
        # ... many more instructions ...
    
    def set_memory(self, memory: Memory) -> None:
        self.memory = memory
    
    def reset(self) -> None:
        # Read reset vector
        if self.memory:
            self.PC = self.memory.read(0xFFFC) | (self.memory.read(0xFFFD) << 8)
        else:
            self.PC = 0xC000  # Default if no memory attached
            
        self.SP = 0xFD
        self.A = 0x00
        self.X = 0x00
        self.Y = 0x00
        self.P = 0x24
        self.cycles = 0
    
    def step(self) -> int:
        if not self.memory:
            raise RuntimeError("CPU has no memory attached")
            
        # Get opcode
        opcode = self.memory.read(self.PC)
        self.PC += 1
        
        # Execute instruction
        if opcode in self.instructions:
            operation, addressing, base_cycles = self.instructions[opcode]
            address = addressing()
            extra_cycles = operation(address)
            cycles_used = base_cycles + extra_cycles
            self.cycles += cycles_used
            return cycles_used
        else:
            raise ValueError(f"Unknown opcode: {opcode:02X}")
    
    def get_state(self) -> dict:
        return {
            "A": self.A,
            "X": self.X,
            "Y": self.Y,
            "SP": self.SP,
            "PC": self.PC,
            "P": self.P,
            "cycles": self.cycles
        }
    
    # Instruction implementations
    def _lda(self, address: int) -> int:
        self.A = self.memory.read(address) & 0xFF
        # Set zero and negative flags
        self.P = (self.P & 0x7D) | (self.A == 0) << 1 | (self.A & 0x80)
        return 0
    
    def _sta(self, address: int) -> int:
        self.memory.write(address, self.A)
        return 0
    
    # ... many more instruction implementations ...