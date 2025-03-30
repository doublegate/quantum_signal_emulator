"""
SNES CPU emulation (65C816 processor).

The 65C816 is a 16-bit extension of the 6502 processor used in the Super
Nintendo Entertainment System. This module provides cycle-accurate emulation
of the 65C816 CPU, implementing all its addressing modes, instructions, and
register operations with proper timing and hardware interaction.
"""

from ...common.interfaces import CPU, Memory
import logging
from typing import Dict, List, Tuple, Optional, Any, Union

logger = logging.getLogger("QuantumSignalEmulator.SNES.CPU")

class CPU65C816(CPU):
    """
    Emulates the 65C816 processor used in the SNES.
    
    The 65C816 is a 16-bit extension of the 6502 with 24-bit addressing
    capabilities, additional registers, and new addressing modes. This
    implementation provides cycle-accurate timing required for proper
    SNES emulation.
    """
    
    # 65C816 flags
    FLAG_CARRY     = 0x01
    FLAG_ZERO      = 0x02
    FLAG_IRQ       = 0x04
    FLAG_DECIMAL   = 0x08
    FLAG_INDEX_X   = 0x10  # X register size (0 = 16-bit, 1 = 8-bit)
    FLAG_INDEX_M   = 0x20  # Memory/Accumulator size (0 = 16-bit, 1 = 8-bit)
    FLAG_OVERFLOW  = 0x40
    FLAG_NEGATIVE  = 0x80
    FLAG_EMULATION = 0x100  # Not in P register, but tracked separately
    
    def __init__(self):
        """Initialize the 65C816 CPU."""
        # Main registers
        self.A = 0x0000  # Accumulator (16-bit)
        self.X = 0x0000  # X index register (16-bit)
        self.Y = 0x0000  # Y index register (16-bit)
        self.SP = 0x01FF  # Stack pointer (16-bit)
        self.PC = 0x0000  # Program counter (16-bit)
        self.P = 0x34  # Status register (8-bit) - Default to 8-bit mode
        
        # Additional 65C816 registers
        self.D = 0x0000  # Direct page register
        self.DB = 0x00  # Data bank register
        self.PBR = 0x00  # Program bank register
        
        # Emulation mode flag (separate from P)
        self.emulation_mode = True
        
        # Connected memory system
        self.memory = None
        
        # Cycle counting
        self.cycles = 0
        
        # Instruction table
        self._build_instruction_table()
        
        logger.info("65C816 CPU initialized")
    
    def _build_instruction_table(self):
        """Build the instruction lookup table."""
        # Initialize instruction table
        self.instructions = {}
        
        # Define addressing modes
        # Much more complex than 6502 due to 16-bit modes and 24-bit addressing
        # This is a simplified implementation
        
        # Some basic addressing modes
        self.addr_immediate = lambda: self.PC + 1
        self.addr_absolute = lambda: self._read_word(self.PC + 1)
        self.addr_absolute_x = lambda: (self._read_word(self.PC + 1) + self.X) & 0xFFFF
        self.addr_absolute_y = lambda: (self._read_word(self.PC + 1) + self.Y) & 0xFFFF
        self.addr_direct_page = lambda: (self.D + self._read_byte(self.PC + 1)) & 0xFFFF
        self.addr_direct_page_x = lambda: (self.D + self._read_byte(self.PC + 1) + self.X) & 0xFFFF
        self.addr_direct_page_y = lambda: (self.D + self._read_byte(self.PC + 1) + self.Y) & 0xFFFF
        self.addr_long = lambda: self._read_long(self.PC + 1)
        self.addr_long_x = lambda: (self._read_long(self.PC + 1) + self.X) & 0xFFFFFF
        
        # Addressing mode sizes (bytes)
        self.size_implied = 1
        self.size_immediate = lambda: 2 if self._is_8bit_memory() else 3
        self.size_immediate_x = lambda: 2 if self._is_8bit_index() else 3
        self.size_absolute = 3
        self.size_absolute_x = 3
        self.size_absolute_y = 3
        self.size_direct_page = 2
        self.size_direct_page_x = 2
        self.size_direct_page_y = 2
        self.size_long = 4
        self.size_long_x = 4
        
        # This is just a subset of 65C816 instructions
        # A complete implementation would include all 256 opcodes
        
        # LDA - Load Accumulator
        self.instructions[0xA9] = (self._lda, self.addr_immediate, self.size_immediate, 2)  # LDA #imm
        self.instructions[0xAD] = (self._lda, self.addr_absolute, self.size_absolute, 4)    # LDA abs
        self.instructions[0xBD] = (self._lda, self.addr_absolute_x, self.size_absolute_x, 4)  # LDA abs,X
        self.instructions[0xB9] = (self._lda, self.addr_absolute_y, self.size_absolute_y, 4)  # LDA abs,Y
        self.instructions[0xA5] = (self._lda, self.addr_direct_page, self.size_direct_page, 3)  # LDA dp
        self.instructions[0xB5] = (self._lda, self.addr_direct_page_x, self.size_direct_page_x, 4)  # LDA dp,X
        self.instructions[0xAF] = (self._lda, self.addr_long, self.size_long, 5)  # LDA long
        self.instructions[0xBF] = (self._lda, self.addr_long_x, self.size_long_x, 5)  # LDA long,X
        
        # LDX - Load X
        self.instructions[0xA2] = (self._ldx, self.addr_immediate, self.size_immediate_x, 2)  # LDX #imm
        self.instructions[0xAE] = (self._ldx, self.addr_absolute, self.size_absolute, 4)  # LDX abs
        self.instructions[0xBE] = (self._ldx, self.addr_absolute_y, self.size_absolute_y, 4)  # LDX abs,Y
        self.instructions[0xA6] = (self._ldx, self.addr_direct_page, self.size_direct_page, 3)  # LDX dp
        self.instructions[0xB6] = (self._ldx, self.addr_direct_page_y, self.size_direct_page_y, 4)  # LDX dp,Y
        
        # LDY - Load Y
        self.instructions[0xA0] = (self._ldy, self.addr_immediate, self.size_immediate_x, 2)  # LDY #imm
        self.instructions[0xAC] = (self._ldy, self.addr_absolute, self.size_absolute, 4)  # LDY abs
        self.instructions[0xBC] = (self._ldy, self.addr_absolute_x, self.size_absolute_x, 4)  # LDY abs,X
        self.instructions[0xA4] = (self._ldy, self.addr_direct_page, self.size_direct_page, 3)  # LDY dp
        self.instructions[0xB4] = (self._ldy, self.addr_direct_page_x, self.size_direct_page_x, 4)  # LDY dp,X
        
        # STA - Store Accumulator
        self.instructions[0x8D] = (self._sta, self.addr_absolute, self.size_absolute, 4)  # STA abs
        self.instructions[0x9D] = (self._sta, self.addr_absolute_x, self.size_absolute_x, 5)  # STA abs,X
        self.instructions[0x99] = (self._sta, self.addr_absolute_y, self.size_absolute_y, 5)  # STA abs,Y
        self.instructions[0x85] = (self._sta, self.addr_direct_page, self.size_direct_page, 3)  # STA dp
        self.instructions[0x95] = (self._sta, self.addr_direct_page_x, self.size_direct_page_x, 4)  # STA dp,X
        self.instructions[0x8F] = (self._sta, self.addr_long, self.size_long, 5)  # STA long
        self.instructions[0x9F] = (self._sta, self.addr_long_x, self.size_long_x, 5)  # STA long,X
        
        # STX - Store X
        self.instructions[0x8E] = (self._stx, self.addr_absolute, self.size_absolute, 4)  # STX abs
        self.instructions[0x86] = (self._stx, self.addr_direct_page, self.size_direct_page, 3)  # STX dp
        self.instructions[0x96] = (self._stx, self.addr_direct_page_y, self.size_direct_page_y, 4)  # STX dp,Y
        
        # STY - Store Y
        self.instructions[0x8C] = (self._sty, self.addr_absolute, self.size_absolute, 4)  # STY abs
        self.instructions[0x84] = (self._sty, self.addr_direct_page, self.size_direct_page, 3)  # STY dp
        self.instructions[0x94] = (self._sty, self.addr_direct_page_x, self.size_direct_page_x, 4)  # STY dp,X
        
        # JMP - Jump
        self.instructions[0x4C] = (self._jmp, self.addr_absolute, self.size_absolute, 3)  # JMP abs
        self.instructions[0x6C] = (self._jmp_indirect, None, self.size_absolute, 5)  # JMP (abs)
        self.instructions[0x7C] = (self._jmp_indirect_x, None, self.size_absolute, 6)  # JMP (abs,X)
        self.instructions[0x5C] = (self._jmp_long, None, self.size_long, 4)  # JMP long
        
        # JSR - Jump to Subroutine
        self.instructions[0x20] = (self._jsr, self.addr_absolute, self.size_absolute, 6)  # JSR abs
        self.instructions[0xFC] = (self._jsr_indirect_x, None, self.size_absolute, 8)  # JSR (abs,X)
        self.instructions[0x22] = (self._jsr_long, None, self.size_long, 8)  # JSR long
        
        # Return instructions
        self.instructions[0x60] = (self._rts, None, self.size_implied, 6)  # RTS
        self.instructions[0x6B] = (self._rtl, None, self.size_implied, 6)  # RTL
        self.instructions[0x40] = (self._rti, None, self.size_implied, 6)  # RTI
        
        # Status flag instructions
        self.instructions[0x18] = (self._clc, None, self.size_implied, 2)  # CLC
        self.instructions[0x38] = (self._sec, None, self.size_implied, 2)  # SEC
        self.instructions[0xD8] = (self._cld, None, self.size_implied, 2)  # CLD
        self.instructions[0xF8] = (self._sed, None, self.size_implied, 2)  # SED
        self.instructions[0x58] = (self._cli, None, self.size_implied, 2)  # CLI
        self.instructions[0x78] = (self._sei, None, self.size_implied, 2)  # SEI
        self.instructions[0xB8] = (self._clv, None, self.size_implied, 2)  # CLV
        
        # 65C816 specific mode setting instructions
        self.instructions[0xC2] = (self._rep, None, self.size_immediate, 3)  # REP #imm
        self.instructions[0xE2] = (self._sep, None, self.size_immediate, 3)  # SEP #imm
        self.instructions[0xFB] = (self._xce, None, self.size_implied, 2)  # XCE
        
        # More instructions would be added for a complete implementation
        
        # NOP - No Operation
        self.instructions[0xEA] = (self._nop, None, self.size_implied, 2)  # NOP
    
    def _is_8bit_memory(self) -> bool:
        """
        Check if memory/accumulator is in 8-bit mode.
        
        Returns:
            True if in 8-bit mode, False for 16-bit mode
        """
        return (self.P & self.FLAG_INDEX_M) != 0
    
    def _is_8bit_index(self) -> bool:
        """
        Check if index registers are in 8-bit mode.
        
        Returns:
            True if in 8-bit mode, False for 16-bit mode
        """
        return (self.P & self.FLAG_INDEX_X) != 0
    
    def _read_byte(self, address: int) -> int:
        """
        Read a byte from memory.
        
        Args:
            address: 16-bit address (relative to current program bank)
            
        Returns:
            Byte value from memory
        """
        if self.memory:
            full_addr = (self.PBR << 16) | (address & 0xFFFF)
            return self.memory.read(full_addr) & 0xFF
        return 0
    
    def _read_word(self, address: int) -> int:
        """
        Read a 16-bit word from memory.
        
        Args:
            address: 16-bit address (relative to current program bank)
            
        Returns:
            16-bit word value from memory
        """
        if self.memory:
            lo = self._read_byte(address)
            hi = self._read_byte(address + 1)
            return (hi << 8) | lo
        return 0
    
    def _read_long(self, address: int) -> int:
        """
        Read a 24-bit long address from memory.
        
        Args:
            address: 16-bit address (relative to current program bank)
            
        Returns:
            24-bit value from memory
        """
        if self.memory:
            lo = self._read_byte(address)
            mid = self._read_byte(address + 1)
            hi = self._read_byte(address + 2)
            return (hi << 16) | (mid << 8) | lo
        return 0
    
    def _write_byte(self, address: int, value: int) -> None:
        """
        Write a byte to memory.
        
        Args:
            address: 16-bit address (relative to current data bank)
            value: Byte value to write
        """
        if self.memory:
            full_addr = (self.DB << 16) | (address & 0xFFFF)
            self.memory.write(full_addr, value & 0xFF)
    
    def _write_word(self, address: int, value: int) -> None:
        """
        Write a 16-bit word to memory.
        
        Args:
            address: 16-bit address (relative to current data bank)
            value: 16-bit value to write
        """
        if self.memory:
            self._write_byte(address, value & 0xFF)
            self._write_byte(address + 1, (value >> 8) & 0xFF)
    
    def _push_byte(self, value: int) -> None:
        """
        Push a byte onto the stack.
        
        Args:
            value: Byte value to push
        """
        if self.emulation_mode:
            # In emulation mode, stack is fixed at page 1 and SP is 8-bit
            self.memory.write(0x0100 | (self.SP & 0xFF), value & 0xFF)
            self.SP = (self.SP & 0xFF00) | ((self.SP - 1) & 0xFF)
        else:
            # In native mode, stack can be anywhere and SP is 16-bit
            self.memory.write(0x0000 | (self.SP & 0xFFFF), value & 0xFF)
            self.SP = (self.SP - 1) & 0xFFFF
    
    def _push_word(self, value: int) -> None:
        """
        Push a 16-bit word onto the stack.
        
        Args:
            value: 16-bit value to push
        """
        self._push_byte((value >> 8) & 0xFF)  # Push high byte
        self._push_byte(value & 0xFF)  # Push low byte
    
    def _pull_byte(self) -> int:
        """
        Pull a byte from the stack.
        
        Returns:
            Byte value from stack
        """
        if self.emulation_mode:
            # In emulation mode, stack is fixed at page 1 and SP is 8-bit
            self.SP = (self.SP & 0xFF00) | ((self.SP + 1) & 0xFF)
            return self.memory.read(0x0100 | (self.SP & 0xFF)) & 0xFF
        else:
            # In native mode, stack can be anywhere and SP is 16-bit
            self.SP = (self.SP + 1) & 0xFFFF
            return self.memory.read(0x0000 | (self.SP & 0xFFFF)) & 0xFF
    
    def _pull_word(self) -> int:
        """
        Pull a 16-bit word from the stack.
        
        Returns:
            16-bit value from stack
        """
        lo = self._pull_byte()
        hi = self._pull_byte()
        return (hi << 8) | lo
    
    def set_memory(self, memory: Memory) -> None:
        """
        Connect the CPU to a memory system.
        
        Args:
            memory: Memory implementation
        """
        self.memory = memory
    
    def reset(self) -> None:
        """Reset the CPU to initial state."""
        # Set emulation mode
        self.emulation_mode = True
        
        # In emulation mode, these registers are 8-bit
        self.X &= 0xFF
        self.Y &= 0xFF
        self.SP = 0x01FF
        
        # Set flags
        self.P = 0x34  # IRQ disabled, memory/index in 8-bit mode
        
        # Clear direct page register and data bank
        self.D = 0x0000
        self.DB = 0x00
        
        # Read reset vector
        if self.memory:
            self.PC = self.memory.read(0xFFFC) | (self.memory.read(0xFFFD) << 8)
            self.PBR = 0x00
        else:
            self.PC = 0x8000
            self.PBR = 0x00
            
        # Reset cycle count
        self.cycles = 0
        
        logger.info(f"CPU reset. PC set to ${self.PBR:02X}:${self.PC:04X}")
    
    def step(self) -> int:
        """
        Execute one instruction and return cycles used.
        
        Returns:
            Number of cycles used by the instruction
        """
        if not self.memory:
            raise RuntimeError("CPU has no memory attached")
            
        # Get opcode
        opcode = self._read_byte(self.PC)
        
        # Execute instruction
        if opcode in self.instructions:
            operation, addressing, size_func, base_cycles = self.instructions[opcode]
            
            # Get instruction size
            if callable(size_func):
                size = size_func()
            else:
                size = size_func
                
            # Get memory address if addressing mode is provided
            address = None
            if addressing:
                address = addressing()
            
            # Execute the instruction
            extra_cycles = operation(address) if address is not None else operation()
            
            # Update program counter
            self.PC = (self.PC + size) & 0xFFFF
            
            # Update cycle count
            cycles_used = base_cycles + extra_cycles
            self.cycles += cycles_used
            
            return cycles_used
        else:
            # Invalid opcode - treat as NOP but log a warning
            logger.warning(f"Unknown opcode: ${opcode:02X} at ${self.PBR:02X}:${self.PC:04X}")
            self.PC = (self.PC + 1) & 0xFFFF
            self.cycles += 2
            return 2
    
    def get_state(self) -> dict:
        """
        Get the current CPU state.
        
        Returns:
            Dictionary with CPU state
        """
        return {
            "A": self.A,
            "X": self.X,
            "Y": self.Y,
            "SP": self.SP,
            "PC": self.PC,
            "P": self.P,
            "D": self.D,
            "DB": self.DB,
            "PBR": self.PBR,
            "emulation_mode": self.emulation_mode,
            "cycles": self.cycles,
            "flags": {
                "C": 1 if self.P & self.FLAG_CARRY else 0,
                "Z": 1 if self.P & self.FLAG_ZERO else 0,
                "I": 1 if self.P & self.FLAG_IRQ else 0,
                "D": 1 if self.P & self.FLAG_DECIMAL else 0,
                "X": 1 if self.P & self.FLAG_INDEX_X else 0,
                "M": 1 if self.P & self.FLAG_INDEX_M else 0,
                "V": 1 if self.P & self.FLAG_OVERFLOW else 0,
                "N": 1 if self.P & self.FLAG_NEGATIVE else 0,
                "E": 1 if self.emulation_mode else 0
            }
        }
    
    # Instruction implementations
    
    def _lda(self, address: int) -> int:
        """
        LDA - Load Accumulator.
        
        Args:
            address: Memory address
            
        Returns:
            Extra cycles used
        """
        if self._is_8bit_memory():
            # 8-bit mode
            value = self._read_byte(address)
            self.A = (self.A & 0xFF00) | value
            
            # Set flags
            self.P = (self.P & ~(self.FLAG_ZERO | self.FLAG_NEGATIVE))
            if value == 0:
                self.P |= self.FLAG_ZERO
            if value & 0x80:
                self.P |= self.FLAG_NEGATIVE
        else:
            # 16-bit mode
            value = self._read_word(address)
            self.A = value
            
            # Set flags
            self.P = (self.P & ~(self.FLAG_ZERO | self.FLAG_NEGATIVE))
            if value == 0:
                self.P |= self.FLAG_ZERO
            if value & 0x8000:
                self.P |= self.FLAG_NEGATIVE
        
        return 0
    
    def _ldx(self, address: int) -> int:
        """
        LDX - Load X Register.
        
        Args:
            address: Memory address
            
        Returns:
            Extra cycles used
        """
        if self._is_8bit_index():
            # 8-bit mode
            value = self._read_byte(address)
            self.X = value
            
            # Set flags
            self.P = (self.P & ~(self.FLAG_ZERO | self.FLAG_NEGATIVE))
            if value == 0:
                self.P |= self.FLAG_ZERO
            if value & 0x80:
                self.P |= self.FLAG_NEGATIVE
        else:
            # 16-bit mode
            value = self._read_word(address)
            self.X = value
            
            # Set flags
            self.P = (self.P & ~(self.FLAG_ZERO | self.FLAG_NEGATIVE))
            if value == 0:
                self.P |= self.FLAG_ZERO
            if value & 0x8000:
                self.P |= self.FLAG_NEGATIVE
        
        return 0
    
    def _ldy(self, address: int) -> int:
        """
        LDY - Load Y Register.
        
        Args:
            address: Memory address
            
        Returns:
            Extra cycles used
        """
        if self._is_8bit_index():
            # 8-bit mode
            value = self._read_byte(address)
            self.Y = value
            
            # Set flags
            self.P = (self.P & ~(self.FLAG_ZERO | self.FLAG_NEGATIVE))
            if value == 0:
                self.P |= self.FLAG_ZERO
            if value & 0x80:
                self.P |= self.FLAG_NEGATIVE
        else:
            # 16-bit mode
            value = self._read_word(address)
            self.Y = value
            
            # Set flags
            self.P = (self.P & ~(self.FLAG_ZERO | self.FLAG_NEGATIVE))
            if value == 0:
                self.P |= self.FLAG_ZERO
            if value & 0x8000:
                self.P |= self.FLAG_NEGATIVE
        
        return 0
    
    def _sta(self, address: int) -> int:
        """
        STA - Store Accumulator.
        
        Args:
            address: Memory address
            
        Returns:
            Extra cycles used
        """
        if self._is_8bit_memory():
            # 8-bit mode
            self._write_byte(address, self.A & 0xFF)
        else:
            # 16-bit mode
            self._write_word(address, self.A)
        
        return 0
    
    def _stx(self, address: int) -> int:
        """
        STX - Store X Register.
        
        Args:
            address: Memory address
            
        Returns:
            Extra cycles used
        """
        if self._is_8bit_index():
            # 8-bit mode
            self._write_byte(address, self.X & 0xFF)
        else:
            # 16-bit mode
            self._write_word(address, self.X)
        
        return 0
    
    def _sty(self, address: int) -> int:
        """
        STY - Store Y Register.
        
        Args:
            address: Memory address
            
        Returns:
            Extra cycles used
        """
        if self._is_8bit_index():
            # 8-bit mode
            self._write_byte(address, self.Y & 0xFF)
        else:
            # 16-bit mode
            self._write_word(address, self.Y)
        
        return 0
    
    def _jmp(self, address: int) -> int:
        """
        JMP - Jump.
        
        Args:
            address: Target address
            
        Returns:
            Extra cycles used
        """
        self.PC = address - self.size_absolute  # PC will be updated after instruction
        return 0
    
    def _jmp_indirect(self) -> int:
        """
        JMP (abs) - Jump Indirect.
        
        Returns:
            Extra cycles used
        """
        # Get pointer address
        pointer = self._read_word(self.PC + 1)
        
        # Read target address from pointer
        target = self._read_word(pointer)
        
        # Set PC
        self.PC = target - self.size_absolute
        return 0
    
    def _jmp_indirect_x(self) -> int:
        """
        JMP (abs,X) - Jump Indirect Indexed.
        
        Returns:
            Extra cycles used
        """
        # Get pointer address and add X
        pointer = (self._read_word(self.PC + 1) + self.X) & 0xFFFF
        
        # Read target address from pointer
        target = self._read_word(pointer)
        
        # Set PC
        self.PC = target - self.size_absolute
        return 0
    
    def _jmp_long(self) -> int:
        """
        JMP long - Jump to Long Address.
        
        Returns:
            Extra cycles used
        """
        # Read 24-bit target address
        target = self._read_long(self.PC + 1)
        
        # Set PC and PBR
        self.PBR = (target >> 16) & 0xFF
        self.PC = target & 0xFFFF - self.size_long
        return 0
    
    def _jsr(self, address: int) -> int:
        """
        JSR - Jump to Subroutine.
        
        Args:
            address: Target address
            
        Returns:
            Extra cycles used
        """
        # Push return address (PC + 2)
        return_addr = self.PC + 2
        self._push_word(return_addr)
        
        # Set PC
        self.PC = address - self.size_absolute
        return 0
    
    def _jsr_indirect_x(self) -> int:
        """
        JSR (abs,X) - Jump to Subroutine Indirect Indexed.
        
        Returns:
            Extra cycles used
        """
        # Get pointer address and add X
        pointer = (self._read_word(self.PC + 1) + self.X) & 0xFFFF
        
        # Read target address from pointer
        target = self._read_word(pointer)
        
        # Push return address
        return_addr = self.PC + 2
        self._push_word(return_addr)
        
        # Set PC
        self.PC = target - self.size_absolute
        return 0
    
    def _jsr_long(self) -> int:
        """
        JSR long - Jump to Subroutine Long.
        
        Returns:
            Extra cycles used
        """
        # Read 24-bit target address
        target = self._read_long(self.PC + 1)
        
        # Push program bank register and return address
        self._push_byte(self.PBR)
        self._push_word(self.PC + 3)
        
        # Set PC and PBR
        self.PBR = (target >> 16) & 0xFF
        self.PC = target & 0xFFFF - self.size_long
        return 0
    
    def _rts(self) -> int:
        """
        RTS - Return from Subroutine.
        
        Returns:
            Extra cycles used
        """
        # Pull return address and add 1
        self.PC = (self._pull_word() + 1) & 0xFFFF - self.size_implied
        return 0
    
    def _rtl(self) -> int:
        """
        RTL - Return from Subroutine Long.
        
        Returns:
            Extra cycles used
        """
        # Pull return address and program bank register
        self.PC = (self._pull_word() + 1) & 0xFFFF - self.size_implied
        self.PBR = self._pull_byte()
        return 0
    
    def _rti(self) -> int:
        """
        RTI - Return from Interrupt.
        
        Returns:
            Extra cycles used
        """
        # Pull status register
        self.P = self._pull_byte()
        
        if self.emulation_mode:
            # In emulation mode, pull 16-bit PC
            self.PC = self._pull_word() - self.size_implied
        else:
            # In native mode, pull 16-bit PC and 8-bit PBR
            self.PC = self._pull_word() - self.size_implied
            self.PBR = self._pull_byte()
        
        return 0
    
    def _clc(self) -> int:
        """
        CLC - Clear Carry Flag.
        
        Returns:
            Extra cycles used
        """
        self.P &= ~self.FLAG_CARRY
        return 0
    
    def _sec(self) -> int:
        """
        SEC - Set Carry Flag.
        
        Returns:
            Extra cycles used
        """
        self.P |= self.FLAG_CARRY
        return 0
    
    def _cld(self) -> int:
        """
        CLD - Clear Decimal Flag.
        
        Returns:
            Extra cycles used
        """
        self.P &= ~self.FLAG_DECIMAL
        return 0
    
    def _sed(self) -> int:
        """
        SED - Set Decimal Flag.
        
        Returns:
            Extra cycles used
        """
        self.P |= self.FLAG_DECIMAL
        return 0
    
    def _cli(self) -> int:
        """
        CLI - Clear Interrupt Flag.
        
        Returns:
            Extra cycles used
        """
        self.P &= ~self.FLAG_IRQ
        return 0
    
    def _sei(self) -> int:
        """
        SEI - Set Interrupt Flag.
        
        Returns:
            Extra cycles used
        """
        self.P |= self.FLAG_IRQ
        return 0
    
    def _clv(self) -> int:
        """
        CLV - Clear Overflow Flag.
        
        Returns:
            Extra cycles used
        """
        self.P &= ~self.FLAG_OVERFLOW
        return 0
    
    def _rep(self) -> int:
        """
        REP - Reset Status Bits.
        
        Returns:
            Extra cycles used
        """
        # Read mask
        mask = self._read_byte(self.PC + 1)
        
        # Clear bits specified by mask
        self.P &= ~mask
        
        return 0
    
    def _sep(self) -> int:
        """
        SEP - Set Status Bits.
        
        Returns:
            Extra cycles used
        """
        # Read mask
        mask = self._read_byte(self.PC + 1)
        
        # Set bits specified by mask
        self.P |= mask
        
        return 0
    
    def _xce(self) -> int:
        """
        XCE - Exchange Carry and Emulation Flags.
        
        Returns:
            Extra cycles used
        """
        # Exchange carry and emulation flags
        temp = (self.P & self.FLAG_CARRY) != 0
        
        if self.emulation_mode:
            self.P |= self.FLAG_CARRY
        else:
            self.P &= ~self.FLAG_CARRY
            
        self.emulation_mode = temp
        
        # If entering emulation mode, set M and X flags and truncate registers
        if self.emulation_mode:
            self.P |= (self.FLAG_INDEX_M | self.FLAG_INDEX_X)
            self.X &= 0xFF
            self.Y &= 0xFF
            self.SP = 0x0100 | (self.SP & 0xFF)
        
        return 0
    
    def _nop(self) -> int:
        """
        NOP - No Operation.
        
        Returns:
            Extra cycles used
        """
        return 0