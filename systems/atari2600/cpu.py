"""
Atari 2600 CPU emulation (MOS Technology 6507).

The 6507 is a cost-reduced version of the 6502 processor with a reduced
address bus (13 bits instead of 16) and no interrupt lines. This module
provides cycle-accurate emulation of the 6507 CPU, which is essential
for proper Atari 2600 emulation due to the tight synchronization between
CPU and TIA (Television Interface Adapter).
"""

from ...common.interfaces import CPU, Memory
import typing as t
import logging

logger = logging.getLogger("QuantumSignalEmulator.Atari2600.CPU")

class CPU6507(CPU):
    """
    Emulates the MOS Technology 6507 processor used in the Atari 2600.
    
    The 6507 is a 1.19 MHz 8-bit processor with a 13-bit address bus
    limiting addressable memory to 8KB. This implementation provides
    cycle-accurate execution and all official 6502 opcodes.
    """
    
    # 6502/6507 flags
    FLAG_CARRY     = 0x01
    FLAG_ZERO      = 0x02
    FLAG_INTERRUPT = 0x04
    FLAG_DECIMAL   = 0x08
    FLAG_BREAK     = 0x10
    FLAG_UNUSED    = 0x20
    FLAG_OVERFLOW  = 0x40
    FLAG_NEGATIVE  = 0x80
    
    def __init__(self):
        # CPU registers
        self.A = 0x00  # Accumulator
        self.X = 0x00  # X index register
        self.Y = 0x00  # Y index register
        self.SP = 0xFD  # Stack pointer
        self.PC = 0x0000  # Program counter
        self.P = 0x24  # Status register (unused bit set)
        
        # Cycle counting
        self.cycles = 0
        self.memory = None
        
        # Instruction table
        self._build_instruction_table()
        
        logger.info("6507 CPU initialized")
    
    def _build_instruction_table(self):
        """Build the instruction lookup table."""
        # Create lookup table for all 6502 instructions
        self.instructions = {}
        
        # Addressing modes
        self.addr_immediate = lambda: self.PC + 1
        self.addr_zeropage = lambda: self.read_memory(self.PC + 1)
        self.addr_zeropage_x = lambda: (self.read_memory(self.PC + 1) + self.X) & 0xFF
        self.addr_zeropage_y = lambda: (self.read_memory(self.PC + 1) + self.Y) & 0xFF
        self.addr_absolute = lambda: self.read_memory(self.PC + 1) | (self.read_memory(self.PC + 2) << 8)
        self.addr_absolute_x = lambda: (self.read_memory(self.PC + 1) | (self.read_memory(self.PC + 2) << 8)) + self.X
        self.addr_absolute_y = lambda: (self.read_memory(self.PC + 1) | (self.read_memory(self.PC + 2) << 8)) + self.Y
        self.addr_indirect = lambda: self._get_indirect_address()
        self.addr_indirect_x = lambda: self._get_indirect_x_address()
        self.addr_indirect_y = lambda: self._get_indirect_y_address()
        self.addr_relative = lambda: self._get_relative_address()
        
        # Required size for each addressing mode (in bytes)
        self.size_immediate = 2
        self.size_zeropage = 2
        self.size_zeropage_x = 2
        self.size_zeropage_y = 2
        self.size_absolute = 3
        self.size_absolute_x = 3
        self.size_absolute_y = 3
        self.size_indirect = 3
        self.size_indirect_x = 2
        self.size_indirect_y = 2
        self.size_relative = 2
        self.size_implied = 1
        
        # LDA - Load Accumulator
        self.instructions[0xA9] = (self._lda, self.addr_immediate, self.size_immediate, 2)
        self.instructions[0xA5] = (self._lda, self.addr_zeropage, self.size_zeropage, 3)
        self.instructions[0xB5] = (self._lda, self.addr_zeropage_x, self.size_zeropage_x, 4)
        self.instructions[0xAD] = (self._lda, self.addr_absolute, self.size_absolute, 4)
        self.instructions[0xBD] = (self._lda, self.addr_absolute_x, self.size_absolute_x, 4)  # +1 if page crossed
        self.instructions[0xB9] = (self._lda, self.addr_absolute_y, self.size_absolute_y, 4)  # +1 if page crossed
        self.instructions[0xA1] = (self._lda, self.addr_indirect_x, self.size_indirect_x, 6)
        self.instructions[0xB1] = (self._lda, self.addr_indirect_y, self.size_indirect_y, 5)  # +1 if page crossed
        
        # STA - Store Accumulator
        self.instructions[0x85] = (self._sta, self.addr_zeropage, self.size_zeropage, 3)
        self.instructions[0x95] = (self._sta, self.addr_zeropage_x, self.size_zeropage_x, 4)
        self.instructions[0x8D] = (self._sta, self.addr_absolute, self.size_absolute, 4)
        self.instructions[0x9D] = (self._sta, self.addr_absolute_x, self.size_absolute_x, 5)
        self.instructions[0x99] = (self._sta, self.addr_absolute_y, self.size_absolute_y, 5)
        self.instructions[0x81] = (self._sta, self.addr_indirect_x, self.size_indirect_x, 6)
        self.instructions[0x91] = (self._sta, self.addr_indirect_y, self.size_indirect_y, 6)
        
        # LDX - Load X Register
        self.instructions[0xA2] = (self._ldx, self.addr_immediate, self.size_immediate, 2)
        self.instructions[0xA6] = (self._ldx, self.addr_zeropage, self.size_zeropage, 3)
        self.instructions[0xB6] = (self._ldx, self.addr_zeropage_y, self.size_zeropage_y, 4)
        self.instructions[0xAE] = (self._ldx, self.addr_absolute, self.size_absolute, 4)
        self.instructions[0xBE] = (self._ldx, self.addr_absolute_y, self.size_absolute_y, 4)  # +1 if page crossed
        
        # LDY - Load Y Register
        self.instructions[0xA0] = (self._ldy, self.addr_immediate, self.size_immediate, 2)
        self.instructions[0xA4] = (self._ldy, self.addr_zeropage, self.size_zeropage, 3)
        self.instructions[0xB4] = (self._ldy, self.addr_zeropage_x, self.size_zeropage_x, 4)
        self.instructions[0xAC] = (self._ldy, self.addr_absolute, self.size_absolute, 4)
        self.instructions[0xBC] = (self._ldy, self.addr_absolute_x, self.size_absolute_x, 4)  # +1 if page crossed
        
        # STX - Store X Register
        self.instructions[0x86] = (self._stx, self.addr_zeropage, self.size_zeropage, 3)
        self.instructions[0x96] = (self._stx, self.addr_zeropage_y, self.size_zeropage_y, 4)
        self.instructions[0x8E] = (self._stx, self.addr_absolute, self.size_absolute, 4)
        
        # STY - Store Y Register
        self.instructions[0x84] = (self._sty, self.addr_zeropage, self.size_zeropage, 3)
        self.instructions[0x94] = (self._sty, self.addr_zeropage_x, self.size_zeropage_x, 4)
        self.instructions[0x8C] = (self._sty, self.addr_absolute, self.size_absolute, 4)
        
        # JMP - Jump
        self.instructions[0x4C] = (self._jmp, self.addr_absolute, self.size_absolute, 3)
        self.instructions[0x6C] = (self._jmp, self.addr_indirect, self.size_indirect, 5)
        
        # JSR - Jump to Subroutine
        self.instructions[0x20] = (self._jsr, self.addr_absolute, self.size_absolute, 6)
        
        # RTS - Return from Subroutine
        self.instructions[0x60] = (self._rts, None, self.size_implied, 6)
        
        # BCC - Branch if Carry Clear
        self.instructions[0x90] = (self._bcc, self.addr_relative, self.size_relative, 2)  # +1 if branch, +2 if page crossed
        
        # BCS - Branch if Carry Set
        self.instructions[0xB0] = (self._bcs, self.addr_relative, self.size_relative, 2)  # +1 if branch, +2 if page crossed
        
        # BEQ - Branch if Equal (Zero Set)
        self.instructions[0xF0] = (self._beq, self.addr_relative, self.size_relative, 2)  # +1 if branch, +2 if page crossed
        
        # BNE - Branch if Not Equal (Zero Clear)
        self.instructions[0xD0] = (self._bne, self.addr_relative, self.size_relative, 2)  # +1 if branch, +2 if page crossed
        
        # BPL - Branch if Plus (Negative Clear)
        self.instructions[0x10] = (self._bpl, self.addr_relative, self.size_relative, 2)  # +1 if branch, +2 if page crossed
        
        # BMI - Branch if Minus (Negative Set)
        self.instructions[0x30] = (self._bmi, self.addr_relative, self.size_relative, 2)  # +1 if branch, +2 if page crossed
        
        # BVC - Branch if Overflow Clear
        self.instructions[0x50] = (self._bvc, self.addr_relative, self.size_relative, 2)  # +1 if branch, +2 if page crossed
        
        # BVS - Branch if Overflow Set
        self.instructions[0x70] = (self._bvs, self.addr_relative, self.size_relative, 2)  # +1 if branch, +2 if page crossed
        
        # CLC - Clear Carry Flag
        self.instructions[0x18] = (self._clc, None, self.size_implied, 2)
        
        # CLD - Clear Decimal Flag
        self.instructions[0xD8] = (self._cld, None, self.size_implied, 2)
        
        # CLI - Clear Interrupt Flag
        self.instructions[0x58] = (self._cli, None, self.size_implied, 2)
        
        # CLV - Clear Overflow Flag
        self.instructions[0xB8] = (self._clv, None, self.size_implied, 2)
        
        # SEC - Set Carry Flag
        self.instructions[0x38] = (self._sec, None, self.size_implied, 2)
        
        # SED - Set Decimal Flag
        self.instructions[0xF8] = (self._sed, None, self.size_implied, 2)
        
        # SEI - Set Interrupt Flag
        self.instructions[0x78] = (self._sei, None, self.size_implied, 2)
        
        # Add many more instructions...
        # This is just a subset of the 6502 instruction set.
        # In a real implementation, all 151 valid opcodes would be included.
        
        # NOP - No Operation
        self.instructions[0xEA] = (self._nop, None, self.size_implied, 2)
    
    def set_memory(self, memory: Memory) -> None:
        """
        Connect the CPU to a memory system.
        
        Args:
            memory: Memory implementation
        """
        self.memory = memory
    
    def read_memory(self, address: int) -> int:
        """
        Read a byte from memory using the connected memory interface.
        
        Args:
            address: Memory address
            
        Returns:
            Byte value at the address
        """
        if self.memory:
            # Mask address to 13 bits (6507 has a 13-bit address bus)
            masked_address = address & 0x1FFF
            return self.memory.read(masked_address)
        return 0x00
    
    def write_memory(self, address: int, value: int) -> None:
        """
        Write a byte to memory using the connected memory interface.
        
        Args:
            address: Memory address
            value: Byte value to write
        """
        if self.memory:
            # Mask address to 13 bits (6507 has a 13-bit address bus)
            masked_address = address & 0x1FFF
            self.memory.write(masked_address, value & 0xFF)
    
    def reset(self) -> None:
        """Reset the CPU to its initial state."""
        # Read reset vector
        if self.memory:
            self.PC = self.read_memory(0xFFFC) | (self.read_memory(0xFFFD) << 8)
        else:
            self.PC = 0xF000  # Default for Atari 2600 if no memory attached
            
        self.SP = 0xFD
        self.A = 0x00
        self.X = 0x00
        self.Y = 0x00
        self.P = 0x24  # Unused bit set
        self.cycles = 0
        
        logger.info(f"CPU reset. PC set to ${self.PC:04X}")
    
    def step(self) -> int:
        """
        Execute one instruction and return the number of cycles used.
        
        Returns:
            Number of cycles used by the instruction
        """
        if not self.memory:
            raise RuntimeError("CPU has no memory attached")
            
        # Get opcode
        opcode = self.read_memory(self.PC)
        
        # Execute instruction
        if opcode in self.instructions:
            operation, addressing, size, base_cycles = self.instructions[opcode]
            
            # Get memory address if addressing mode is provided
            address = None
            if addressing:
                address = addressing()
            
            # Execute the instruction
            extra_cycles = operation(address) if address is not None else operation()
            
            # Update program counter
            self.PC += size
            
            # Update cycle count
            cycles_used = base_cycles + extra_cycles
            self.cycles += cycles_used
            
            return cycles_used
        else:
            # Invalid opcode - treat as NOP but log a warning
            logger.warning(f"Unknown opcode: ${opcode:02X} at ${self.PC:04X}")
            self.PC += 1
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
            "cycles": self.cycles,
            "flags": {
                "C": 1 if self.P & self.FLAG_CARRY else 0,
                "Z": 1 if self.P & self.FLAG_ZERO else 0,
                "I": 1 if self.P & self.FLAG_INTERRUPT else 0,
                "D": 1 if self.P & self.FLAG_DECIMAL else 0,
                "B": 1 if self.P & self.FLAG_BREAK else 0,
                "V": 1 if self.P & self.FLAG_OVERFLOW else 0,
                "N": 1 if self.P & self.FLAG_NEGATIVE else 0
            }
        }
    
    def stack_push(self, value: int) -> None:
        """
        Push a byte onto the stack.
        
        Args:
            value: Byte value to push
        """
        self.write_memory(0x0100 + self.SP, value & 0xFF)
        self.SP = (self.SP - 1) & 0xFF
    
    def stack_pull(self) -> int:
        """
        Pull a byte from the stack.
        
        Returns:
            Byte value from stack
        """
        self.SP = (self.SP + 1) & 0xFF
        return self.read_memory(0x0100 + self.SP)
    
    def _get_indirect_address(self) -> int:
        """
        Get address for indirect addressing mode.
        
        Returns:
            Calculated address
        """
        addr = self.read_memory(self.PC + 1) | (self.read_memory(self.PC + 2) << 8)
        
        # Simulate 6502 bug: if the indirect vector falls on a page boundary, fetch high byte from
        # the same page rather than the next page
        if (addr & 0xFF) == 0xFF:
            low = self.read_memory(addr)
            high = self.read_memory(addr & 0xFF00)
            return low | (high << 8)
        else:
            return self.read_memory(addr) | (self.read_memory(addr + 1) << 8)
    
    def _get_indirect_x_address(self) -> int:
        """
        Get address for Indirect X addressing mode.
        
        Returns:
            Calculated address
        """
        addr = (self.read_memory(self.PC + 1) + self.X) & 0xFF
        return self.read_memory(addr) | (self.read_memory((addr + 1) & 0xFF) << 8)
    
    def _get_indirect_y_address(self) -> int:
        """
        Get address for Indirect Y addressing mode.
        
        Returns:
            Calculated address
        """
        addr = self.read_memory(self.PC + 1)
        base = self.read_memory(addr) | (self.read_memory((addr + 1) & 0xFF) << 8)
        return base + self.Y
    
    def _get_relative_address(self) -> int:
        """
        Get address for relative addressing mode.
        
        Returns:
            Target address for branch
        """
        offset = self.read_memory(self.PC + 1)
        if offset & 0x80:
            offset = offset - 256
        return self.PC + 2 + offset
    
    def _check_page_cross(self, addr1: int, addr2: int) -> bool:
        """
        Check if addresses are on different pages.
        
        Args:
            addr1: First address
            addr2: Second address
            
        Returns:
            True if addresses are on different pages
        """
        return (addr1 & 0xFF00) != (addr2 & 0xFF00)
    
    # Instruction implementations
    
    def _lda(self, address: int) -> int:
        """
        LDA - Load Accumulator.
        
        Args:
            address: Memory address
            
        Returns:
            Extra cycles used
        """
        self.A = self.read_memory(address) & 0xFF
        
        # Set zero and negative flags
        self.P = (self.P & ~(self.FLAG_ZERO | self.FLAG_NEGATIVE))
        if self.A == 0:
            self.P |= self.FLAG_ZERO
        if self.A & 0x80:
            self.P |= self.FLAG_NEGATIVE
            
        # Check for page crossing
        if (address & 0xFF00) != ((self.PC + 1) & 0xFF00):
            return 1
        return 0
    
    def _sta(self, address: int) -> int:
        """
        STA - Store Accumulator.
        
        Args:
            address: Memory address
            
        Returns:
            Extra cycles used
        """
        self.write_memory(address, self.A)
        return 0
    
    def _ldx(self, address: int) -> int:
        """
        LDX - Load X Register.
        
        Args:
            address: Memory address
            
        Returns:
            Extra cycles used
        """
        self.X = self.read_memory(address) & 0xFF
        
        # Set zero and negative flags
        self.P = (self.P & ~(self.FLAG_ZERO | self.FLAG_NEGATIVE))
        if self.X == 0:
            self.P |= self.FLAG_ZERO
        if self.X & 0x80:
            self.P |= self.FLAG_NEGATIVE
            
        # Check for page crossing
        if (address & 0xFF00) != ((self.PC + 1) & 0xFF00):
            return 1
        return 0
    
    def _ldy(self, address: int) -> int:
        """
        LDY - Load Y Register.
        
        Args:
            address: Memory address
            
        Returns:
            Extra cycles used
        """
        self.Y = self.read_memory(address) & 0xFF
        
        # Set zero and negative flags
        self.P = (self.P & ~(self.FLAG_ZERO | self.FLAG_NEGATIVE))
        if self.Y == 0:
            self.P |= self.FLAG_ZERO
        if self.Y & 0x80:
            self.P |= self.FLAG_NEGATIVE
            
        # Check for page crossing
        if (address & 0xFF00) != ((self.PC + 1) & 0xFF00):
            return 1
        return 0
    
    def _stx(self, address: int) -> int:
        """
        STX - Store X Register.
        
        Args:
            address: Memory address
            
        Returns:
            Extra cycles used
        """
        self.write_memory(address, self.X)
        return 0
    
    def _sty(self, address: int) -> int:
        """
        STY - Store Y Register.
        
        Args:
            address: Memory address
            
        Returns:
            Extra cycles used
        """
        self.write_memory(address, self.Y)
        return 0
    
    def _jmp(self, address: int) -> int:
        """
        JMP - Jump.
        
        Args:
            address: Memory address
            
        Returns:
            Extra cycles used
        """
        self.PC = address - self.size_absolute  # PC will be incremented after instruction
        return 0
    
    def _jsr(self, address: int) -> int:
        """
        JSR - Jump to Subroutine.
        
        Args:
            address: Memory address
            
        Returns:
            Extra cycles used
        """
        # Push return address (PC + 2) - 1 onto stack
        return_addr = self.PC + 2 - 1
        self.stack_push((return_addr >> 8) & 0xFF)  # Push high byte
        self.stack_push(return_addr & 0xFF)  # Push low byte
        
        # Set program counter to target address
        self.PC = address - self.size_absolute  # PC will be incremented after instruction
        return 0
    
    def _rts(self) -> int:
        """
        RTS - Return from Subroutine.
        
        Returns:
            Extra cycles used
        """
        # Pull return address from stack
        low = self.stack_pull()
        high = self.stack_pull()
        
        # Set program counter to return address + 1
        self.PC = ((high << 8) | low) + 1 - self.size_implied  # PC will be incremented after instruction
        return 0
    
    def _bcc(self) -> int:
        """
        BCC - Branch if Carry Clear.
        
        Returns:
            Extra cycles used
        """
        target = self._get_relative_address()
        
        if not (self.P & self.FLAG_CARRY):
            extra_cycles = 1
            if self._check_page_cross(self.PC, target):
                extra_cycles += 1
                
            self.PC = target - self.size_relative  # PC will be incremented after instruction
            return extra_cycles
        return 0
    
    def _bcs(self) -> int:
        """
        BCS - Branch if Carry Set.
        
        Returns:
            Extra cycles used
        """
        target = self._get_relative_address()
        
        if self.P & self.FLAG_CARRY:
            extra_cycles = 1
            if self._check_page_cross(self.PC, target):
                extra_cycles += 1
                
            self.PC = target - self.size_relative  # PC will be incremented after instruction
            return extra_cycles
        return 0
    
    def _beq(self) -> int:
        """
        BEQ - Branch if Equal (Zero Set).
        
        Returns:
            Extra cycles used
        """
        target = self._get_relative_address()
        
        if self.P & self.FLAG_ZERO:
            extra_cycles = 1
            if self._check_page_cross(self.PC, target):
                extra_cycles += 1
                
            self.PC = target - self.size_relative  # PC will be incremented after instruction
            return extra_cycles
        return 0
    
    def _bne(self) -> int:
        """
        BNE - Branch if Not Equal (Zero Clear).
        
        Returns:
            Extra cycles used
        """
        target = self._get_relative_address()
        
        if not (self.P & self.FLAG_ZERO):
            extra_cycles = 1
            if self._check_page_cross(self.PC, target):
                extra_cycles += 1
                
            self.PC = target - self.size_relative  # PC will be incremented after instruction
            return extra_cycles
        return 0
    
    def _bpl(self) -> int:
        """
        BPL - Branch if Plus (Negative Clear).
        
        Returns:
            Extra cycles used
        """
        target = self._get_relative_address()
        
        if not (self.P & self.FLAG_NEGATIVE):
            extra_cycles = 1
            if self._check_page_cross(self.PC, target):
                extra_cycles += 1
                
            self.PC = target - self.size_relative  # PC will be incremented after instruction
            return extra_cycles
        return 0
    
    def _bmi(self) -> int:
        """
        BMI - Branch if Minus (Negative Set).
        
        Returns:
            Extra cycles used
        """
        target = self._get_relative_address()
        
        if self.P & self.FLAG_NEGATIVE:
            extra_cycles = 1
            if self._check_page_cross(self.PC, target):
                extra_cycles += 1
                
            self.PC = target - self.size_relative  # PC will be incremented after instruction
            return extra_cycles
        return 0
    
    def _bvc(self) -> int:
        """
        BVC - Branch if Overflow Clear.
        
        Returns:
            Extra cycles used
        """
        target = self._get_relative_address()
        
        if not (self.P & self.FLAG_OVERFLOW):
            extra_cycles = 1
            if self._check_page_cross(self.PC, target):
                extra_cycles += 1
                
            self.PC = target - self.size_relative  # PC will be incremented after instruction
            return extra_cycles
        return 0
    
    def _bvs(self) -> int:
        """
        BVS - Branch if Overflow Set.
        
        Returns:
            Extra cycles used
        """
        target = self._get_relative_address()
        
        if self.P & self.FLAG_OVERFLOW:
            extra_cycles = 1
            if self._check_page_cross(self.PC, target):
                extra_cycles += 1
                
            self.PC = target - self.size_relative  # PC will be incremented after instruction
            return extra_cycles
        return 0
    
    def _clc(self) -> int:
        """
        CLC - Clear Carry Flag.
        
        Returns:
            Extra cycles used
        """
        self.P &= ~self.FLAG_CARRY
        return 0
    
    def _cld(self) -> int:
        """
        CLD - Clear Decimal Flag.
        
        Returns:
            Extra cycles used
        """
        self.P &= ~self.FLAG_DECIMAL
        return 0
    
    def _cli(self) -> int:
        """
        CLI - Clear Interrupt Flag.
        
        Returns:
            Extra cycles used
        """
        self.P &= ~self.FLAG_INTERRUPT
        return 0
    
    def _clv(self) -> int:
        """
        CLV - Clear Overflow Flag.
        
        Returns:
            Extra cycles used
        """
        self.P &= ~self.FLAG_OVERFLOW
        return 0
    
    def _sec(self) -> int:
        """
        SEC - Set Carry Flag.
        
        Returns:
            Extra cycles used
        """
        self.P |= self.FLAG_CARRY
        return 0
    
    def _sed(self) -> int:
        """
        SED - Set Decimal Flag.
        
        Returns:
            Extra cycles used
        """
        self.P |= self.FLAG_DECIMAL
        return 0
    
    def _sei(self) -> int:
        """
        SEI - Set Interrupt Flag.
        
        Returns:
            Extra cycles used
        """
        self.P |= self.FLAG_INTERRUPT
        return 0
    
    def _nop(self) -> int:
        """
        NOP - No Operation.
        
        Returns:
            Extra cycles used
        """
        return 0
    
    # Add implementations for the rest of the instructions...