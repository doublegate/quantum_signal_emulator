"""
Sega Genesis/Mega Drive Z80 CPU emulation.

The Genesis uses a Z80 CPU running at 3.58 MHz as a secondary processor,
primarily for sound control. This module provides a cycle-accurate emulation
of the Z80 CPU with all its instructions and addressing modes.
"""

from ...common.interfaces import CPU, Memory
import logging
from typing import Dict, List, Tuple, Optional, Any, Union

logger = logging.getLogger("QuantumSignalEmulator.Genesis.Z80CPU")

class Z80CPU(CPU):
    """
    Emulates the Z80 processor used in the Sega Genesis/Mega Drive.
    
    The Z80 is an 8-bit CPU with a 16-bit address bus, running at ~3.58 MHz
    in the Genesis. It's primarily used for sound control and can access a
    limited portion of the 68000 memory space.
    """
    
    # Z80 flags
    FLAG_C = 0x01  # Carry
    FLAG_N = 0x02  # Subtract
    FLAG_P = 0x04  # Parity/Overflow
    FLAG_H = 0x10  # Half Carry
    FLAG_Z = 0x40  # Zero
    FLAG_S = 0x80  # Sign
    
    def __init__(self):
        """Initialize the Z80 CPU."""
        # Main registers
        self.a = 0        # Accumulator
        self.f = 0        # Flags
        self.b = 0        # B register
        self.c = 0        # C register
        self.d = 0        # D register
        self.e = 0        # E register
        self.h = 0        # H register
        self.l = 0        # L register
        
        # Special registers
        self.i = 0        # Interrupt vector
        self.r = 0        # Memory refresh
        self.ix = 0       # Index register X
        self.iy = 0       # Index register Y
        self.sp = 0xFFFF  # Stack pointer
        self.pc = 0       # Program counter
        
        # Alternate register set
        self.a_alt = 0
        self.f_alt = 0
        self.b_alt = 0
        self.c_alt = 0
        self.d_alt = 0
        self.e_alt = 0
        self.h_alt = 0
        self.l_alt = 0
        
        # Interrupt state
        self.iff1 = False  # Interrupt flip-flop 1
        self.iff2 = False  # Interrupt flip-flop 2
        self.im = 0        # Interrupt mode
        
        # Connected memory
        self.memory = None
        
        # Cycle counting
        self.cycles = 0
        
        # Halt state
        self.halted = False
        
        # Instruction lookup tables
        self._build_instruction_tables()
        
        logger.info("Z80 CPU initialized")
    
    def _build_instruction_tables(self):
        """Build the instruction lookup tables for all Z80 opcodes."""
        # Create main instruction table for single-byte opcodes
        self.instructions = {}
        
        # This is a simplified implementation
        # In a complete emulator, all Z80 instructions would be included
        # (about 700 instructions including prefixed ones)
        
        # Some examples of essential instructions
        
        # 8-bit load instructions
        # LD r, r'
        for src in range(8):
            for dst in range(8):
                if src == 6 and dst == 6:
                    continue  # Skip invalid HALT opcode
                self.instructions[0x40 | (dst << 3) | src] = (
                    self._ld_r_r, (dst, src), 1, 4
                )
        
        # LD r, n
        for r in range(8):
            if r == 6:
                continue  # Skip (HL) version
            self.instructions[0x06 | (r << 3)] = (
                self._ld_r_n, (r,), 2, 7
            )
        
        # LD A, (BC)
        self.instructions[0x0A] = (self._ld_a_bc, (), 1, 7)
        
        # LD A, (DE)
        self.instructions[0x1A] = (self._ld_a_de, (), 1, 7)
        
        # LD A, (nn)
        self.instructions[0x3A] = (self._ld_a_nn, (), 3, 13)
        
        # PUSH/POP
        self.instructions[0xC5] = (self._push_bc, (), 1, 11)
        self.instructions[0xD5] = (self._push_de, (), 1, 11)
        self.instructions[0xE5] = (self._push_hl, (), 1, 11)
        self.instructions[0xF5] = (self._push_af, (), 1, 11)
        
        self.instructions[0xC1] = (self._pop_bc, (), 1, 10)
        self.instructions[0xD1] = (self._pop_de, (), 1, 10)
        self.instructions[0xE1] = (self._pop_hl, (), 1, 10)
        self.instructions[0xF1] = (self._pop_af, (), 1, 10)
        
        # 8-bit arithmetic instructions
        # ADD A, r
        for r in range(8):
            self.instructions[0x80 | r] = (self._add_a_r, (r,), 1, 4)
        
        # SUB r
        for r in range(8):
            self.instructions[0x90 | r] = (self._sub_r, (r,), 1, 4)
        
        # AND r
        for r in range(8):
            self.instructions[0xA0 | r] = (self._and_r, (r,), 1, 4)
        
        # OR r
        for r in range(8):
            self.instructions[0xB0 | r] = (self._or_r, (r,), 1, 4)
        
        # Control instructions
        self.instructions[0x00] = (self._nop, (), 1, 4)
        self.instructions[0x76] = (self._halt, (), 1, 4)
        
        # Jump instructions
        self.instructions[0xC3] = (self._jp_nn, (), 3, 10)
        self.instructions[0xC2] = (self._jp_nz_nn, (), 3, 10)
        self.instructions[0xCA] = (self._jp_z_nn, (), 3, 10)
        self.instructions[0xD2] = (self._jp_nc_nn, (), 3, 10)
        self.instructions[0xDA] = (self._jp_c_nn, (), 3, 10)
        
        # Call/Return instructions
        self.instructions[0xCD] = (self._call_nn, (), 3, 17)
        self.instructions[0xC9] = (self._ret, (), 1, 10)
        
        # Extended instruction tables (0xCB, 0xDD, 0xED, 0xFD prefixes)
        # would be implemented in a complete emulator
    
    def set_memory(self, memory: Memory) -> None:
        """
        Connect the CPU to a memory system.
        
        Args:
            memory: Memory implementation
        """
        self.memory = memory
    
    def read_byte(self, address: int) -> int:
        """
        Read a byte from memory.
        
        Args:
            address: Memory address
            
        Returns:
            Byte value
        """
        if self.memory:
            # Z80 in the Genesis has its own address space
            if hasattr(self.memory, 'z80_read'):
                return self.memory.z80_read(address & 0xFFFF)
            else:
                # Fallback for simple memory implementations
                return self.memory.read(address & 0xFFFF)
        return 0
    
    def write_byte(self, address: int, value: int) -> None:
        """
        Write a byte to memory.
        
        Args:
            address: Memory address
            value: Byte value to write
        """
        if self.memory:
            # Z80 in the Genesis has its own address space
            if hasattr(self.memory, 'z80_write'):
                self.memory.z80_write(address & 0xFFFF, value & 0xFF)
            else:
                # Fallback for simple memory implementations
                self.memory.write(address & 0xFFFF, value & 0xFF)
    
    def read_word(self, address: int) -> int:
        """
        Read a 16-bit word from memory.
        
        Args:
            address: Memory address
            
        Returns:
            Word value
        """
        low = self.read_byte(address)
        high = self.read_byte((address + 1) & 0xFFFF)
        return (high << 8) | low
    
    def write_word(self, address: int, value: int) -> None:
        """
        Write a 16-bit word to memory.
        
        Args:
            address: Memory address
            value: Word value to write
        """
        self.write_byte(address, value & 0xFF)
        self.write_byte((address + 1) & 0xFFFF, (value >> 8) & 0xFF)
    
    def reset(self) -> None:
        """Reset the CPU to its initial state."""
        # Reset registers
        self.a = 0
        self.f = 0
        self.b = 0
        self.c = 0
        self.d = 0
        self.e = 0
        self.h = 0
        self.l = 0
        
        self.i = 0
        self.r = 0
        self.ix = 0
        self.iy = 0
        self.sp = 0xFFFF
        self.pc = 0
        
        self.a_alt = 0
        self.f_alt = 0
        self.b_alt = 0
        self.c_alt = 0
        self.d_alt = 0
        self.e_alt = 0
        self.h_alt = 0
        self.l_alt = 0
        
        # Reset interrupt state
        self.iff1 = False
        self.iff2 = False
        self.im = 0
        
        # Reset cycle counter
        self.cycles = 0
        
        # Reset halt state
        self.halted = False
        
        logger.info("Z80 CPU reset")
    
    def step(self) -> int:
        """
        Execute one instruction and return the number of cycles used.
        
        Returns:
            Number of cycles used by the instruction
        """
        if not self.memory:
            raise RuntimeError("CPU has no memory attached")
            
        # Check if halted
        if self.halted:
            # When halted, just count cycles without executing
            self.cycles += 4
            return 4
            
        # Fetch opcode
        opcode = self.read_byte(self.pc)
        
        # Check for extended instruction sets
        if opcode == 0xCB:
            # CB-prefixed instructions (bit operations)
            # Not implemented in this simplified version
            self.pc += 1
            return 8
            
        elif opcode == 0xDD or opcode == 0xFD:
            # DD/FD-prefixed instructions (IX/IY operations)
            # Not implemented in this simplified version
            self.pc += 1
            return 8
            
        elif opcode == 0xED:
            # ED-prefixed instructions
            # Not implemented in this simplified version
            self.pc += 1
            return 8
            
        # Execute instruction
        if opcode in self.instructions:
            operation, args, size, base_cycles = self.instructions[opcode]
            
            # Increment PC before execution (for multi-byte opcodes)
            self.pc += size
            
            # Execute the instruction
            extra_cycles = operation(*args)
            
            # Update cycle count
            cycles_used = base_cycles + extra_cycles
            self.cycles += cycles_used
            
            return cycles_used
        else:
            # Unknown opcode - treat as NOP
            logger.warning(f"Unknown Z80 opcode: ${opcode:02X} at ${self.pc:04X}")
            self.pc += 1
            self.cycles += 4
            return 4
    
    def get_state(self) -> dict:
        """
        Get the current CPU state.
        
        Returns:
            Dictionary with CPU state
        """
        return {
            "a": self.a,
            "f": self.f,
            "b": self.b,
            "c": self.c,
            "d": self.d,
            "e": self.e,
            "h": self.h,
            "l": self.l,
            "i": self.i,
            "r": self.r,
            "ix": self.ix,
            "iy": self.iy,
            "sp": self.sp,
            "pc": self.pc,
            "iff1": self.iff1,
            "iff2": self.iff2,
            "im": self.im,
            "cycles": self.cycles,
            "halted": self.halted,
            "flags": {
                "C": 1 if self.f & self.FLAG_C else 0,
                "N": 1 if self.f & self.FLAG_N else 0,
                "P": 1 if self.f & self.FLAG_P else 0,
                "H": 1 if self.f & self.FLAG_H else 0,
                "Z": 1 if self.f & self.FLAG_Z else 0,
                "S": 1 if self.f & self.FLAG_S else 0
            }
        }
    
    # Register getters
    def _get_bc(self) -> int:
        """Get BC register pair."""
        return (self.b << 8) | self.c
    
    def _get_de(self) -> int:
        """Get DE register pair."""
        return (self.d << 8) | self.e
    
    def _get_hl(self) -> int:
        """Get HL register pair."""
        return (self.h << 8) | self.l
    
    def _get_af(self) -> int:
        """Get AF register pair."""
        return (self.a << 8) | self.f
    
    # Register setters
    def _set_bc(self, value: int) -> None:
        """Set BC register pair."""
        self.b = (value >> 8) & 0xFF
        self.c = value & 0xFF
    
    def _set_de(self, value: int) -> None:
        """Set DE register pair."""
        self.d = (value >> 8) & 0xFF
        self.e = value & 0xFF
    
    def _set_hl(self, value: int) -> None:
        """Set HL register pair."""
        self.h = (value >> 8) & 0xFF
        self.l = value & 0xFF
    
    def _set_af(self, value: int) -> None:
        """Set AF register pair."""
        self.a = (value >> 8) & 0xFF
        self.f = value & 0xFF
    
    # Simple register getters for r operand
    def _get_reg(self, r: int) -> int:
        """
        Get register value based on 3-bit register code.
        
        Args:
            r: 3-bit register code (0=B, 1=C, 2=D, 3=E, 4=H, 5=L, 6=(HL), 7=A)
            
        Returns:
            Register value
        """
        if r == 0:
            return self.b
        elif r == 1:
            return self.c
        elif r == 2:
            return self.d
        elif r == 3:
            return self.e
        elif r == 4:
            return self.h
        elif r == 5:
            return self.l
        elif r == 6:
            # Memory access at (HL)
            return self.read_byte(self._get_hl())
        elif r == 7:
            return self.a
        return 0
    
    # Simple register setters for r operand
    def _set_reg(self, r: int, value: int) -> None:
        """
        Set register value based on 3-bit register code.
        
        Args:
            r: 3-bit register code (0=B, 1=C, 2=D, 3=E, 4=H, 5=L, 6=(HL), 7=A)
            value: Value to set
        """
        value &= 0xFF  # Ensure value is 8-bit
        
        if r == 0:
            self.b = value
        elif r == 1:
            self.c = value
        elif r == 2:
            self.d = value
        elif r == 3:
            self.e = value
        elif r == 4:
            self.h = value
        elif r == 5:
            self.l = value
        elif r == 6:
            # Memory access at (HL)
            self.write_byte(self._get_hl(), value)
        elif r == 7:
            self.a = value
    
    # Flag calculation helpers
    def _update_flag_sign_zero(self, value: int) -> None:
        """
        Update Sign and Zero flags based on value.
        
        Args:
            value: Value to check
        """
        # Clear S and Z flags
        self.f &= ~(self.FLAG_S | self.FLAG_Z)
        
        # Set S flag if bit 7 is set
        if value & 0x80:
            self.f |= self.FLAG_S
            
        # Set Z flag if value is zero
        if value == 0:
            self.f |= self.FLAG_Z
    
    def _update_flag_parity(self, value: int) -> None:
        """
        Update Parity flag based on value.
        
        Args:
            value: Value to check
        """
        # Clear P flag
        self.f &= ~self.FLAG_P
        
        # Count bits set
        bits = 0
        for i in range(8):
            if value & (1 << i):
                bits += 1
                
        # Set P flag if even parity (even number of bits set)
        if bits % 2 == 0:
            self.f |= self.FLAG_P
    
    # Basic instruction implementations
    
    def _ld_r_r(self, dst: int, src: int) -> int:
        """
        LD r, r' - Load register with value from another register.
        
        Args:
            dst: Destination register code
            src: Source register code
            
        Returns:
            Extra cycles used
        """
        value = self._get_reg(src)
        self._set_reg(dst, value)
        return 0
    
    def _ld_r_n(self, r: int) -> int:
        """
        LD r, n - Load register with immediate value.
        
        Args:
            r: Register code
            
        Returns:
            Extra cycles used
        """
        value = self.read_byte(self.pc - 1)
        self._set_reg(r, value)
        return 0
    
    def _ld_a_bc(self) -> int:
        """
        LD A, (BC) - Load A with value at address in BC.
        
        Returns:
            Extra cycles used
        """
        self.a = self.read_byte(self._get_bc())
        return 0
    
    def _ld_a_de(self) -> int:
        """
        LD A, (DE) - Load A with value at address in DE.
        
        Returns:
            Extra cycles used
        """
        self.a = self.read_byte(self._get_de())
        return 0
    
    def _ld_a_nn(self) -> int:
        """
        LD A, (nn) - Load A with value at direct address.
        
        Returns:
            Extra cycles used
        """
        address = self.read_word(self.pc - 2)
        self.a = self.read_byte(address)
        return 0
    
    def _push_bc(self) -> int:
        """
        PUSH BC - Push BC onto stack.
        
        Returns:
            Extra cycles used
        """
        self.sp = (self.sp - 1) & 0xFFFF
        self.write_byte(self.sp, self.b)
        self.sp = (self.sp - 1) & 0xFFFF
        self.write_byte(self.sp, self.c)
        return 0
    
    def _push_de(self) -> int:
        """
        PUSH DE - Push DE onto stack.
        
        Returns:
            Extra cycles used
        """
        self.sp = (self.sp - 1) & 0xFFFF
        self.write_byte(self.sp, self.d)
        self.sp = (self.sp - 1) & 0xFFFF
        self.write_byte(self.sp, self.e)
        return 0
    
    def _push_hl(self) -> int:
        """
        PUSH HL - Push HL onto stack.
        
        Returns:
            Extra cycles used
        """
        self.sp = (self.sp - 1) & 0xFFFF
        self.write_byte(self.sp, self.h)
        self.sp = (self.sp - 1) & 0xFFFF
        self.write_byte(self.sp, self.l)
        return 0
    
    def _push_af(self) -> int:
        """
        PUSH AF - Push AF onto stack.
        
        Returns:
            Extra cycles used
        """
        self.sp = (self.sp - 1) & 0xFFFF
        self.write_byte(self.sp, self.a)
        self.sp = (self.sp - 1) & 0xFFFF
        self.write_byte(self.sp, self.f)
        return 0
    
    def _pop_bc(self) -> int:
        """
        POP BC - Pop BC from stack.
        
        Returns:
            Extra cycles used
        """
        self.c = self.read_byte(self.sp)
        self.sp = (self.sp + 1) & 0xFFFF
        self.b = self.read_byte(self.sp)
        self.sp = (self.sp + 1) & 0xFFFF
        return 0
    
    def _pop_de(self) -> int:
        """
        POP DE - Pop DE from stack.
        
        Returns:
            Extra cycles used
        """
        self.e = self.read_byte(self.sp)
        self.sp = (self.sp + 1) & 0xFFFF
        self.d = self.read_byte(self.sp)
        self.sp = (self.sp + 1) & 0xFFFF
        return 0
    
    def _pop_hl(self) -> int:
        """
        POP HL - Pop HL from stack.
        
        Returns:
            Extra cycles used
        """
        self.l = self.read_byte(self.sp)
        self.sp = (self.sp + 1) & 0xFFFF
        self.h = self.read_byte(self.sp)
        self.sp = (self.sp + 1) & 0xFFFF
        return 0
    
    def _pop_af(self) -> int:
        """
        POP AF - Pop AF from stack.
        
        Returns:
            Extra cycles used
        """
        self.f = self.read_byte(self.sp)
        self.sp = (self.sp + 1) & 0xFFFF
        self.a = self.read_byte(self.sp)
        self.sp = (self.sp + 1) & 0xFFFF
        return 0
    
    def _add_a_r(self, r: int) -> int:
        """
        ADD A, r - Add register to A.
        
        Args:
            r: Register code
            
        Returns:
            Extra cycles used
        """
        value = self._get_reg(r)
        result = self.a + value
        
        # Update flags
        self.f = 0
        
        # Carry flag
        if result > 0xFF:
            self.f |= self.FLAG_C
            
        # Half carry flag (carry from bit 3 to bit 4)
        if ((self.a & 0x0F) + (value & 0x0F)) > 0x0F:
            self.f |= self.FLAG_H
            
        # Set result
        self.a = result & 0xFF
        
        # Update sign and zero flags
        self._update_flag_sign_zero(self.a)
        
        # Update parity flag (used as overflow for add/sub)
        if ((self.a ^ value) & 0x80) == 0 and ((self.a ^ result) & 0x80) != 0:
            self.f |= self.FLAG_P
            
        return 0
    
    def _sub_r(self, r: int) -> int:
        """
        SUB r - Subtract register from A.
        
        Args:
            r: Register code
            
        Returns:
            Extra cycles used
        """
        value = self._get_reg(r)
        result = self.a - value
        
        # Update flags
        self.f = self.FLAG_N  # Set subtract flag
        
        # Carry flag
        if result < 0:
            self.f |= self.FLAG_C
            
        # Half carry flag (borrow from bit 4 to bit 3)
        if ((self.a & 0x0F) - (value & 0x0F)) < 0:
            self.f |= self.FLAG_H
            
        # Set result
        self.a = result & 0xFF
        
        # Update sign and zero flags
        self._update_flag_sign_zero(self.a)
        
        # Update parity flag (used as overflow for add/sub)
        if ((self.a ^ value) & 0x80) != 0 and ((self.a ^ result) & 0x80) != 0:
            self.f |= self.FLAG_P
            
        return 0
    
    def _and_r(self, r: int) -> int:
        """
        AND r - Logical AND register with A.
        
        Args:
            r: Register code
            
        Returns:
            Extra cycles used
        """
        value = self._get_reg(r)
        self.a &= value
        
        # Update flags
        self.f = self.FLAG_H  # H is always set
        
        # Update sign and zero flags
        self._update_flag_sign_zero(self.a)
        
        # Update parity flag
        self._update_flag_parity(self.a)
        
        return 0
    
    def _or_r(self, r: int) -> int:
        """
        OR r - Logical OR register with A.
        
        Args:
            r: Register code
            
        Returns:
            Extra cycles used
        """
        value = self._get_reg(r)
        self.a |= value
        
        # Update flags
        self.f = 0
        
        # Update sign and zero flags
        self._update_flag_sign_zero(self.a)
        
        # Update parity flag
        self._update_flag_parity(self.a)
        
        return 0
    
    def _nop(self) -> int:
        """
        NOP - No operation.
        
        Returns:
            Extra cycles used
        """
        return 0
    
    def _halt(self) -> int:
        """
        HALT - Halt CPU until interrupt.
        
        Returns:
            Extra cycles used
        """
        self.halted = True
        return 0
    
    def _jp_nn(self) -> int:
        """
        JP nn - Jump to address nn.
        
        Returns:
            Extra cycles used
        """
        self.pc = self.read_word(self.pc - 2)
        return 0
    
    def _jp_nz_nn(self) -> int:
        """
        JP NZ, nn - Jump to address if Zero flag is reset.
        
        Returns:
            Extra cycles used
        """
        if not (self.f & self.FLAG_Z):
            self.pc = self.read_word(self.pc - 2)
        return 0
    
    def _jp_z_nn(self) -> int:
        """
        JP Z, nn - Jump to address if Zero flag is set.
        
        Returns:
            Extra cycles used
        """
        if self.f & self.FLAG_Z:
            self.pc = self.read_word(self.pc - 2)
        return 0
    
    def _jp_nc_nn(self) -> int:
        """
        JP NC, nn - Jump to address if Carry flag is reset.
        
        Returns:
            Extra cycles used
        """
        if not (self.f & self.FLAG_C):
            self.pc = self.read_word(self.pc - 2)
        return 0
    
    def _jp_c_nn(self) -> int:
        """
        JP C, nn - Jump to address if Carry flag is set.
        
        Returns:
            Extra cycles used
        """
        if self.f & self.FLAG_C:
            self.pc = self.read_word(self.pc - 2)
        return 0
    
    def _call_nn(self) -> int:
        """
        CALL nn - Call subroutine at address nn.
        
        Returns:
            Extra cycles used
        """
        address = self.read_word(self.pc - 2)
        
        # Push return address onto stack
        self.sp = (self.sp - 1) & 0xFFFF
        self.write_byte(self.sp, (self.pc >> 8) & 0xFF)
        self.sp = (self.sp - 1) & 0xFFFF
        self.write_byte(self.sp, self.pc & 0xFF)
        
        # Jump to subroutine
        self.pc = address
        return 0
    
    def _ret(self) -> int:
        """
        RET - Return from subroutine.
        
        Returns:
            Extra cycles used
        """
        # Pop return address from stack
        low = self.read_byte(self.sp)
        self.sp = (self.sp + 1) & 0xFFFF
        high = self.read_byte(self.sp)
        self.sp = (self.sp + 1) & 0xFFFF
        
        # Set PC to return address
        self.pc = (high << 8) | low
        return 0