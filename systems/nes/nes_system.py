# systems/nes/nes_system.py
from ...common.interfaces import System
from .cpu import CPU6502
from .ppu import NESPPU
from .apu import NESAPU
from .memory import NESMemory

class NESSystem(System):
    def __init__(self, config: dict):
        self.config = config
        
        # Initialize components
        self.memory = NESMemory(config)
        self.cpu = CPU6502()
        self.ppu = NESPPU(self.memory)
        self.apu = NESAPU()
        
        # Connect components
        self.cpu.set_memory(self.memory)
        
        # System state
        self.cycle_count = 0
        self.frame_count = 0
        self.state_history = []
    
    def load_rom(self, rom_path: str) -> None:
        with open(rom_path, 'rb') as f:
            rom_data = f.read()
            self.memory.load_rom(rom_data)
    
    def reset(self) -> None:
        self.cpu.reset()
        # Reset other components
        self.cycle_count = 0
        self.frame_count = 0
        self.state_history = []
    
    def run_frame(self) -> dict:
        """Run until a complete frame is rendered."""
        frame_completed = False
        
        while not frame_completed:
            # Execute one CPU instruction
            cpu_cycles = self.cpu.step()
            
            # Run PPU for 3x CPU cycles (NES PPU runs at 3x CPU clock)
            frame_completed = self.ppu.step(cpu_cycles * 3)
            
            # Run APU
            self.apu.step(cpu_cycles)
            
            # Update cycle count
            self.cycle_count += cpu_cycles
            
            # Record system state
            self._record_state()
        
        self.frame_count += 1
        
        return self.get_system_state()
    
    def get_system_state(self) -> dict:
        """Get the current state of the entire system."""
        return {
            "cycle_count": self.cycle_count,
            "frame_count": self.frame_count,
            "cpu_state": self.cpu.get_state(),
            "ppu_state": self.ppu.get_state(),
            "apu_state": self.apu.get_state() if hasattr(self.apu, 'get_state') else {},
            "frame_buffer": self.ppu.get_frame_buffer()
        }
    
    def _record_state(self) -> None:
        """Record the current state for analysis."""
        # Create a simplified state snapshot for analysis
        state = {
            "cycle": self.cycle_count,
            "scanline": self.ppu.scanline,
            "dot": self.ppu.cycle,
            "registers": {
                # CPU registers
                "A": self.cpu.A,
                "X": self.cpu.X,
                "Y": self.cpu.Y,
                "PC": self.cpu.PC,
                "SP": self.cpu.SP,
                "P": self.cpu.P,
                # PPU registers
                "PPUCTRL": self.ppu.control,
                "PPUMASK": self.ppu.mask,
                "PPUSTATUS": self.ppu.status,
                "OAMADDR": self.ppu.oam_addr,
            }
        }
        
        self.state_history.append(state)
        
        # Keep history to a reasonable size
        if len(self.state_history) > 100000:
            self.state_history = self.state_history[-50000:]