# common/interfaces.py
from abc import ABC, abstractmethod
import typing as t

class CPU(ABC):
    @abstractmethod
    def reset(self) -> None:
        """Reset the CPU to initial state."""
        pass
        
    @abstractmethod
    def step(self) -> int:
        """Execute one instruction and return cycles used."""
        pass
        
    @abstractmethod
    def get_state(self) -> dict:
        """Return the current CPU state as a dictionary."""
        pass
        
    @abstractmethod
    def set_memory(self, memory: 'Memory') -> None:
        """Connect the CPU to a memory system."""
        pass

class Memory(ABC):
    @abstractmethod
    def read(self, address: int) -> int:
        """Read a byte from the specified address."""
        pass
        
    @abstractmethod
    def write(self, address: int, value: int) -> None:
        """Write a byte to the specified address."""
        pass
        
    @abstractmethod
    def load_rom(self, rom_data: bytes) -> None:
        """Load ROM data into memory."""
        pass

class VideoProcessor(ABC):
    @abstractmethod
    def step(self, cycles: int) -> bool:
        """Run for specified number of cycles. Return True if frame completed."""
        pass
        
    @abstractmethod
    def get_frame_buffer(self) -> bytes:
        """Get the current frame buffer."""
        pass
        
    @abstractmethod
    def get_state(self) -> dict:
        """Return the current PPU state as a dictionary."""
        pass

class AudioProcessor(ABC):
    @abstractmethod
    def step(self, cycles: int) -> t.List[float]:
        """Generate audio samples for specified cycles."""
        pass
        
    @abstractmethod
    def get_state(self) -> dict:
        """Return the current audio processor state as a dictionary."""
        pass

class System(ABC):
    @abstractmethod
    def __init__(self, config: dict):
        """Initialize the system with configuration."""
        pass
        
    @abstractmethod
    def load_rom(self, rom_path: str) -> None:
        """Load a ROM file."""
        pass
        
    @abstractmethod
    def reset(self) -> None:
        """Reset the system."""
        pass
        
    @abstractmethod
    def run_frame(self) -> dict:
        """Run one frame and return state data."""
        pass
        
    @abstractmethod
    def get_system_state(self) -> dict:
        """Get complete system state."""
        pass