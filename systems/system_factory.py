# systems/system_factory.py
from ..common.interfaces import System
from ..system_configs import SYSTEM_CONFIGS

# Import system implementations
from .nes.nes_system import NESSystem
from .snes.snes_system import SNESSystem
#from .genesis.genesis_system import GenesisSystem
from .atari2600.atari_system import AtariSystem

class SystemFactory:
    @staticmethod
    def create_system(system_type: str) -> System:
        """Create and return a system implementation based on type."""
        if system_type not in SYSTEM_CONFIGS:
            raise ValueError(f"Unknown system type: {system_type}")
            
        config = SYSTEM_CONFIGS[system_type]
        
        if system_type == "nes":
            return NESSystem(config)
        elif system_type == "snes":
            return SNESSystem(config)
        elif system_type == "genesis":
            return GenesisSystem(config)
        elif system_type == "atari2600":
            return AtariSystem(config)
        else:
            raise ValueError(f"System implementation not available for: {system_type}")