"""
Global constants for the Quantum Signal Emulator.
"""

# Analysis constants
ANALYSIS_MODES = ['quantum', 'classical', 'hybrid']

# Quantum analysis constants
DEFAULT_NUM_QUBITS = 8
DEFAULT_QUANTUM_SHOTS = 8192

# Visualization constants
COLOR_MAPS = {
    'spectrum': 'viridis',
    'phase': 'plasma',
    'register': 'inferno',
    'timing': 'coolwarm',
    'heatmap': 'magma',
    'state_space': 'jet'
}

# Performance constants
MAX_HISTORY_SIZE = 100000
HISTORY_TRIM_SIZE = 50000

# System frequency multipliers (relative to CPU)
FREQ_MULTIPLIERS = {
    'nes': {
        'ppu': 3.0,  # PPU runs at 3x CPU speed
        'apu': 1.0   # APU runs at same speed as CPU
    },
    'snes': {
        'ppu': 1.5,  # Approximation for demo purposes
        'dsp': 1.0
    },
    'genesis': {
        'vdp': 1.75, # Approximation based on 7.67MHz CPU vs 13.4MHz VDP
        'z80': 0.47  # Z80 runs at 3.58MHz vs 7.67MHz main CPU
    },
    'atari2600': {
        'tia': 3.0,  # TIA color clock is 3x CPU clock
        'riot': 1.0
    }
}

# File extensions by system
ROM_EXTENSIONS = {
    'nes': ['.nes'],
    'snes': ['.sfc', '.smc'],
    'genesis': ['.md', '.bin', '.gen'],
    'atari2600': ['.a26', '.bin']
}