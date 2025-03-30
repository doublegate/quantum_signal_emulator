# Installation and Usage Guide

This guide provides detailed instructions for installing and using the Quantum Signal Emulator.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
   - [Basic Installation](#basic-installation)
   - [Development Installation](#development-installation)
   - [GPU Acceleration](#gpu-acceleration)
3. [Usage](#usage)
   - [Command-Line Interface](#command-line-interface)
   - [Python API](#python-api)
4. [Examples](#examples)
5. [Troubleshooting](#troubleshooting)

## Prerequisites

The Quantum Signal Emulator requires:

- Python 3.9 or higher
- CUDA-capable GPU (optional, for acceleration)

## Installation

### Basic Installation

To install the Quantum Signal Emulator with basic dependencies:

```bash
# Install from PyPI
pip install quantum-signal-emulator

# OR install from source
git clone https://github.com/doublegate/quantum_signal_emulator.git
cd quantum_signal_emulator
pip install .
```

### Development Installation

For development, install the package in editable mode with development dependencies:

```bash
git clone https://github.com/doublegate/quantum_signal_emulator.git
cd quantum_signal_emulator
pip install -e ".[dev]"
```

### GPU Acceleration

To enable GPU acceleration, install the CUDA dependencies:

```bash
# With pip
pip install "quantum-signal-emulator[gpu]"

# OR from source
pip install -e ".[gpu]"
```

## Usage

### Command-Line Interface

The Quantum Signal Emulator can be run from the command line:

```bash
# Run with default parameters (NES system, hybrid analysis)
quantum-signal-emulator --rom path/to/rom.nes

# Run with specific parameters
quantum-signal-emulator --system snes --rom path/to/rom.sfc --analysis-mode quantum --frames 5 --output json
```

Available command-line options:

| Option | Description |
|--------|-------------|
| `--system` | Target system (`nes`, `snes`, `genesis`, `atari2600`) |
| `--rom` | Path to ROM file |
| `--analysis-mode` | Analysis mode (`quantum`, `classical`, `hybrid`) |
| `--frames` | Number of frames to analyze |
| `--output` | Output format (`json`, `csv`, `binary`) |
| `--no-gpu` | Disable GPU acceleration |
| `--no-3d` | Disable 3D visualizations |
| `--no-visualization` | Disable all visualizations |
| `--save-state` | Path to save final state |
| `--load-state` | Path to load initial state |
| `--config` | Path to custom configuration file |
| `--debug` | Enable debug mode |
| `--log-level` | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |

### Python API

You can also use the Quantum Signal Emulator as a Python library:

```python
from quantum_signal_emulator.systems.system_factory import SystemFactory
from quantum_signal_emulator.analysis.quantum_analyzer import QuantumAnalyzer
from quantum_signal_emulator.common.visualizer import SignalVisualizer

# Create a system
system = SystemFactory.create_system("nes")

# Load ROM and run simulation
system.load_rom("path/to/rom.nes")
system.reset()
system.run_frame()

# Analyze results
analyzer = QuantumAnalyzer()
results = analyzer.analyze_hardware_state(system.state_history)

# Visualize results
visualizer = SignalVisualizer()
visualizer.plot_quantum_results(results)
visualizer.visualize_cycle_timing(system)
```

## Examples

Example scripts are provided in the `examples` directory:

- `basic_analysis.py`: Basic cycle analysis without quantum features
- `quantum_analysis.py`: Quantum-inspired analysis with visualization

To run an example:

```bash
cd examples
python basic_analysis.py --rom path/to/rom.nes
```

## Troubleshooting

### Missing Quantum Libraries

If you see warnings about missing Qiskit libraries:

```
WARNING: Qiskit not available. Quantum processing will use simulation mode.
```

Install the quantum computing dependencies:

```bash
pip install qiskit qiskit-aer
```

### CUDA/GPU Issues

If GPU acceleration isn't working:

1. Ensure you have installed CUDA and the `cupy` package
2. Check that your GPU is CUDA-capable and has the required drivers
3. Try running with the `--no-gpu` flag to use CPU-only mode

### Visualization Problems

If visualizations aren't displaying properly:

1. Ensure you have installed matplotlib correctly
2. For 3D visualizations, make sure your system supports 3D graphics
3. Try running with the `--no-3d` flag to use 2D visualizations only