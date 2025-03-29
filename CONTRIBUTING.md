# Contributing to Quantum Signal Emulator

Thank you for considering contributing to the Quantum Signal Emulator project! 

## How to contribute

1. Fork the repository
2. Create a new branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests
5. Commit your changes (`git commit -m 'Add some amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## Coding Standards

- Follow PEP 8 style guide
- Write docstrings for all functions, classes, and modules
- Add type hints where appropriate
- Include unit tests for new functionality

## Adding New System Support

To add support for a new classic video game system:

1. Create a new directory under `systems/` with the system name
2. Implement all required components (CPU, video processor, memory, etc.)
3. Add the system configuration to `system_configs.py`
4. Update the SystemFactory class to create your new system

## Pull Request Process

1. Update the README.md with details of changes if appropriate
2. Update the documentation if needed
3. The PR should work for all supported platforms
4. A maintainer will review and merge your Pull Request
