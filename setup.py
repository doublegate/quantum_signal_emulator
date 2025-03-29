from setuptools import setup, find_packages

setup(
    name="quantum_signal_emulator",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "torch>=1.10.0",
        "qiskit>=0.45.0",
        "qiskit-aer>=0.9.0",
        "matplotlib>=3.4.0",
        "scikit-learn>=1.0.0",
        "tqdm>=4.62.0",
        "pandas>=1.3.0",
        "pywavelets>=1.1.0"
    ],
    extras_require={
        "gpu": ["cupy>=10.0.0"],
    },
    entry_points={
        "console_scripts": [
            "quantum-signal-emulator=quantum_signal_emulator.main:main",
        ],
    },
    author="DoubleGate",
    author_email="parobek@gmail.com",
    description="A quantum-inspired signal analyzer for cycle-accurate hardware emulation",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/doublegate/quantum_signal_emulator",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering",
    ],
    python_requires=">=3.9",
)
