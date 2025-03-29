"""
Advanced visualization tools for analyzing hardware signals and emulation results.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
from .interfaces import System

logger = logging.getLogger("QuantumSignalEmulator.Visualizer")

class SignalVisualizer:
    """
    Advanced visualization tools for analyzing hardware signals and emulation results.
    """
    
    def __init__(self, use_3d: bool = True, dark_mode: bool = True):
        """
        Initialize the signal visualizer.
        
        Args:
            use_3d: Whether to enable 3D visualizations
            dark_mode: Whether to use dark background for plots
        """
        self.use_3d = use_3d
        self.dark_mode = dark_mode
        self.color_map = cm.viridis
        
        # Set plot style
        if self.dark_mode:
            plt.style.use('dark_background')
        else:
            plt.style.use('default')
            
        # Check for 3D capability
        self.has_3d = True
        try:
            from mpl_toolkits.mplot3d import Axes3D
        except ImportError:
            logger.warning("3D plotting not available. Falling back to 2D plots.")
            self.has_3d = False
            self.use_3d = False
        
        logger.info("Initialized signal visualizer")
    
    def plot_quantum_results(self, results: Dict[str, Any], figsize: Tuple[int, int] = (12, 10)) -> None:
        """
        Plot results from quantum signal analysis.
        
        Args:
            results: Results from QuantumSignalProcessor.analyze_signal()
            figsize: Figure size (width, height) in inches
        """
        if not results:
            logger.warning("No quantum results to plot")
            return
            
        fig = plt.figure(figsize=figsize)
        
        # Plot frequency spectrum
        ax1 = fig.add_subplot(221)
        freq_bins = results.get("frequency_bins", [])
        amplitudes = results.get("amplitudes", [])
        
        if freq_bins and amplitudes:
            # Sort by frequency bin
            sorted_indices = np.argsort(freq_bins)
            sorted_freqs = [freq_bins[i] for i in sorted_indices]
            sorted_amps = [amplitudes[i] for i in sorted_indices]
            
            ax1.bar(sorted_freqs, sorted_amps, color='cyan', alpha=0.7)
            ax1.set_title("Frequency Spectrum")
            ax1.set_xlabel("Frequency Bin")
            ax1.set_ylabel("Amplitude")
            ax1.grid(True, alpha=0.3)
        else:
            ax1.text(0.5, 0.5, "No frequency data available", ha='center', va='center')
            ax1.set_title("Frequency Spectrum")
            
        # Plot phase information
        ax2 = fig.add_subplot(222, projection='polar')
        phases = results.get("phases", np.array([]))
        
        if phases.size > 0:
            # Extract phase angles and weights
            angles = phases[:, 0]
            weights = phases[:, 1]
            
            ax2.scatter(angles, weights, c=weights, cmap='plasma', alpha=0.7, s=100)
            ax2.set_title("Phase Distribution")
            ax2.grid(True, alpha=0.3)
        else:
            # Can't use text in polar projection easily, so just leave it empty
            ax2.set_title("Phase Distribution (No Data)")
        
        # Plot interference pattern
        ax3 = fig.add_subplot(223)
        interference = results.get("interference_pattern", np.array([]))
        
        if interference.size > 0:
            x = np.arange(len(interference))
            ax3.plot(x, np.abs(interference), 'g-', linewidth=2, alpha=0.7)
            ax3.set_title("Interference Pattern")
            ax3.set_xlabel("State")
            ax3.set_ylabel("Magnitude")
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, "No interference data available", ha='center', va='center')
            ax3.set_title("Interference Pattern")
        
        # Plot entropy and statistics
        ax4 = fig.add_subplot(224)
        entropy = results.get("quantum_entropy", 0.0)
        
        # Find peak frequency
        if freq_bins and amplitudes:
            peak_freq = sorted_freqs[np.argmax(sorted_amps)]
            peak_amp = max(sorted_amps)
        else:
            peak_freq = "N/A"
            peak_amp = "N/A"
            
        # Find dominant phase
        if phases.size > 0:
            dom_phase = angles[np.argmax(weights)]
        else:
            dom_phase = "N/A"
            
        # Find interference peak
        if interference.size > 0:
            int_peak = np.max(np.abs(interference))
        else:
            int_peak = "N/A"
        
        # Create text summary
        analysis_method = results.get("analysis_method", "quantum")
        text = f"{analysis_method.capitalize()} Analysis Summary:\n\n" \
               f"Entropy: {entropy:.4f} bits\n" \
               f"Peak Frequency Bin: {peak_freq}\n" \
               f"Peak Amplitude: {peak_amp}\n" \
               f"Dominant Phase: {dom_phase}\n" \
               f"Interference Peak: {int_peak}"
               
        ax4.text(0.5, 0.5, text, ha='center', va='center', fontsize=12)
        ax4.set_title("Analysis Summary")
        ax4.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def visualize_cycle_timing(self, system: System, figsize: Tuple[int, int] = (14, 10)) -> None:
        """
        Visualize cycle timing information from system.
        
        Args:
            system: System with state history
            figsize: Figure size (width, height) in inches
        """
        # Extract state history
        state_history = getattr(system, 'state_history', [])
        
        if not state_history:
            logger.warning("No system state history available for visualization")
            return
            
        fig = plt.figure(figsize=figsize)
        
        # Extract data
        cycles = [state.get("cycle", i) for i, state in enumerate(state_history)]
        scanlines = [state.get("scanline", 0) for state in state_history]
        dots = [state.get("dot", 0) for state in state_history]
        jitters = [state.get("timing_jitter_ns", 0) for state in state_history]
        
        # Get register data if available
        registers = {}
        register_names = []
        
        # Find all register names in the first state that has registers
        for state in state_history:
            if "registers" in state and state["registers"]:
                register_names = list(state["registers"].keys())
                break
                
        # Limit to first 8 registers to avoid overcrowding
        if len(register_names) > 8:
            register_names = register_names[:8]
            
        # Extract register values
        for reg in register_names:
            registers[reg] = [state.get("registers", {}).get(reg, 0) for state in state_history]
        
        # Plot scanline vs. cycle
        ax1 = fig.add_subplot(221)
        ax1.plot(cycles, scanlines, 'b-', alpha=0.7)
        ax1.set_title("Scanline vs. Cycle")
        ax1.set_xlabel("Cycle")
        ax1.set_ylabel("Scanline")
        ax1.grid(True, alpha=0.3)
        
        # Plot jitter histogram
        ax2 = fig.add_subplot(222)
        if any(jitters):
            ax2.hist(jitters, bins=50, color='green', alpha=0.7)
            ax2.set_title("Timing Jitter Distribution")
            ax2.set_xlabel("Jitter (ns)")
            ax2.set_ylabel("Frequency")
        else:
            ax2.text(0.5, 0.5, "No jitter data available", ha='center', va='center')
            ax2.set_title("Timing Jitter Distribution")
        ax2.grid(True, alpha=0.3)
        
        # Plot register values heatmap
        ax3 = fig.add_subplot(223)
        if registers and register_names:
            register_matrix = np.array([registers[reg][:min(1000, len(cycles))] for reg in register_names])
            im = ax3.imshow(register_matrix, aspect='auto', cmap='inferno')
            ax3.set_title("Register Values Over Time")
            ax3.set_xlabel("Cycle (first 1000)")
            ax3.set_ylabel("Register")
            ax3.set_yticks(np.arange(len(register_names)))
            ax3.set_yticklabels(register_names)
            plt.colorbar(im, ax=ax3, label="Value")
        else:
            ax3.text(0.5, 0.5, "No register data available", ha='center', va='center')
            ax3.set_title("Register Values Over Time")
        
        # 3D visualization if enabled
        if self.use_3d and self.has_3d:
            ax4 = fig.add_subplot(224, projection='3d')
            
            # Sample data for 3D plot (use every 10th point to avoid overcrowding)
            sample_step = 10
            max_points = 500
            
            if len(scanlines) > sample_step and len(dots) > sample_step:
                x = scanlines[::sample_step][:max_points]
                y = dots[::sample_step][:max_points]
                
                if any(jitters):
                    z = jitters[::sample_step][:max_points]
                else:
                    # Use cycle number for z-axis if no jitter data
                    z = cycles[::sample_step][:max_points]
                
                # Create colormap based on cycle
                colors = cycles[::sample_step][:max_points]
                
                scatter = ax4.scatter(x, y, z, c=colors, cmap=self.color_map, 
                                     s=30, alpha=0.7)
                ax4.set_title("3D Timing Visualization")
                ax4.set_xlabel("Scanline")
                ax4.set_ylabel("Dot")
                ax4.set_zlabel("Value")
                plt.colorbar(scatter, ax=ax4, label="Cycle")
            else:
                # Not enough data points
                ax4.text(0.5, 0.5, 0.5, "Insufficient data for 3D plot", 
                        ha='center', va='center', zdir='y')
                ax4.set_title("3D Timing Visualization")
        else:
            ax4 = fig.add_subplot(224)
            # 2D alternative - dot vs. scanline colored by jitter or cycle
            if len(scanlines) > 0 and len(dots) > 0:
                if any(jitters):
                    colors = jitters[:1000]
                    label = "Jitter (ns)"
                else:
                    colors = cycles[:1000]
                    label = "Cycle"
                    
                scatter = ax4.scatter(scanlines[:1000], dots[:1000], c=colors, 
                                     cmap='coolwarm', alpha=0.7, s=10)
                ax4.set_title("Dot vs. Scanline")
                ax4.set_xlabel("Scanline")
                ax4.set_ylabel("Dot")
                plt.colorbar(scatter, ax=ax4, label=label)
            else:
                ax4.text(0.5, 0.5, "Insufficient data for plot", ha='center', va='center')
                ax4.set_title("Dot vs. Scanline")
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_register_states(self, system: System, 
                            register_names: Optional[List[str]] = None,
                            figsize: Tuple[int, int] = (12, 8)) -> None:
        """
        Plot register state changes over time.
        
        Args:
            system: System with state history
            register_names: List of register names to plot (plots all if None)
            figsize: Figure size (width, height) in inches
        """
        # Extract state history
        state_history = getattr(system, 'state_history', [])
        
        if not state_history:
            logger.warning("No system state history available for visualization")
            return
            
        # Find all register names if not specified
        if register_names is None:
            for state in state_history:
                if "registers" in state and state["registers"]:
                    register_names = list(state["registers"].keys())
                    break
        
        if not register_names:
            logger.warning("No registers found in state history")
            return
            
        # Limit number of registers to display to avoid overcrowding
        if len(register_names) > 8:
            logger.info(f"Limiting visualization to first 8 of {len(register_names)} registers")
            register_names = register_names[:8]
            
        # Extract data
        cycles = [state.get("cycle", i) for i, state in enumerate(state_history)]
        register_data = {}
        for reg in register_names:
            register_data[reg] = [state.get("registers", {}).get(reg, 0) for state in state_history]
            
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot each register
        for i, reg in enumerate(register_names):
            color = plt.cm.tab10(i % 10)
            ax.plot(cycles, register_data[reg], label=reg, color=color, alpha=0.7)
            
        ax.set_title("Register States Over Time")
        ax.set_xlabel("Cycle")
        ax.set_ylabel("Value")
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
        
        plt.tight_layout()
        plt.show()
    
    def visualize_state_space(self, system: System, 
                             dimensionality_reduction: str = 'pca',
                             figsize: Tuple[int, int] = (10, 8)) -> None:
        """
        Visualize the state space of the emulation using dimensionality reduction.
        
        Args:
            system: System with state history
            dimensionality_reduction: Method to use ('pca' or 'tsne')
            figsize: Figure size (width, height) in inches
        """
        # Extract state history
        state_history = getattr(system, 'state_history', [])
        
        if not state_history:
            logger.warning("No system state history available for visualization")
            return
            
        # Extract register state vectors
        register_states = []
        cycles = []
        scanlines = []
        
        for state in state_history:
            if "registers" in state and state["registers"]:
                # Create a vector of all register values
                vector = list(state["registers"].values())
                register_states.append(vector)
                cycles.append(state.get("cycle", 0))
                scanlines.append(state.get("scanline", 0))
                
        if not register_states:
            logger.warning("No register state data available for visualization")
            return
            
        # Ensure all vectors have the same length
        min_length = min(len(vector) for vector in register_states)
        register_states = [vector[:min_length] for vector in register_states]
            
        # Convert to numpy array
        register_states = np.array(register_states)
        
        # Check for scikit-learn
        try:
            if dimensionality_reduction == 'pca':
                from sklearn.decomposition import PCA
                if len(register_states[0]) < 3:
                    # Add zeros to make at least 3 dimensions
                    padding = np.zeros((register_states.shape[0], 3 - register_states.shape[1]))
                    register_states = np.hstack((register_states, padding))
                
                reducer = PCA(n_components=3)
                reduced_data = reducer.fit_transform(register_states)
                reduction_name = "PCA"
            else:  # t-SNE
                from sklearn.manifold import TSNE
                if len(register_states[0]) < 2:
                    # Add zeros to make at least 2 dimensions
                    padding = np.zeros((register_states.shape[0], 2 - register_states.shape[1]))
                    register_states = np.hstack((register_states, padding))
                
                reducer = TSNE(n_components=3 if self.use_3d and self.has_3d else 2, 
                              perplexity=min(30, len(register_states)//5) if len(register_states) > 50 else 5,
                              n_iter=1000)
                reduced_data = reducer.fit_transform(register_states)
                if reduced_data.shape[1] < 3 and self.use_3d and self.has_3d:
                    # Add zeros for 3rd dimension if needed
                    padding = np.zeros((reduced_data.shape[0], 1))
                    reduced_data = np.hstack((reduced_data, padding))
                reduction_name = "t-SNE"
                
        except ImportError:
            logger.error("scikit-learn not installed. Cannot perform dimensionality reduction.")
            plt.figure(figsize=figsize)
            plt.text(0.5, 0.5, "scikit-learn required for state space visualization", 
                    ha='center', va='center')
            plt.title("State Space Visualization Error")
            plt.tight_layout()
            plt.show()
            return
        except Exception as e:
            logger.error(f"Error in dimensionality reduction: {e}")
            plt.figure(figsize=figsize)
            plt.text(0.5, 0.5, f"Error in dimensionality reduction: {str(e)}", 
                    ha='center', va='center')
            plt.title("State Space Visualization Error")
            plt.tight_layout()
            plt.show()
            return
            
        # Create visualization
        fig = plt.figure(figsize=figsize)
        
        if self.use_3d and self.has_3d and reduced_data.shape[1] >= 3:
            ax = fig.add_subplot(111, projection='3d')
            
            # Color points by cycle or scanline
            colors = scanlines if scanlines else cycles
            
            scatter = ax.scatter(
                reduced_data[:, 0], reduced_data[:, 1], reduced_data[:, 2],
                c=colors, cmap=self.color_map, alpha=0.7, s=30
            )
            
            ax.set_title(f"State Space Visualization ({reduction_name})")
            ax.set_xlabel(f"Component 1")
            ax.set_ylabel(f"Component 2")
            ax.set_zlabel(f"Component 3")
            
            plt.colorbar(scatter, ax=ax, label="Scanline" if scanlines else "Cycle")
        else:
            # 2D visualization
            ax = fig.add_subplot(111)
            
            # Color points by cycle or scanline
            colors = scanlines if scanlines else cycles
            
            scatter = ax.scatter(
                reduced_data[:, 0], reduced_data[:, 1],
                c=colors, cmap=self.color_map, alpha=0.7, s=30
            )
            
            ax.set_title(f"State Space Visualization ({reduction_name})")
            ax.set_xlabel(f"Component 1")
            ax.set_ylabel(f"Component 2")
            ax.grid(True, alpha=0.3)
            
            plt.colorbar(scatter, ax=ax, label="Scanline" if scanlines else "Cycle")
            
        plt.tight_layout()
        plt.show()