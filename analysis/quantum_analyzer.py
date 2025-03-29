"""
Quantum analysis module for extracting quantum-inspired insights from hardware signals.
"""

import numpy as np
import logging
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any, Union
import scipy.signal as signal

# Use Qiskit if available
try:
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False
    logging.warning("Qiskit not available. Quantum processing will use simulation mode.")

logger = logging.getLogger("QuantumSignalEmulator.QuantumAnalyzer")

class QuantumAnalyzer:
    """
    Quantum analyzer for extracting insights from hardware signals using
    quantum-inspired algorithms and processing techniques.
    """
    
    def __init__(self, num_qubits: int = 8, 
                quantum_mode: str = 'hybrid',
                shots: int = 8192):
        """
        Initialize the quantum analyzer.
        
        Args:
            num_qubits: Number of qubits to use for quantum processing
            quantum_mode: Processing mode ('quantum', 'classical', or 'hybrid')
            shots: Number of shots for quantum circuit execution
        """
        self.num_qubits = num_qubits
        self.quantum_mode = quantum_mode
        self.shots = shots
        self.results_cache = {}
        
        # Initialize quantum backend if available
        self.quantum_available = QISKIT_AVAILABLE
        if self.quantum_available:
            try:
                self.backend = AerSimulator(method='statevector')
                logger.info(f"Initialized quantum analyzer with {num_qubits} qubits")
            except Exception as e:
                logger.error(f"Error initializing quantum backend: {e}")
                self.quantum_available = False
        
        if not self.quantum_available and quantum_mode != 'classical':
            logger.warning(f"Qiskit not available. Falling back to classical mode despite requested '{quantum_mode}' mode")
            self.quantum_mode = 'classical'
    
    def analyze_hardware_state(self, state_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Perform quantum-inspired analysis of hardware state history.
        
        Args:
            state_history: List of system state snapshots
            
        Returns:
            Dictionary with analysis results
        """
        if not state_history:
            logger.warning("No state history to analyze")
            return {"error": "No state history available"}
            
        # Extract signal data from state history
        signal_data = self._extract_signal_from_states(state_history)
        
        # Perform analysis based on mode
        if self.quantum_mode == 'quantum' and self.quantum_available:
            return self._quantum_analysis(signal_data)
        elif self.quantum_mode == 'hybrid' and self.quantum_available:
            quantum_results = self._quantum_analysis(signal_data)
            classical_results = self._classical_analysis(signal_data)
            return self._combine_analysis_results(quantum_results, classical_results)
        else:
            return self._classical_analysis(signal_data)
    
    def _extract_signal_from_states(self, state_history: List[Dict[str, Any]]) -> np.ndarray:
        """
        Extract meaningful signal data from state history.
        
        Args:
            state_history: List of system state snapshots
            
        Returns:
            Numpy array of signal values
        """
        # Extract register transitions as signal
        signal_data = []
        
        # Create multi-dimensional signal array using all register values
        registers_seen = set()
        for state in state_history:
            if "registers" in state:
                registers_seen.update(state["registers"].keys())
        
        register_list = sorted(list(registers_seen))
        
        if not register_list:
            # Fallback to cycle information if no registers
            logger.info("No register data found, using cycle information as signal")
            signal_data = np.array([state.get("cycle", i) for i, state in enumerate(state_history)])
        else:
            # Create multi-dimensional signal from registers
            signal_matrix = []
            for state in state_history:
                if "registers" in state:
                    row = [state["registers"].get(reg, 0) for reg in register_list]
                    signal_matrix.append(row)
                else:
                    # Use zeros if no register data
                    signal_matrix.append([0] * len(register_list))
            
            # Convert to numpy array
            signal_data = np.array(signal_matrix)
            
            # Reduce dimensionality if needed
            if signal_data.shape[1] > 1:
                # Use first principal component as simplified 1D signal
                signal_data = self._extract_principal_component(signal_data)
        
        return signal_data
    
    def _extract_principal_component(self, signal_matrix: np.ndarray) -> np.ndarray:
        """
        Extract first principal component from multi-dimensional signal.
        
        Args:
            signal_matrix: Multi-dimensional signal array
            
        Returns:
            1D signal array (first principal component)
        """
        try:
            # Center the data
            centered_data = signal_matrix - np.mean(signal_matrix, axis=0)
            
            # Calculate covariance matrix
            cov_matrix = np.cov(centered_data, rowvar=False)
            
            # Calculate eigenvectors and eigenvalues
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
            
            # Sort eigenvectors by decreasing eigenvalues
            idx = np.argsort(eigenvalues)[::-1]
            eigenvectors = eigenvectors[:, idx]
            
            # Get the first principal component
            first_pc = eigenvectors[:, 0]
            
            # Project data onto first principal component
            pc1 = centered_data.dot(first_pc)
            
            return pc1
            
        except Exception as e:
            logger.error(f"Error extracting principal component: {e}")
            # Fallback to using the first register values
            return signal_matrix[:, 0]
    
    def _quantum_analysis(self, signal_data: np.ndarray) -> Dict[str, Any]:
        """
        Perform quantum analysis of signal data.
        
        Args:
            signal_data: Signal data array
            
        Returns:
            Dictionary with quantum analysis results
        """
        # Normalize input signal if needed
        if len(signal_data.shape) > 1:
            logger.warning("Multi-dimensional signal detected in quantum analysis. Using first column.")
            signal_data = signal_data[:, 0]
        
        # Ensure signal is normalized for quantum encoding
        signal_norm = (signal_data - np.min(signal_data)) / (np.max(signal_data) - np.min(signal_data) + 1e-10)
        
        # Create quantum circuit for encoding
        qc = QuantumCircuit(self.num_qubits)
        
        # Perform amplitude encoding (simplified version)
        data_size = min(2**self.num_qubits, len(signal_norm))
        for i in range(data_size):
            bin_idx = format(i, f'0{self.num_qubits}b')
            
            # Apply X gates where bit is 1
            for q_idx, bit in enumerate(bin_idx):
                if bit == '1':
                    qc.x(q_idx)
            
            # Apply controlled rotation based on amplitude
            angle = 2 * np.arcsin(np.sqrt(signal_norm[i % len(signal_norm)]))
            qc.ry(angle, self.num_qubits - 1)
            
            # Uncompute X gates
            for q_idx, bit in enumerate(bin_idx):
                if bit == '1':
                    qc.x(q_idx)
        
        # Apply quantum Fourier transform
        qc_fft = self._apply_qft(qc)
        
        # Add measurement
        measure_qc = qc_fft.copy()
        measure_qc.measure_all()
        
        # Execute circuit
        job = self.backend.run(measure_qc, shots=self.shots)
        result = job.result()
        counts = result.get_counts()
        
        # Process results
        freq_data = self._process_quantum_results(counts)
        
        # Calculate quantum fingerprint and entropy
        fingerprint = self._calculate_quantum_fingerprint(counts)
        entropy = self._calculate_quantum_entropy(counts)
        
        return {
            "method": "quantum",
            "frequency_data": freq_data,
            "quantum_fingerprint": fingerprint,
            "quantum_entropy": entropy,
            "coherence_measure": self._calculate_coherence_measure(freq_data),
            "analysis_summary": self._generate_quantum_summary(freq_data, fingerprint, entropy)
        }
    
    def _apply_qft(self, circuit: QuantumCircuit) -> QuantumCircuit:
        """
        Apply quantum Fourier transform to circuit.
        
        Args:
            circuit: Input quantum circuit
            
        Returns:
            Circuit with QFT applied
        """
        qc = circuit.copy()
        
        # Apply QFT to all qubits
        for i in range(self.num_qubits):
            qc.h(i)
            for j in range(i + 1, self.num_qubits):
                qc.cp(2 * np.pi / 2**(j-i), j, i)
        
        # Swap qubits (needed for correct QFT)
        for i in range(self.num_qubits // 2):
            qc.swap(i, self.num_qubits - i - 1)
            
        return qc
    
    def _process_quantum_results(self, counts: Dict[str, int]) -> Dict[str, Any]:
        """
        Process counts from quantum circuit execution.
        
        Args:
            counts: Measurement counts dictionary
            
        Returns:
            Dictionary with processed frequency data
        """
        # Convert bitstrings to frequency bins
        freq_bins = []
        amplitudes = []
        
        total_shots = sum(counts.values())
        for bitstring, count in counts.items():
            # Convert bitstring to frequency bin
            freq_idx = int(bitstring, 2)
            freq_bins.append(freq_idx)
            # Normalize amplitude by shot count
            amplitudes.append(count / total_shots)
        
        # Sort by frequency bin
        sorted_indices = np.argsort(freq_bins)
        sorted_freqs = [freq_bins[i] for i in sorted_indices]
        sorted_amps = [amplitudes[i] for i in sorted_indices]
        
        # Find dominant frequencies (peaks)
        peaks = []
        for i in range(1, len(sorted_freqs) - 1):
            if sorted_amps[i] > sorted_amps[i-1] and sorted_amps[i] > sorted_amps[i+1]:
                if sorted_amps[i] > 0.05:  # Threshold to filter noise
                    peaks.append({
                        "frequency": sorted_freqs[i],
                        "amplitude": sorted_amps[i]
                    })
        
        # Sort peaks by amplitude
        peaks.sort(key=lambda x: x["amplitude"], reverse=True)
        
        return {
            "frequency_bins": sorted_freqs,
            "amplitudes": sorted_amps,
            "dominant_frequencies": peaks[:5]  # Top 5 peaks
        }
    
    def _calculate_quantum_fingerprint(self, counts: Dict[str, int]) -> List[float]:
        """
        Calculate quantum fingerprint from measurement counts.
        
        Args:
            counts: Measurement counts dictionary
            
        Returns:
            List of fingerprint values
        """
        # Create full array of all possible states
        full_counts = np.zeros(2**self.num_qubits)
        for bitstring, count in counts.items():
            idx = int(bitstring, 2)
            full_counts[idx] = count
            
        # Normalize
        fingerprint = full_counts / np.sum(full_counts)
        
        # Apply wavelet transform for feature extraction
        coeffs = signal.cwt(fingerprint, signal.ricker, np.arange(1, 8))
        
        # Extract characteristic values
        features = []
        for i, coeff in enumerate(coeffs):
            features.append(np.mean(np.abs(coeff)))
            features.append(np.std(coeff))
        
        return features
    
    def _calculate_quantum_entropy(self, counts: Dict[str, int]) -> float:
        """
        Calculate quantum entropy from measurement probabilities.
        
        Args:
            counts: Measurement counts dictionary
            
        Returns:
            Calculated entropy value
        """
        total_shots = sum(counts.values())
        probabilities = [count / total_shots for count in counts.values()]
        entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)
        return entropy
    
    def _calculate_coherence_measure(self, freq_data: Dict[str, Any]) -> float:
        """
        Calculate a coherence measure based on frequency distribution.
        
        Args:
            freq_data: Frequency data dictionary
            
        Returns:
            Coherence measure value
        """
        # Calculate coherence as concentration of frequency components
        amplitudes = freq_data["amplitudes"]
        if not amplitudes:
            return 0.0
            
        # Gini coefficient as measure of concentration
        amplitudes_sorted = sorted(amplitudes)
        n = len(amplitudes_sorted)
        
        if n <= 1 or sum(amplitudes_sorted) == 0:
            return 0.0
            
        # Calculate area under Lorenz curve
        cumsum = np.cumsum(amplitudes_sorted)
        area = np.sum(cumsum) / (cumsum[-1] * n)
        
        # Gini coefficient
        return 1 - 2 * area
    
    def _generate_quantum_summary(self, freq_data: Dict[str, Any], 
                               fingerprint: List[float], 
                               entropy: float) -> str:
        """
        Generate a summary of quantum analysis results.
        
        Args:
            freq_data: Frequency data dictionary
            fingerprint: Quantum fingerprint values
            entropy: Quantum entropy value
            
        Returns:
            Summary string
        """
        # Summarize dominant frequencies
        dom_freqs = freq_data.get("dominant_frequencies", [])
        
        if not dom_freqs:
            freq_summary = "No dominant frequencies detected"
        else:
            top_freq = dom_freqs[0]
            freq_summary = f"Primary frequency component at bin {top_freq['frequency']} with amplitude {top_freq['amplitude']:.4f}"
            
            if len(dom_freqs) > 1:
                freq_summary += f", followed by bin {dom_freqs[1]['frequency']} ({dom_freqs[1]['amplitude']:.4f})"
        
        # Interpret entropy
        if entropy < 1.0:
            entropy_desc = "very low (highly ordered)"
        elif entropy < 2.0:
            entropy_desc = "low (moderately ordered)"
        elif entropy < 3.0:
            entropy_desc = "moderate (balanced)"
        elif entropy < 4.0:
            entropy_desc = "high (complex)"
        else:
            entropy_desc = "very high (chaotic)"
        
        # Create summary
        summary = f"{freq_summary}. Signal entropy is {entropy_desc} at {entropy:.2f} bits."
        
        return summary
    
    def _classical_analysis(self, signal_data: np.ndarray) -> Dict[str, Any]:
        """
        Perform classical analysis of signal data.
        
        Args:
            signal_data: Signal data array
            
        Returns:
            Dictionary with classical analysis results
        """
        # Ensure signal is 1D
        if len(signal_data.shape) > 1:
            logger.warning("Multi-dimensional signal detected in classical analysis. Using first column.")
            signal_data = signal_data[:, 0]
        
        # Perform FFT
        if len(signal_data) >= 2:
            fft_result = np.fft.fft(signal_data)
            fft_freq = np.fft.fftfreq(len(signal_data))
            
            # Extract amplitude and phase
            amplitude = np.abs(fft_result)
            phase = np.angle(fft_result)
            
            # Normalize amplitude
            normalized_amplitude = amplitude / np.max(amplitude) if np.max(amplitude) > 0 else amplitude
            
            # Find peaks in frequency domain
            peaks = []
            for i in range(1, len(fft_freq) - 1):
                if normalized_amplitude[i] > normalized_amplitude[i-1] and normalized_amplitude[i] > normalized_amplitude[i+1]:
                    if normalized_amplitude[i] > 0.05:  # Threshold to filter noise
                        peaks.append({
                            "frequency": fft_freq[i],
                            "amplitude": normalized_amplitude[i]
                        })
            
            # Sort peaks by amplitude
            peaks.sort(key=lambda x: x["amplitude"], reverse=True)
            
            # Calculate spectral entropy
            prob_dist = normalized_amplitude / np.sum(normalized_amplitude)
            entropy = -np.sum(p * np.log2(p) for p in prob_dist if p > 0)
            
            # Autocorrelation for pattern detection
            autocorr = np.correlate(signal_data, signal_data, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            autocorr = autocorr / autocorr[0]
            
            # Find significant repeating patterns
            patterns = []
            for i in range(1, min(100, len(autocorr) - 1)):
                if autocorr[i] > 0.5 and autocorr[i] > autocorr[i-1] and autocorr[i] > autocorr[i+1]:
                    patterns.append({
                        "period": i,
                        "correlation": autocorr[i]
                    })
            
            # Sort patterns by correlation strength
            patterns.sort(key=lambda x: x["correlation"], reverse=True)
            
            # Generate summary
            if peaks:
                top_freq = peaks[0]
                freq_summary = f"Primary frequency component at {top_freq['frequency']:.4f} with amplitude {top_freq['amplitude']:.4f}"
                
                if len(peaks) > 1:
                    freq_summary += f", followed by {peaks[1]['frequency']:.4f} ({peaks[1]['amplitude']:.4f})"
            else:
                freq_summary = "No significant frequency components detected"
                
            if patterns:
                pattern_desc = f"Detected repeating pattern with period {patterns[0]['period']} cycles"
            else:
                pattern_desc = "No clear repeating patterns detected"
                
            summary = f"{freq_summary}. {pattern_desc}. Signal entropy: {entropy:.2f} bits."
            
            return {
                "method": "classical",
                "frequency_data": {
                    "frequencies": fft_freq.tolist(),
                    "amplitudes": normalized_amplitude.tolist(),
                    "dominant_frequencies": peaks[:5]  # Top 5 peaks
                },
                "autocorrelation": autocorr.tolist(),
                "entropy": entropy,
                "repeating_patterns": patterns[:3],  # Top 3 patterns
                "analysis_summary": summary
            }
        else:
            # Not enough data for FFT
            return {
                "method": "classical",
                "error": "Insufficient data for spectral analysis",
                "analysis_summary": "Signal is too short for meaningful analysis"
            }
    
    def _combine_analysis_results(self, quantum_results: Dict[str, Any], 
                               classical_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Combine quantum and classical analysis results.
        
        Args:
            quantum_results: Results from quantum analysis
            classical_results: Results from classical analysis
            
        Returns:
            Combined analysis results
        """
        # Check for errors
        if "error" in quantum_results or "error" in classical_results:
            # Use whichever result doesn't have an error
            if "error" not in quantum_results:
                return quantum_results
            elif "error" not in classical_results:
                return classical_results
            else:
                return {
                    "method": "hybrid",
                    "error": "Both analysis methods failed",
                    "quantum_error": quantum_results.get("error", "Unknown error"),
                    "classical_error": classical_results.get("error", "Unknown error")
                }
        
        # Combine frequency data
        combined_freq_data = {
            "quantum_frequencies": quantum_results.get("frequency_data", {}),
            "classical_frequencies": classical_results.get("frequency_data", {})
        }
        
        # Create unified dominant frequencies list
        quantum_freqs = quantum_results.get("frequency_data", {}).get("dominant_frequencies", [])
        classical_freqs = classical_results.get("frequency_data", {}).get("dominant_frequencies", [])
        
        # Map quantum frequencies to classical frequency space (approximate)
        quantum_dom_mapped = []
        for qf in quantum_freqs:
            # Normalize quantum frequency bin to [0, 0.5] range (Nyquist limit)
            norm_freq = qf["frequency"] / (2**self.num_qubits) * 0.5
            quantum_dom_mapped.append({
                "frequency": norm_freq,
                "amplitude": qf["amplitude"],
                "source": "quantum"
            })
            
        # Format classical frequencies
        classical_dom_mapped = []
        for cf in classical_freqs:
            classical_dom_mapped.append({
                "frequency": abs(cf["frequency"]),  # Take absolute value
                "amplitude": cf["amplitude"],
                "source": "classical"
            })
            
        # Combine and sort by amplitude
        combined_dom_freqs = quantum_dom_mapped + classical_dom_mapped
        combined_dom_freqs.sort(key=lambda x: x["amplitude"], reverse=True)
        
        # Generate combined summary
        q_summary = quantum_results.get("analysis_summary", "")
        c_summary = classical_results.get("analysis_summary", "")
        
        # Merge into a combined summary
        combined_summary = "Hybrid analysis results: "
        
        if quantum_results.get("quantum_entropy", 0) > classical_results.get("entropy", 0):
            combined_summary += "Quantum analysis suggests higher complexity than classical analysis. "
        else:
            combined_summary += "Classical and quantum analyses show similar complexity patterns. "
            
        if combined_dom_freqs:
            top_freq = combined_dom_freqs[0]
            combined_summary += f"Dominant frequency component detected at {top_freq['frequency']:.4f} " \
                              f"(source: {top_freq['source']})."
        else:
            combined_summary += "No dominant frequency components detected."
        
        # Return combined results
        return {
            "method": "hybrid",
            "frequency_data": combined_freq_data,
            "dominant_frequencies": combined_dom_freqs[:5],  # Top 5
            "quantum_entropy": quantum_results.get("quantum_entropy", 0),
            "classical_entropy": classical_results.get("entropy", 0),
            "quantum_fingerprint": quantum_results.get("quantum_fingerprint", []),
            "repeating_patterns": classical_results.get("repeating_patterns", []),
            "coherence_measure": quantum_results.get("coherence_measure", 0),
            "analysis_summary": combined_summary
        }
    
    def visualize_results(self, results: Dict[str, Any], figsize: Tuple[int, int] = (12, 10)) -> None:
        """
        Visualize quantum analysis results.
        
        Args:
            results: Analysis results dictionary
            figsize: Figure size (width, height) in inches
        """
        if not results or "error" in results:
            logger.warning(f"Cannot visualize results: {results.get('error', 'Unknown error')}")
            return
            
        # Create figure
        fig = plt.figure(figsize=figsize)
        method = results.get("method", "unknown")
        
        if method == "quantum":
            # Quantum visualization
            self._visualize_quantum_results(fig, results)
        elif method == "classical":
            # Classical visualization
            self._visualize_classical_results(fig, results)
        elif method == "hybrid":
            # Hybrid visualization
            self._visualize_hybrid_results(fig, results)
        else:
            logger.warning(f"Unknown analysis method: {method}")
            plt.text(0.5, 0.5, f"Cannot visualize results for unknown method: {method}",
                    ha='center', va='center', transform=fig.transFigure)
        
        plt.tight_layout()
        plt.show()
    
    def _visualize_quantum_results(self, fig: plt.Figure, results: Dict[str, Any]) -> None:
        """
        Create visualization for quantum analysis results.
        
        Args:
            fig: Matplotlib figure
            results: Quantum analysis results
        """
        # Get frequency data
        freq_data = results.get("frequency_data", {})
        freq_bins = freq_data.get("frequency_bins", [])
        amplitudes = freq_data.get("amplitudes", [])
        
        # Plot frequency spectrum
        ax1 = fig.add_subplot(221)
        if freq_bins and amplitudes:
            ax1.bar(freq_bins, amplitudes, color='cyan', alpha=0.7)
            ax1.set_title("Quantum Frequency Spectrum")
            ax1.set_xlabel("Frequency Bin")
            ax1.set_ylabel("Probability Amplitude")
            ax1.grid(True, alpha=0.3)
        else:
            ax1.text(0.5, 0.5, "No frequency data available", ha='center', va='center')
            ax1.set_title("Quantum Frequency Spectrum")
        
        # Plot fingerprint features
        ax2 = fig.add_subplot(222)
        fingerprint = results.get("quantum_fingerprint", [])
        if fingerprint:
            ax2.bar(range(len(fingerprint)), fingerprint, color='green', alpha=0.7)
            ax2.set_title("Quantum Fingerprint")
            ax2.set_xlabel("Feature Index")
            ax2.set_ylabel("Feature Value")
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, "No fingerprint data available", ha='center', va='center')
            ax2.set_title("Quantum Fingerprint")
        
        # Plot dominant frequencies
        ax3 = fig.add_subplot(223)
        dom_freqs = freq_data.get("dominant_frequencies", [])
        if dom_freqs:
            frequencies = [f["frequency"] for f in dom_freqs]
            amplitudes = [f["amplitude"] for f in dom_freqs]
            ax3.bar(frequencies, amplitudes, color='purple', alpha=0.7)
            ax3.set_title("Dominant Frequencies")
            ax3.set_xlabel("Frequency Bin")
            ax3.set_ylabel("Amplitude")
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, "No dominant frequencies detected", ha='center', va='center')
            ax3.set_title("Dominant Frequencies")
        
        # Plot summary statistics
        ax4 = fig.add_subplot(224)
        entropy = results.get("quantum_entropy", 0)
        coherence = results.get("coherence_measure", 0)
        summary = results.get("analysis_summary", "No summary available")
        
        # Create text summary
        text = f"Quantum Analysis Summary:\n\n" \
               f"Entropy: {entropy:.4f} bits\n" \
               f"Coherence: {coherence:.4f}\n\n" \
               f"Summary: {summary}"
        
        ax4.text(0.5, 0.5, text, ha='center', va='center', fontsize=10, wrap=True)
        ax4.set_title("Analysis Summary")
        ax4.axis('off')
    
    def _visualize_classical_results(self, fig: plt.Figure, results: Dict[str, Any]) -> None:
        """
        Create visualization for classical analysis results.
        
        Args:
            fig: Matplotlib figure
            results: Classical analysis results
        """
        # Get frequency data
        freq_data = results.get("frequency_data", {})
        frequencies = freq_data.get("frequencies", [])
        amplitudes = freq_data.get("amplitudes", [])
        
        # Plot frequency spectrum
        ax1 = fig.add_subplot(221)
        if frequencies and amplitudes:
            # Only show positive frequencies up to Nyquist limit (0.5)
            mask = np.logical_and(np.array(frequencies) >= 0, np.array(frequencies) <= 0.5)
            pos_freqs = np.array(frequencies)[mask]
            pos_amps = np.array(amplitudes)[mask]
            
            ax1.plot(pos_freqs, pos_amps, 'b-', alpha=0.7)
            ax1.set_title("Frequency Spectrum")
            ax1.set_xlabel("Frequency")
            ax1.set_ylabel("Amplitude")
            ax1.grid(True, alpha=0.3)
        else:
            ax1.text(0.5, 0.5, "No frequency data available", ha='center', va='center')
            ax1.set_title("Frequency Spectrum")
        
        # Plot autocorrelation
        ax2 = fig.add_subplot(222)
        autocorr = results.get("autocorrelation", [])
        if autocorr:
            lags = np.arange(len(autocorr))
            ax2.plot(lags, autocorr, 'g-', alpha=0.7)
            ax2.set_title("Autocorrelation")
            ax2.set_xlabel("Lag")
            ax2.set_ylabel("Correlation")
            ax2.grid(True, alpha=0.3)
            # Add horizontal line at zero
            ax2.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        else:
            ax2.text(0.5, 0.5, "No autocorrelation data available", ha='center', va='center')
            ax2.set_title("Autocorrelation")
        
        # Plot repeating patterns
        ax3 = fig.add_subplot(223)
        patterns = results.get("repeating_patterns", [])
        if patterns:
            periods = [p["period"] for p in patterns]
            correlations = [p["correlation"] for p in patterns]
            ax3.bar(periods, correlations, color='orange', alpha=0.7)
            ax3.set_title("Repeating Patterns")
            ax3.set_xlabel("Period (cycles)")
            ax3.set_ylabel("Correlation")
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, "No repeating patterns detected", ha='center', va='center')
            ax3.set_title("Repeating Patterns")
        
        # Plot summary statistics
        ax4 = fig.add_subplot(224)
        entropy = results.get("entropy", 0)
        summary = results.get("analysis_summary", "No summary available")
        
        # Get dominant frequencies
        dom_freqs = freq_data.get("dominant_frequencies", [])
        dom_freq_text = ""
        if dom_freqs:
            dom_freq_text = "Dominant frequencies:\n"
            for i, df in enumerate(dom_freqs[:3]):  # Top 3
                dom_freq_text += f"{i+1}. f={df['frequency']:.4f}, amp={df['amplitude']:.4f}\n"
        
        # Create text summary
        text = f"Classical Analysis Summary:\n\n" \
               f"Entropy: {entropy:.4f} bits\n" \
               f"{dom_freq_text}\n" \
               f"Summary: {summary}"
        
        ax4.text(0.5, 0.5, text, ha='center', va='center', fontsize=10, wrap=True)
        ax4.set_title("Analysis Summary")
        ax4.axis('off')
    
    def _visualize_hybrid_results(self, fig: plt.Figure, results: Dict[str, Any]) -> None:
        """
        Create visualization for hybrid analysis results.
        
        Args:
            fig: Matplotlib figure
            results: Hybrid analysis results
        """
        # Get combined frequency data
        freq_data = results.get("frequency_data", {})
        
        # Plot quantum frequencies
        ax1 = fig.add_subplot(221)
        q_freq_data = freq_data.get("quantum_frequencies", {})
        q_bins = q_freq_data.get("frequency_bins", [])
        q_amps = q_freq_data.get("amplitudes", [])
        
        if q_bins and q_amps:
            ax1.bar(q_bins, q_amps, color='cyan', alpha=0.7)
            ax1.set_title("Quantum Frequency Spectrum")
            ax1.set_xlabel("Frequency Bin")
            ax1.set_ylabel("Amplitude")
            ax1.grid(True, alpha=0.3)
        else:
            ax1.text(0.5, 0.5, "No quantum frequency data available", ha='center', va='center')
            ax1.set_title("Quantum Frequency Spectrum")
        
        # Plot classical frequencies
        ax2 = fig.add_subplot(222)
        c_freq_data = freq_data.get("classical_frequencies", {})
        c_freqs = c_freq_data.get("frequencies", [])
        c_amps = c_freq_data.get("amplitudes", [])
        
        if c_freqs and c_amps:
            # Only show positive frequencies up to Nyquist limit (0.5)
            mask = np.logical_and(np.array(c_freqs) >= 0, np.array(c_freqs) <= 0.5)
            pos_freqs = np.array(c_freqs)[mask]
            pos_amps = np.array(c_amps)[mask]
            
            ax2.plot(pos_freqs, pos_amps, 'b-', alpha=0.7)
            ax2.set_title("Classical Frequency Spectrum")
            ax2.set_xlabel("Frequency")
            ax2.set_ylabel("Amplitude")
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, "No classical frequency data available", ha='center', va='center')
            ax2.set_title("Classical Frequency Spectrum")
        
        # Plot combined dominant frequencies
        ax3 = fig.add_subplot(223)
        dom_freqs = results.get("dominant_frequencies", [])
        
        if dom_freqs:
            # Group by source
            quantum_freqs = [df for df in dom_freqs if df.get("source") == "quantum"]
            classical_freqs = [df for df in dom_freqs if df.get("source") == "classical"]
            
            # Plot quantum frequencies
            if quantum_freqs:
                q_x = [df["frequency"] for df in quantum_freqs]
                q_y = [df["amplitude"] for df in quantum_freqs]
                ax3.bar(q_x, q_y, color='cyan', alpha=0.7, label='Quantum')
                
            # Plot classical frequencies
            if classical_freqs:
                c_x = [df["frequency"] for df in classical_freqs]
                c_y = [df["amplitude"] for df in classical_freqs]
                ax3.bar(c_x, c_y, color='blue', alpha=0.7, label='Classical')
            
            ax3.set_title("Dominant Frequencies (Combined)")
            ax3.set_xlabel("Frequency")
            ax3.set_ylabel("Amplitude")
            ax3.grid(True, alpha=0.3)
            ax3.legend()
        else:
            ax3.text(0.5, 0.5, "No dominant frequencies detected", ha='center', va='center')
            ax3.set_title("Dominant Frequencies (Combined)")
        
        # Plot summary statistics
        ax4 = fig.add_subplot(224)
        q_entropy = results.get("quantum_entropy", 0)
        c_entropy = results.get("classical_entropy", 0)
        coherence = results.get("coherence_measure", 0)
        summary = results.get("analysis_summary", "No summary available")
        
        # Create text summary
        text = f"Hybrid Analysis Summary:\n\n" \
               f"Quantum Entropy: {q_entropy:.4f} bits\n" \
               f"Classical Entropy: {c_entropy:.4f} bits\n" \
               f"Coherence: {coherence:.4f}\n\n" \
               f"Summary: {summary}"
        
        ax4.text(0.5, 0.5, text, ha='center', va='center', fontsize=10, wrap=True)
        ax4.set_title("Analysis Summary")
        ax4.axis('off')