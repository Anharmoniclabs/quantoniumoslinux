#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-SPX-Proprietary
# Copyright (C) 2025 Luis M. Minier / quantoniumos
# This file is listed in CLAIMS_PRACTICING_FILES.txt and is governed
# by LICENSE-SPX-PROPRIETARY.md. Commercial use requires a separate
# written patent license from Luis M. Minier.
"""
quantoniumos Middleware Transform Engine
Oscillating Wave Computation Layer - Binary→Wave→Compute

This middleware bridges classical binary (0/1) hardware with quantum-inspired
oscillating wave computation using RFT transforms. The system:

1. Takes binary input from hardware
2. Converts to oscillating waveforms via selected RFT variant
3. Performs computation in wave-space (frequency domain)
4. Converts back to binary for output

The middleware automatically selects the best RFT transform variant based on:
- Data type (text, image, audio, crypto keys, etc.)
- Computational requirements (speed, accuracy, security)
- Hardware capabilities
"""

import time
import numpy as np
from functools import lru_cache
from typing import Optional, Tuple, Dict, Any, List, Union
from dataclasses import dataclass
from algorithms.rft.variants.registry import VARIANTS, VariantInfo
from algorithms.rft.core.resonant_fourier_transform import rft_forward_square as rft_forward, rft_inverse_square as rft_inverse, rft_matrix_canonical as rft_matrix, PHI

try:
    from algorithms.rft.kernels.python_bindings.unitary_rft import (
        UnitaryRFT,
        RFT_FLAG_QUANTUM_SAFE,
    )
    HAS_ASSEMBLY = True
except ImportError:  # pragma: no cover - assembly optional
    UnitaryRFT = None
    RFT_FLAG_QUANTUM_SAFE = 0
    HAS_ASSEMBLY = False

# Import quantum gates for unified engine
try:
    from algorithms.rft.quantum_inspired.quantum_gates import (
        QuantumGate, PauliGates, RotationGates, PhaseGates,
        HadamardGates, ControlledGates
    )
    HAS_QUANTUM_GATES = True
except ImportError:
    HAS_QUANTUM_GATES = False
    QuantumGate = None

@dataclass
class TransformProfile:
    """Profile for selecting optimal transform"""
    data_type: str  # 'text', 'image', 'audio', 'crypto', 'generic'
    priority: str   # 'speed', 'accuracy', 'security', 'compression'
    size: int       # Data size in bytes
    
@dataclass
class WaveComputeResult:
    """Result from wave-space computation"""
    output_binary: bytes
    transform_used: str
    wave_spectrum: np.ndarray
    computation_time: float
    oscillation_frequency: float  # Average frequency in Hz


class MiddlewareTransformEngine:
    """
    Middleware layer that converts binary hardware I/O into oscillating
    waveforms for computation in wave-space using RFT transforms.
    
    This implements the core concept: Hardware (01) → Waves → Computation → (01)
    """
    
    def __init__(self):
        self.variants = VARIANTS
        self.experimental_variants: Dict[str, VariantInfo] = {
            "wave_fibonacci_pruned": VariantInfo(
                name="Wave-Fibonacci Pruned",
                generator=lambda n: np.empty((0, 0), dtype=np.complex128),
                innovation="RFTMW-discovered sparse golden-bin projection",
                use_case="Experimental O(N log N) approximate wave transform",
            ),
        }
        self.selected_variant = "original"  # Default
        self.phi = PHI
        
    def select_optimal_transform(self, profile: TransformProfile) -> str:
        """
        Intelligently select the best RFT transform variant based on
        the data profile and computational requirements.
        
        Returns the variant name (key in VARIANTS dict)
        """
        # Selection logic based on use case
        if profile.priority == 'security':
            if profile.data_type == 'crypto':
                return "fibonacci_tilt"  # Post-quantum crypto optimized
            else:
                return "chaotic_mix"  # Secure scrambling
                
        elif profile.priority == 'compression':
            if profile.data_type == 'text':
                return "log_periodic"  # ASCII bottleneck mitigation
            if profile.size < 1024:
                return "harmonic_phase"  # Good for small data
            else:
                return "adaptive_phi"  # Universal compression
                
        elif profile.priority == 'speed':
            if profile.data_type in {'texture', 'mixed'}:
                return "convex_mix"
            return "original"  # Fastest, cleanest transform
            
        elif profile.priority == 'accuracy':
            if profile.data_type == 'image' or profile.data_type == 'audio':
                return "geometric_lattice"  # Analog/optical optimized
            if profile.data_type == 'text':
                return "convex_mix"
            else:
                return "phi_chaotic_hybrid"  # Resilient codec
        
        # Default fallback
        return "original"
    
    def binary_to_waveform(self, binary_data: bytes) -> np.ndarray:
        """
        Convert binary data into oscillating waveform.

        Maps each byte to a single float64 sample in [−1, 1], matching the
        native C++/ASM engine behaviour (direct amplitude mapping, no bit
        expansion).  The signal is zero-padded to the next power of two
        before the forward RFT.
        """
        signal = np.frombuffer(binary_data, dtype=np.uint8).astype(np.float64)
        signal = signal / 127.5 - 1.0          # [0,255] → [−1,1]
        n = len(signal)
        next_pow2 = 2 ** int(np.ceil(np.log2(max(n, 1))))
        if next_pow2 > n:
            signal = np.pad(signal, (0, next_pow2 - n), mode='constant')
        if self.selected_variant == "original":
            waveform = rft_forward(signal)
        elif self.selected_variant == "wave_fibonacci_pruned":
            waveform = self._wave_fibonacci_pruned_forward(signal)
        else:
            variant_matrix = self.variants[self.selected_variant].generator(len(signal))
            waveform = variant_matrix @ signal
        return waveform

    def waveform_to_binary(self, waveform: np.ndarray, original_size: int) -> bytes:
        """
        Convert oscillating waveform back to binary data.

        Inverse of binary_to_waveform: recovers byte-level amplitudes from
        the wave-space representation.
        """
        if self.selected_variant == "original":
            signal = rft_inverse(waveform).real
        elif self.selected_variant == "wave_fibonacci_pruned":
            signal = self._wave_fibonacci_pruned_inverse(waveform).real
        else:
            variant_matrix = self.variants[self.selected_variant].generator(len(waveform))
            signal = np.conj(variant_matrix).T @ waveform
            signal = signal.real
        signal = signal[:original_size]
        signal = np.clip(np.round((signal + 1.0) * 127.5), 0, 255)
        return signal.astype(np.uint8).tobytes()
    
    def compute_in_wavespace(
        self, 
        binary_input: bytes,
        operation: str = "identity",
        profile: Optional[TransformProfile] = None
    ) -> WaveComputeResult:
        """
        Main middleware function: Perform computation in wave-space.
        """
        import time
        start_time = time.time()
        if profile is None:
            profile = TransformProfile(
                data_type='generic',
                priority='speed',
                size=len(binary_input)
            )
        self.selected_variant = self.select_optimal_transform(profile)
        waveform = self.binary_to_waveform(binary_input)
        if operation == "identity":
            result_waveform = waveform
        elif operation == "compress":
            threshold = 0.1 * np.max(np.abs(waveform))
            result_waveform = np.where(np.abs(waveform) > threshold, waveform, 0)
        elif operation == "encrypt":
            random_phases = np.exp(2j * np.pi * np.random.rand(len(waveform)))
            result_waveform = waveform * random_phases
        elif operation == "hash":
            phases = np.angle(waveform)
            hash_bits = ((phases + np.pi) / (2 * np.pi) * 255).astype(np.uint8)
            result_waveform = waveform
        else:
            result_waveform = waveform
        output_bits = len(binary_input) * 8
        output_binary = self.waveform_to_binary(result_waveform, output_bits)
        computation_time = time.time() - start_time
        freq_bins = np.arange(len(waveform))
        spectrum_power = np.abs(waveform) ** 2
        avg_frequency = np.sum(freq_bins * spectrum_power) / np.sum(spectrum_power)
        return WaveComputeResult(
            output_binary=output_binary[:len(binary_input)],
            transform_used=self.selected_variant,
            wave_spectrum=waveform,
            computation_time=computation_time,
            oscillation_frequency=float(avg_frequency)
        )

    def validate_all_unitaries(
        self,
        *,
        matrix_sizes: Tuple[int, ...] = (64,),
        sample_bytes: int = 64,
        rng_seed: int = 1337,
    ) -> Dict[str, Any]:
        """Validate Φ-RFT middleware across all registered unitary variants."""
        results = []
        rng = np.random.default_rng(rng_seed)
        previous_variant = self.selected_variant
        try:
            for key, info in self.variants.items():
                variant_entry: Dict[str, Any] = {
                    "key": key,
                    "name": info.name,
                    "unitarity": [],
                }
                last_matrix = None
                for size in matrix_sizes:
                    try:
                        matrix = info.generator(size)
                        identity = np.eye(size, dtype=np.complex128)
                        error = float(np.linalg.norm(matrix.conj().T @ matrix - identity))
                        variant_entry["unitarity"].append({
                            "size": size,
                            "error": error,
                            "passed": error < 1e-10,
                        })
                        if size == matrix_sizes[-1]:
                            last_matrix = matrix
                    except Exception as exc:
                        variant_entry["unitarity"].append({
                            "size": size,
                            "error": None,
                            "passed": False,
                            "message": str(exc),
                        })
                sample = rng.integers(0, 256, size=sample_bytes, dtype=np.uint8).tobytes()
                start = time.time()
                self.selected_variant = key
                try:
                    waveform = self.binary_to_waveform(sample)
                    reconstructed = self.waveform_to_binary(waveform, len(sample) * 8)
                finally:
                    self.selected_variant = previous_variant
                elapsed = time.time() - start
                original_bits = np.unpackbits(np.frombuffer(sample, dtype=np.uint8))
                recovered_bits = np.unpackbits(np.frombuffer(reconstructed[:len(sample)], dtype=np.uint8))
                min_len = min(original_bits.size, recovered_bits.size)
                bit_errors = int(np.sum(original_bits[:min_len] != recovered_bits[:min_len]))
                bit_error_rate = bit_errors / max(1, min_len)
                freq_bins = np.arange(len(waveform))
                power = np.abs(waveform) ** 2
                frequency = float(np.sum(freq_bins * power) / np.sum(power)) if power.sum() else 0.0
                variant_entry["round_trip"] = {
                    "bytes": sample_bytes,
                    "bit_errors": bit_errors,
                    "bit_error_rate": bit_error_rate,
                    "time_ms": elapsed * 1000.0,
                    "oscillation_frequency": frequency,
                    "passed": bit_errors == 0,
                }
                if HAS_ASSEMBLY and key == "original" and last_matrix is not None:
                    try:
                        asm = UnitaryRFT(matrix_sizes[-1], RFT_FLAG_QUANTUM_SAFE)
                        vec = rng.standard_normal(matrix_sizes[-1]) + 1j * rng.standard_normal(matrix_sizes[-1])
                        python_wave = last_matrix @ vec
                        asm_wave = asm.forward(vec.astype(np.complex128))
                        delta = float(np.linalg.norm(python_wave - asm_wave) / np.linalg.norm(vec))
                        variant_entry["assembly_delta"] = {
                            "size": matrix_sizes[-1],
                            "relative_error": delta,
                        }
                    except Exception as exc:
                        variant_entry["assembly_delta"] = {
                            "size": matrix_sizes[-1],
                            "error": str(exc),
                        }
                else:
                    variant_entry["assembly_delta"] = None
                results.append(variant_entry)
        finally:
            self.selected_variant = previous_variant
        return {
            "assembly_available": HAS_ASSEMBLY,
            "matrix_sizes": list(matrix_sizes),
            "sample_bytes": sample_bytes,
            "variants": results,
        }
    
    def get_variant_info(self, variant_name: str) -> Optional[VariantInfo]:
        return self.variants.get(variant_name) or self.experimental_variants.get(variant_name)
    
    def list_all_variants(self) -> Dict[str, VariantInfo]:
        return {**self.variants, **self.experimental_variants}

    @staticmethod
    @lru_cache(maxsize=64)
    def _wave_fibonacci_candidate_bins(n: int) -> np.ndarray:
        bins = {0}
        a, b = 1, 1
        while a < n:
            bins.add(a)
            bins.add((n - a) % n)
            a, b = b, a + b
        if n > 1:
            bins.add(n // 2)
        return np.array(sorted(bins), dtype=np.int64)

    @staticmethod
    @lru_cache(maxsize=64)
    def _wave_fibonacci_phase(n: int) -> np.ndarray:
        idx = np.arange(n, dtype=np.float64)
        frac = np.modf((idx + 1.0) * PHI)[0]
        return np.exp(2j * np.pi * frac).astype(np.complex128)

    def _wave_fibonacci_pruned_forward(self, signal: np.ndarray) -> np.ndarray:
        # The probe showed most energy landing on a small Fibonacci/golden bin set.
        n = len(signal)
        phase = self._wave_fibonacci_phase(n)
        bins = self._wave_fibonacci_candidate_bins(n)
        dense = np.fft.fft(signal, norm="ortho").astype(np.complex128) * phase
        sparse = np.zeros(n, dtype=np.complex128)
        sparse[bins] = dense[bins]
        return sparse

    def _wave_fibonacci_pruned_inverse(self, waveform: np.ndarray) -> np.ndarray:
        n = len(waveform)
        phase = self._wave_fibonacci_phase(n)
        coeffs = np.asarray(waveform, dtype=np.complex128) / phase
        return np.fft.ifft(coeffs, norm="ortho").astype(np.complex128)


class QuantumEngine:
    """
    Unified Quantum Operations Engine
    """
    
    def __init__(self, num_qubits: int = 4):
        self.num_qubits = num_qubits
        self.state_size = 2 ** num_qubits
        self.state = np.zeros(self.state_size, dtype=complex)
        self.state[0] = 1.0
        self._gate_cache: Dict[str, np.ndarray] = {}
        self.circuit: List[Dict[str, Any]] = []
        self._init_standard_gates()
        self._init_rft_unitaries()
        print(f"🔬 QuantumEngine initialized: {num_qubits} qubits ({self.state_size} amplitudes)")
    
    def _init_standard_gates(self):
        if not HAS_QUANTUM_GATES:
            print("⚠️  Quantum gates module not available")
            return
        self._gate_cache['X'] = PauliGates.X().matrix
        self._gate_cache['Y'] = PauliGates.Y().matrix
        self._gate_cache['Z'] = PauliGates.Z().matrix
        self._gate_cache['H'] = HadamardGates.H().matrix
        self._gate_cache['S'] = PhaseGates.S().matrix
        self._gate_cache['T'] = PhaseGates.T().matrix
        self._gate_cache['CNOT'] = ControlledGates.CNOT().matrix
        self._gate_cache['CZ'] = ControlledGates.CZ().matrix
        self._gate_cache['Toffoli'] = ControlledGates.Toffoli().matrix
        print(f"   ✅ Loaded {len(self._gate_cache)} standard quantum gates")
    
    def _init_rft_unitaries(self):
        for size in [2, 4, 8, 16, 32, 64]:
            if size <= self.state_size:
                try:
                    rft = rft_matrix(size)
                    self._gate_cache[f'RFT_{size}'] = rft
                except Exception as e:
                    print(f"   ⚠️  Could not create RFT_{size}: {e}")
        phi = PHI
        for k in range(1, 5):
            theta = 2 * np.pi * phi * k / 10
            self._gate_cache[f'Rphi_{k}'] = RotationGates.Rz(theta).matrix
        print(f"   ✅ Loaded RFT unitaries up to size {self.state_size}")
    
    def reset(self):
        self.state = np.zeros(self.state_size, dtype=complex)
        self.state[0] = 1.0
        self.circuit = []
    
    def get_state(self) -> np.ndarray:
        return self.state.copy()
    
    def set_state(self, state: np.ndarray):
        if len(state) != self.state_size:
            raise ValueError(f"State size mismatch: expected {self.state_size}, got {len(state)}")
        norm = np.linalg.norm(state)
        if not np.isclose(norm, 1.0, atol=1e-10):
            raise ValueError(f"State not normalized: |ψ| = {norm}")
        self.state = state.astype(complex)
    
    def apply_gate(self, gate_name: str, target: int, control: Optional[int] = None) -> np.ndarray:
        if gate_name not in self._gate_cache:
            raise ValueError(f"Unknown gate: {gate_name}")
        gate = self._gate_cache[gate_name]
        gate_size = gate.shape[0]
        gate_qubits = int(np.log2(gate_size))
        if gate_qubits == 1:
            full_gate = self._expand_single_qubit_gate(gate, target)
        elif gate_qubits == 2:
            if control is None:
                control = target
                target = (target + 1) % self.num_qubits
            full_gate = self._expand_two_qubit_gate(gate, control, target)
        else:
            raise NotImplementedError(f"{gate_qubits}-qubit gates not yet supported")
        self.state = full_gate @ self.state
        self.circuit.append({'gate': gate_name, 'target': target, 'control': control})
        return self.state
    
    def _expand_single_qubit_gate(self, gate: np.ndarray, target: int) -> np.ndarray:
        result = np.array([[1]], dtype=complex)
        for i in range(self.num_qubits):
            if i == target:
                result = np.kron(result, gate)
            else:
                result = np.kron(result, np.eye(2, dtype=complex))
        return result
    
    def _expand_two_qubit_gate(self, gate: np.ndarray, control: int, target: int) -> np.ndarray:
        n = self.num_qubits
        dim = 2 ** n
        full_gate = np.eye(dim, dtype=complex)
        for i in range(dim):
            ctrl_bit = (i >> (n - 1 - control)) & 1
            tgt_bit = (i >> (n - 1 - target)) & 1
            if ctrl_bit == 1:
                for j in range(dim):
                    ctrl_bit_j = (j >> (n - 1 - control)) & 1
                    tgt_bit_j = (j >> (n - 1 - target)) & 1
                    if ctrl_bit_j == 1:
                        mask = ~((1 << (n - 1 - control)) | (1 << (n - 1 - target)))
                        if (i & mask) == (j & mask):
                            gate_row = tgt_bit
                            gate_col = tgt_bit_j
                            full_gate[i, j] = gate[2 + gate_row, 2 + gate_col]
        return full_gate
    
    def apply_rotation(self, axis: str, theta: float, target: int) -> np.ndarray:
        if axis.lower() == 'x':
            gate = RotationGates.Rx(theta).matrix
        elif axis.lower() == 'y':
            gate = RotationGates.Ry(theta).matrix
        elif axis.lower() == 'z':
            gate = RotationGates.Rz(theta).matrix
        else:
            raise ValueError(f"Unknown axis: {axis}")
        full_gate = self._expand_single_qubit_gate(gate, target)
        self.state = full_gate @ self.state
        self.circuit.append({'gate': f'R{axis}({theta:.4f})', 'target': target, 'control': None})
        return self.state
    
    def apply_rft(self, size: Optional[int] = None) -> np.ndarray:
        if size is None:
            size = self.state_size
        rft_key = f'RFT_{size}'
        if rft_key not in self._gate_cache:
            self._gate_cache[rft_key] = rft_matrix(size)
        rft = self._gate_cache[rft_key]
        if size == self.state_size:
            self.state = rft @ self.state
        else:
            self.state[:size] = rft @ self.state[:size]
        self.circuit.append({'gate': f'RFT_{size}', 'target': 'all', 'control': None})
        return self.state
    
    def apply_inverse_rft(self, size: Optional[int] = None) -> np.ndarray:
        if size is None:
            size = self.state_size
        rft_key = f'RFT_{size}'
        if rft_key not in self._gate_cache:
            self._gate_cache[rft_key] = rft_matrix(size)
        rft_dag = self._gate_cache[rft_key].conj().T
        if size == self.state_size:
            self.state = rft_dag @ self.state
        else:
            self.state[:size] = rft_dag @ self.state[:size]
        return self.state
    
    def create_bell_state(self, qubit1: int = 0, qubit2: int = 1) -> np.ndarray:
        self.reset()
        self.apply_gate('H', qubit1)
        self.apply_gate('CNOT', qubit2, control=qubit1)
        return self.state
    
    def create_ghz_state(self) -> np.ndarray:
        self.reset()
        self.apply_gate('H', 0)
        for i in range(1, self.num_qubits):
            self.apply_gate('CNOT', i, control=i-1)
        return self.state
    
    def measure_probabilities(self) -> np.ndarray:
        return np.abs(self.state) ** 2
    
    def measure(self) -> int:
        probs = self.measure_probabilities()
        result = np.random.choice(len(probs), p=probs)
        self.state = np.zeros(self.state_size, dtype=complex)
        self.state[result] = 1.0
        return result
    
    def fidelity(self, target_state: np.ndarray) -> float:
        return float(np.abs(np.vdot(self.state, target_state)) ** 2)
    
    def validate_unitarity(self, gate_name: str) -> float:
        if gate_name not in self._gate_cache:
            raise ValueError(f"Unknown gate: {gate_name}")
        gate = self._gate_cache[gate_name]
        identity = np.eye(gate.shape[0], dtype=complex)
        return float(np.linalg.norm(gate.conj().T @ gate - identity))
    
    def list_available_gates(self) -> List[str]:
        return list(self._gate_cache.keys())
    
    def get_circuit_depth(self) -> int:
        return len(self.circuit)
    
    def __repr__(self) -> str:
        return f"QuantumEngine({self.num_qubits} qubits, {len(self._gate_cache)} gates, depth={self.get_circuit_depth()})"


_engine = MiddlewareTransformEngine()

def select_transform(data_type: str, priority: str = 'speed', size: int = 1024) -> str:
    profile = TransformProfile(data_type=data_type, priority=priority, size=size)
    return _engine.select_optimal_transform(profile)

def compute_wave(binary_data: bytes, operation: str = "identity") -> bytes:
    result = _engine.compute_in_wavespace(binary_data, operation)
    return result.output_binary

def get_available_transforms() -> list:
    return list(_engine.list_all_variants().keys())


if __name__ == "__main__":
    print("=" * 70)
    print("quantoniumos Middleware Transform Engine")
    print("Binary (01) → Oscillating Waves → Computation → Binary (01)")
    print("=" * 70)
    test_data = b"quantoniumos: Wave Computing"
    print(f"\n📥 Input: {test_data.decode()}")
    print(f"   Binary size: {len(test_data)} bytes = {len(test_data) * 8} bits")
    print(f"\n🔄 Available Transform Variants: {len(VARIANTS)}")
    for variant_name, info in VARIANTS.items():
        print(f"   • {info.name:25} → {info.use_case}")
    for priority in ['speed', 'accuracy', 'security', 'compression']:
        print(f"\n🎯 Testing with priority: {priority.upper()}")
        profile = TransformProfile(data_type='text', priority=priority, size=len(test_data))
        result = _engine.compute_in_wavespace(test_data, operation="identity", profile=profile)
        print(f"   Transform: {result.transform_used}")
        print(f"   Wave frequency: {result.oscillation_frequency:.2f} Hz")
        print(f"   Computation time: {result.computation_time * 1000:.3f} ms")
        print(f"   Output matches: {result.output_binary == test_data}")
    print("\n✅ Middleware engine operational!")
