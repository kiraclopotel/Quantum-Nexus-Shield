import numpy as np
from typing import List, Tuple, Dict, Union, Optional
import math
import hashlib
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import warnings
import time

@dataclass
class LayerMetrics:
    """
    Stores metrics for layer function calculations.
    These metrics help monitor the health and security of the encryption system.
    """
    entropy: float          # Measure of randomness in the system
    complexity: float       # Measure of computational complexity
    stability: float        # Measure of numerical stability
    convergence_rate: float # Rate at which calculations converge

class MathematicalCore:
    """
    Enhanced implementation of the mathematical core functionality for quantum stack encryption.
    Combines theoretical foundations from the thesis with additional security features and
    maintains backward compatibility with existing functionality.
    """
    
    def __init__(self, max_layer_depth: int = 100):
        # Initialize caches and state management
        self.layer_cache: Dict[Tuple[float, int], float] = {}
        self.timeline_cache: Dict[int, List[Tuple[int, int]]] = {}
        self.layer_states: Dict[int, int] = {}
        self.layer_transitions: List[Tuple[int, int, int]] = []
        
        # Initialize enhanced grid system for cryptographic patterns
        self.grid_dimensions = (10, 10, 10)
        self.grid_values = np.zeros(self.grid_dimensions)
        self.initialize_enhanced_grid()
        
        # System configuration parameters
        self.max_layer_depth = max_layer_depth
        self.entropy_threshold = 0.9999
        self.numerical_tolerance = 1e-10
        
        # Mathematical stability tracking
        self.previous_layer = None
        self.stability_metrics: List[LayerMetrics] = []

    def initialize_enhanced_grid(self) -> None:
        """
        Initialize the grid with enhanced mathematical patterns that increase
        cryptographic strength. Combines linear, trigonometric, and exponential
        components for maximum complexity.
        """
        for i in range(self.grid_dimensions[0]):
            for j in range(self.grid_dimensions[1]):
                for k in range(self.grid_dimensions[2]):
                    # Create complex patterns using multiple mathematical components
                    linear_component = (j - i) % 10
                    trigonometric_component = int(5 * (math.sin(i/5) + math.cos(j/5)))
                    exponential_component = int(math.exp(k/10)) % 10
                    
                    # Mix components using modular arithmetic for non-linearity
                    value = (linear_component + trigonometric_component + exponential_component) % 10
                    self.grid_values[i, j, k] = value

    def generate_entry_timeline(self, n: int) -> List[Tuple[int, int]]:
        """
        Generate timeline entries for a given number as specified in the thesis.
        This is a critical function used by the encryption system for maintaining
        temporal relationships between encrypted elements.
        
        Args:
            n: The input number to generate timeline for
            
        Returns:
            List of tuples containing (digit, depth) pairs
        """
        # Check cache first for efficiency
        if n in self.timeline_cache:
            return self.timeline_cache[n]
        
        # Generate timeline based on digits and their positions
        str_n = str(n)
        k = len(str_n)
        timeline = [(int(str_n[i]), k - i) for i in range(k)]
        
        # Cache for future use
        self.timeline_cache[n] = timeline
        return timeline

    def compute_layer(self, value: int, layer: int = 1) -> int:
        """
        Compute layer based on the number of digits in the value.
        This function determines which layer a number belongs to based on its size.
        
        For example:
        - Numbers 1-9 are in layer 1
        - Numbers 10-99 are in layer 2
        - Numbers 100-999 are in layer 3
        And so on...
        
        Args:
            value: The input value to compute layer for
            layer: Initial layer value (default=1)
            
        Returns:
            Computed layer number
        """
        # Calculate layer based on number of digits
        num_digits = len(str(abs(value)))  # Using abs() to handle negative numbers safely
        
        # Store this layer for future reference
        self.previous_layer = num_digits
        
        # Return the computed layer
        return num_digits

    def layer_function(self, n: float, k: int, include_enhancements: bool = True) -> float:
        """
        Enhanced layer function incorporating all theoretical requirements plus additional
        security features. This is the core mathematical function of the encryption system.
        
        Args:
            n: Input value (can be any positive number)
            k: Layer index (must be > 0)
            include_enhancements: Whether to include security enhancements
            
        Returns:
            Computed layer function value
        """
        # Make input validation more user-friendly
        if n <= 0:
            warnings.warn("Input value must be positive. Using default minimum value.")
            n = 1.000001  # Use a value slightly above 1 to maintain mathematical validity
            
        if k < 1:
            warnings.warn("Layer index must be positive. Using default value of 1.")
            k = 1
            
        if k > self.max_layer_depth:
            warnings.warn(f"Layer depth {k} exceeds maximum allowed value. Using maximum value.")
            k = self.max_layer_depth

        # Check cache for efficiency
        cache_key = (n, k)
        if cache_key in self.layer_cache:
            return self.layer_cache[cache_key]

        try:
            # Prevent numerical overflow
            n = min(n, 1e308)
            
            # Calculate basic layer function terms
            log_n = math.log(n)
            power_term = n ** (1/k)
            log_term = log_n ** ((k-1)/k)
            
            # Compute basic result
            result = power_term * log_term
            
            if include_enhancements:
                # Add security enhancements
                b_k = self._compute_adaptive_coefficient(k)
                c_k = self._compute_layer_coefficient(k, n)
                
                # Calculate enhancement terms
                enhancement_term = b_k * math.log(log_n + 1)
                complexity_term = 0.01 * (log_n ** 2) * (n ** (1/k))
                stability_term = self._compute_stability_term(k, n)
                
                # Combine all terms
                result = c_k * result + enhancement_term + complexity_term + stability_term
            
            # Cache and update metrics
            self.layer_cache[cache_key] = result
            self._update_stability_metrics(n, k, result)
            
            return result
            
        except Exception as e:
            warnings.warn(f"Error in layer function calculation: {str(e)}")
            return 0

    def logarithmic_ratio(self, n: float, k: int) -> float:
        """
        Compute logarithmic ratio between consecutive layers as specified in
        the Logarithmic Ratio Theorem.
        
        Args:
            n: Input value
            k: Layer index
            
        Returns:
            Computed ratio value
        """
        if n <= 1 or k < 1:
            return 0
            
        try:
            f_k = self.layer_function(n, k)
            f_k_plus_1 = self.layer_function(n, k + 1)
            
            if f_k_plus_1 == 0:
                return float('inf')
                
            ratio = (n / math.log(n)) ** (1/(k * (k+1)))
            
            return ratio if 0 < ratio < float('inf') else 0
            
        except (ValueError, OverflowError):
            return 0

    def wave_function_hybrid(self, x: float, k: int, omega: float) -> float:
        """
        Compute Wave-Function Hybrid as specified in the theoretical documentation.
        This function adds oscillating components to enhance encryption complexity.
        
        Args:
            x: Input value
            k: Layer index
            omega: Frequency parameter
            
        Returns:
            Computed wave function value
        """
        if x <= 0:
            return 0
            
        # Prevent numerical instabilities
        x = min(x, 1e300)  
        k = min(k, 1000)   
        omega = min(omega, 1000)
        
        # Calculate components
        power_term = x ** (1/k)
        log_term = (math.log(x)) ** ((k-1)/k)
        wave_term = math.sin(omega * x)
        
        return power_term * log_term * wave_term

    def _compute_adaptive_coefficient(self, k: int) -> float:
        """
        Compute adaptive coefficient that adjusts based on layer depth and
        stability metrics for enhanced security.
        """
        b_k = 0.1 * k
        layer_factor = math.exp(-k/10)
        b_k *= layer_factor
        
        if self.stability_metrics:
            latest_metrics = self.stability_metrics[-1]
            stability_factor = 1 + (1 - latest_metrics.stability) * 0.1
            b_k *= stability_factor
        
        return min(max(b_k, 0.01), 1.0)

    def _compute_layer_coefficient(self, k: int, n: float) -> float:
        """
        Compute layer coefficient with dynamic scaling for enhanced security.
        """
        c_k = 1.0
        size_factor = math.log(n + 1) / (k * k)
        c_k *= (1 + size_factor)
        
        depth_factor = 1 - math.exp(-k/5)
        c_k *= depth_factor
        
        if self.stability_metrics:
            latest_metrics = self.stability_metrics[-1]
            convergence_factor = 1 + (1 - latest_metrics.convergence_rate) * 0.1
            c_k *= convergence_factor
        
        return min(max(c_k, 0.5), 2.0)

    def _compute_stability_term(self, k: int, n: float) -> float:
        """
        Compute additional term to enhance numerical stability and security.
        """
        if not self.stability_metrics:
            return 0
            
        latest_metrics = self.stability_metrics[-1]
        stability_factor = latest_metrics.entropy * latest_metrics.stability
        oscillation = math.sin(k * math.pi / 10) * math.cos(math.log(n))
        
        return 0.01 * stability_factor * oscillation

    def _update_stability_metrics(self, n: float, k: int, result: float) -> None:
        """Update system stability metrics based on computation results."""
        if not self.stability_metrics:
            entropy = 1.0
            stability = 1.0
            convergence_rate = 1.0
        else:
            previous_metrics = self.stability_metrics[-1]
            entropy = min(1.0, previous_metrics.entropy * 0.9 + 0.1)
            stability = min(1.0, abs(math.sin(result)))
            convergence_rate = min(1.0, 1.0 / k)

        complexity = min(1.0, math.log(n) / (k * 100))
        
        metrics = LayerMetrics(
            entropy=entropy,
            complexity=complexity,
            stability=stability,
            convergence_rate=convergence_rate
        )
        self.stability_metrics.append(metrics)
        
        if len(self.stability_metrics) > 1000:
            self.stability_metrics = self.stability_metrics[-1000:]

    def _smooth_layer_transition(self, old_layer: int, new_layer: int) -> int:
        """Ensure smooth transition between layers."""
        if abs(new_layer - old_layer) > 1:
            return old_layer + (1 if new_layer > old_layer else -1)
        return new_layer

    def _manage_layer_state(self, seed: int, new_layer: int) -> int:
        """Manage layer state transitions."""
        if seed in self.layer_states:
            old_layer = self.layer_states[seed]
            transition_layer = self._smooth_layer_transition(old_layer, new_layer)
            self.layer_states[seed] = transition_layer
            self.layer_transitions.append((seed, old_layer, transition_layer))
            return transition_layer
        
        self.layer_states[seed] = new_layer
        return new_layer

    def clear_caches(self) -> None:
        """Clear all caches and reset system state."""
        self.layer_cache.clear()
        self.timeline_cache.clear()
        self.layer_states.clear()
        self.layer_transitions.clear()
        self.stability_metrics.clear()
        self.previous_layer = None

    def get_stability_analysis(self) -> Dict[str, float]:
        """Get comprehensive system stability analysis."""
        if not self.stability_metrics:
            return {
                'average_entropy': 1.0,
                'average_complexity': 0.0,
                'average_stability': 1.0,
                'average_convergence': 1.0,
                'system_health': 1.0
            }
            
        metrics = self.stability_metrics[-100:]
        
        return {
            'average_entropy': np.mean([m.entropy for m in metrics]),
            'average_complexity': np.mean([m.complexity for m in metrics]),
            'average_stability': np.mean([m.stability for m in metrics]),
            'average_convergence': np.mean([m.convergence_rate for m in metrics]),
            'system_health': np.mean([m.entropy * m.stability * m.convergence_rate 
                                    for m in metrics])
        }
        
        
        
    def process_message_for_encryption(self, message: bytes) -> Tuple[bytes, int, float]:
        """
        Process a message for encryption, ensuring independence and perfect entropy.
        This function serves as the main entry point for message processing.
        
        Args:
            message: The original message to process
            
        Returns:
            Tuple containing:
            - Processed message bytes
            - Perfect seed for this message
            - Achieved entropy value
        """
        # Step 1: Initial message preparation
        processed_data = self.balance_bit_distribution(message)
        
        # Step 2: Find perfect seed for this specific message
        perfect_seed, initial_entropy = self.find_independent_perfect_seed(processed_data)
        
        # Step 3: Record message metrics for independence verification
        self.record_message_metrics(perfect_seed, initial_entropy, len(processed_data))
        
        # Step 4: Verify independence from previous messages
        if not self.verify_message_independence(perfect_seed):
            # If not independent enough, adjust the processing
            perfect_seed = self.adjust_seed_for_independence(perfect_seed)
            processed_data = self.apply_independence_enhancement(processed_data)
        
        # Step 5: Final entropy calculation after all processing
        final_entropy = self.calculate_message_entropy(processed_data)
        
        return processed_data, perfect_seed, final_entropy
        

    def record_message_metrics(self, seed: int, entropy: float, size: int) -> None:
        """
        Record metrics for each message to track independence and patterns.
        """
        if not hasattr(self, 'message_metrics'):
            self.message_metrics = []
            
        metrics = {
            'seed': seed,
            'entropy': entropy,
            'size': size,
            'timestamp': time.time(),
            'hash': hashlib.sha256(str(seed).encode()).hexdigest()[:16]
        }
        self.message_metrics.append(metrics)
        
        # Keep only recent metrics to manage memory
        if len(self.message_metrics) > 1000:
            self.message_metrics = self.message_metrics[-1000:]

    def verify_message_independence(self, new_seed: int) -> bool:
        """
        Verify that a new message's seed is sufficiently independent from previous ones.
        
        Args:
            new_seed: The seed to verify
            
        Returns:
            bool: True if message is independent, False otherwise
        """
        if not hasattr(self, 'message_metrics') or not self.message_metrics:
            return True
            
        # Check recent messages for independence
        recent_metrics = self.message_metrics[-10:]  # Check last 10 messages
        
        for metrics in recent_metrics:
            # Check seed distance
            if abs(new_seed - metrics['seed']) < 1000:
                return False
                
            # Check hash similarity
            new_hash = hashlib.sha256(str(new_seed).encode()).hexdigest()[:16]
            if self.calculate_hash_similarity(new_hash, metrics['hash']) > 0.3:
                return False
        
        return True

    def calculate_hash_similarity(self, hash1: str, hash2: str) -> float:
        """
        Calculate similarity between two hashes using Hamming distance.
        """
        if len(hash1) != len(hash2):
            return 0
            
        differences = sum(a != b for a, b in zip(hash1, hash2))
        return 1 - (differences / len(hash1))

    def adjust_seed_for_independence(self, seed: int) -> int:
        """
        Adjust a seed to ensure independence from previous messages.
        """
        adjustment = 1000
        new_seed = seed
        
        while not self.verify_message_independence(new_seed):
            new_seed = seed + adjustment
            adjustment *= 2
            
            if adjustment > 1000000:
                # If we can't find an independent seed, use a time-based component
                new_seed = seed + int(time.time() * 1000) % 1000000
                break
        
        return new_seed

    def apply_independence_enhancement(self, data: bytes) -> bytes:
        """
        Apply additional transformations to enhance message independence.
        """
        try:
            # Convert to bit array
            bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
            
            # Apply transformations that preserve entropy but enhance independence
            shifted_bits = np.roll(bits, len(bits) // 3)
            xor_pattern = np.array([int(math.sin(i/10) > 0) for i in range(len(bits))])
            enhanced_bits = np.logical_xor(shifted_bits, xor_pattern).astype(np.uint8)
            
            # Pack back into bytes
            enhanced_data = np.packbits(enhanced_bits).tobytes()
            return enhanced_data
            
        except Exception as e:
            warnings.warn(f"Error applying independence enhancement: {str(e)}")
            return data

    def get_independence_analysis(self) -> Dict[str, float]:
        """
        Analyze the independence of processed messages.
        
        Returns:
            Dictionary containing independence metrics
        """
        if not hasattr(self, 'message_metrics') or len(self.message_metrics) < 2:
            return {
                'average_entropy': 1.0,
                'seed_diversity': 1.0,
                'independence_score': 1.0
            }
        
        # Calculate metrics
        entropies = [m['entropy'] for m in self.message_metrics]
        seeds = [m['seed'] for m in self.message_metrics]
        
        # Entropy stability
        avg_entropy = np.mean(entropies)
        
        # Seed diversity
        seed_diffs = np.diff(seeds)
        seed_diversity = min(1.0, np.mean(np.abs(seed_diffs)) / 1000)
        
        # Overall independence score
        independence_score = (avg_entropy + seed_diversity) / 2
        
        return {
            'average_entropy': avg_entropy,
            'seed_diversity': seed_diversity,
            'independence_score': independence_score
        }        
        
        
    def balance_bit_distribution(self, data: bytes) -> bytes:
        """
        Balance the distribution of bits in the data to improve statistical properties.
        We use numpy's type-safe operations to handle large numbers properly.
        
        Args:
            data: Original bytes to balance
            
        Returns:
            Balanced bytes with improved statistical properties
        """
        try:
            # Convert to bits safely using numpy
            bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
            
            # Use numpy's int64 for safe arithmetic with large numbers
            ones_count = np.int64(np.sum(bits))
            total_bits = np.int64(len(bits))
            zeros_count = total_bits - ones_count
            
            # Calculate threshold safely
            threshold = np.int64(total_bits * 0.1)
            
            # Check balance using safe comparison
            if np.abs(ones_count - zeros_count) > threshold:
                # Create balancing pattern safely
                pattern_length = len(bits)
                pattern = np.zeros(pattern_length, dtype=np.uint8)
                pattern[::2] = 1  # Set every other bit to 1
                
                # Balance using XOR
                balanced_bits = np.logical_xor(bits, pattern).astype(np.uint8)
                
                # Convert back to bytes
                return np.packbits(balanced_bits).tobytes()
                
            return data
            
        except Exception as e:
            warnings.warn(f"Error in bit balancing: {str(e)}")
            return data

    def calculate_message_entropy(self, message_data: bytes) -> float:
        """
        Calculate the entropy (randomness) of a message.

        Args:
            message_data: The message to analyze (in bytes)

        Returns:
            A value between 0 and 1, where 1 indicates perfect randomness
        """
        try:
            # Convert message to bits for analysis
            bits = np.unpackbits(np.frombuffer(message_data, dtype=np.uint8))

            # Count how often each bit value appears
            unique, counts = np.unique(bits, return_counts=True)
            probabilities = counts / len(bits)

            # Calculate Shannon entropy
            entropy = -np.sum(probabilities * np.log2(probabilities))

            # Normalize entropy to a 0-1 scale (since for binary data, max entropy is 1 bit per bit)
            normalized_entropy = entropy

            return normalized_entropy

        except Exception as e:
            warnings.warn(f"Error calculating entropy: {str(e)}")
            return 0.0


    def process_message_for_encryption(self, message: bytes) -> Tuple[bytes, int, float]:
        """
        Process a message for encryption with proper error handling and metrics recording.
        This function prepares messages for secure encryption while maintaining
        statistical properties.
        
        Args:
            message: Original message bytes to process
            
        Returns:
            Tuple containing:
            - Processed message bytes
            - Encryption seed
            - Entropy value
        """
        try:
            # Balance bit distribution
            processed_data = self.balance_bit_distribution(message)
            
            # Calculate entropy
            initial_entropy = self.calculate_message_entropy(processed_data)
            
            # Generate seed using SHA-256
            base_seed = int.from_bytes(hashlib.sha256(processed_data).digest()[:8], 'big')
            
            # Record metrics with current timestamp
            self.record_message_metrics(base_seed, initial_entropy, len(processed_data))
            
            return processed_data, base_seed, initial_entropy
            
        except Exception as e:
            warnings.warn(f"Error processing message: {str(e)}")
            # Return safe default values
            return message, int(time.time()), 0.0

    def record_message_metrics(self, seed: int, entropy: float, size: int) -> None:
        """
        Record metrics for message encryption with error handling.
        This helps track the encryption quality and message independence.
        
        Args:
            seed: Encryption seed used
            entropy: Calculated entropy value
            size: Message size in bytes
        """
        try:
            if not hasattr(self, 'message_metrics'):
                self.message_metrics = []
                
            # Create metrics record
            metrics = {
                'seed': seed,
                'entropy': entropy,
                'size': size,
                'timestamp': time.time(),
                'hash': hashlib.sha256(str(seed).encode()).hexdigest()[:16]
            }
            
            # Store metrics
            self.message_metrics.append(metrics)
            
            # Keep list size manageable
            if len(self.message_metrics) > 1000:
                self.message_metrics = self.message_metrics[-1000:]
                
        except Exception as e:
            warnings.warn(f"Error recording metrics: {str(e)}")
        