# encryption.py
from utils.helpers import ConfigManager
import numpy as np
from numpy.random import default_rng
from scipy.special import erfc
from scipy.stats import chi2
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
from typing import Tuple, Optional, List, Dict, Union
import struct
import time
import hashlib
from core.mathematics import MathematicalCore
from concurrent.futures import ThreadPoolExecutor, as_completed
import pickle
from pathlib import Path
import random
import traceback


class EnhancedTimeline:
    def __init__(self):
        self.markers = {}
        self.checksum_history = []
        self.mathematical_core = MathematicalCore()

    def create_marker(self, seed: int, message_id: int, message: bytes, entropy: float, layer: int = 1) -> dict:
        """Create enhanced timeline marker with specific layer and entropy."""
        timeline = self.mathematical_core.generate_entry_timeline(seed)
        layer_value = self.mathematical_core.compute_layer(seed, layer)
        
        marker = {
            'seed': seed,
            'id': message_id,
            'timestamp': time.time(),
            'entropy': entropy,
            'layer': layer_value,
            'timeline': timeline,
            'checksum': self.generate_checksum(seed, message_id, message)
        }
        self.markers[seed] = marker
        return marker
        
    def restore_from_data(self, timeline_data: Dict[str, List[float]]):
        """Restore timeline state from saved data"""
        try:
            if not timeline_data or not isinstance(timeline_data, dict):
                return

            # Reconstruct markers from saved data
            self.markers = {}
            for i, (seed, timestamp, entropy, layer) in enumerate(zip(
                timeline_data['seeds'],
                timeline_data['timestamps'],
                timeline_data['entropies'],
                timeline_data['layers']
            )):
                self.markers[seed] = {
                    'seed': seed,
                    'id': i,
                    'timestamp': timestamp,
                    'entropy': entropy,
                    'layer': layer,
                    'timeline': self.mathematical_core.generate_entry_timeline(seed),
                    'checksum': self.generate_checksum(seed, i, b"")  # Placeholder checksum
                }

            # Restore checksum history if available
            if 'checksum_counts' in timeline_data and timeline_data['checksum_counts']:
                self.checksum_history = [""] * max(timeline_data['checksum_counts'])

        except Exception as e:
            print(f"Error restoring timeline data: {e}")
            # Initialize empty state if restoration fails
            self.markers = {}
            self.checksum_history = []    

    def verify_marker(self, seed: int, message_id: int, message: bytes) -> bool:
        """Verify if a given marker's checksum matches calculated checksum."""
        if seed not in self.markers:
            return False
        marker = self.markers[seed]
        expected_checksum = self.generate_checksum(seed, message_id, message)
        return marker['checksum'] == expected_checksum        

    def generate_checksum(self, seed: int, message_id: int, message: bytes) -> str:
        """Generate a checksum for a marker, including timeline properties."""
        timeline_str = str(self.mathematical_core.generate_entry_timeline(seed))
        checksum_data = f"{seed}{message_id}{message}{timeline_str}".encode()
        checksum = hashlib.sha256(checksum_data).hexdigest()
        self.checksum_history.append(checksum)
        return checksum

    def verify_cumulative_hash(self, cumulative_hash: str) -> bool:
        """Verify cumulative hash by comparing it to combined checksums only if there are markers."""
        if not self.markers:
            print("No markers available to verify cumulative hash.")
            return False

        combined_checksums = ''.join(self.markers[seed]['checksum'] for seed in sorted(self.markers))
        expected_hash = hashlib.sha256(combined_checksums.encode()).hexdigest()
        return cumulative_hash == expected_hash

    def get_visualization_data(self) -> Dict[str, List[float]]:
        """Provide data for visualization in the format expected by VisualizationTab."""
        if not self.markers:
            return {
                'timestamps': [],
                'layers': [],
                'entropies': [],
                'depths': [],
                'seeds': [],
                'checksum_counts': []
            }

        visualization_data = {
            'timestamps': [],
            'layers': [],
            'entropies': [],
            'depths': [],
            'seeds': [],
            'checksum_counts': []
        }

        for seed, marker in self.markers.items():
            visualization_data['timestamps'].append(marker['timestamp'])
            visualization_data['layers'].append(marker['layer'])
            visualization_data['entropies'].append(marker['entropy'])
            visualization_data['depths'].append(len(marker['timeline']))
            visualization_data['seeds'].append(seed)
            visualization_data['checksum_counts'].append(len(self.checksum_history))

        return visualization_data

    def get_layer_statistics(self) -> Dict[str, float]:
        """Calculate statistical properties of the layer history."""
        layers = [marker['layer'] for marker in self.markers.values()]
        
        if not layers:
            return {
                'mean_layer': 0,
                'std_layer': 0,
                'min_layer': 0,
                'max_layer': 0,
                'total_layers': 0
            }
        
        return {
            'mean_layer': float(np.mean(layers)),
            'std_layer': float(np.std(layers)),
            'min_layer': int(np.min(layers)),
            'max_layer': int(np.max(layers)),
            'total_layers': len(set(layers))
        }
    
 
    
class QuantumStackEncryption:
    def __init__(self):
        self.messages = []
        self.perfect_seeds = []
        self.encryption_data = []
        self.timeline = EnhancedTimeline()
        self.entropy_history = []
        self.math_core = MathematicalCore()
        self.used_seeds = set()
        self.state = {}
        self.config_manager = ConfigManager()
        self.load_state()

    def save_state(self):
        """Save encryption state to file"""
        state_data = {
            'messages': [m.hex() if isinstance(m, bytes) else m for m in self.messages],
            'perfect_seeds': list(self.perfect_seeds),
            'encryption_data': [(iv.hex(), ct.hex(), e) for iv, ct, e in self.encryption_data],
            'compressed_data': self.state.get('compressed_data', {
                'encrypted_storage': {},
                'pattern_storage': {},
                'active_identifiers': set()
            }),
            'timeline_data': self.timeline.get_visualization_data(),
            'entropy_history': self.entropy_history,
            'used_seeds': list(self.used_seeds)
        }
        
        try:
            # Convert sets to lists for serialization
            if 'compressed_data' in state_data:
                if 'active_identifiers' in state_data['compressed_data']:
                    state_data['compressed_data']['active_identifiers'] = list(
                        state_data['compressed_data']['active_identifiers']
                    )
                
                # Convert bytes to hex strings
                if 'encrypted_storage' in state_data['compressed_data']:
                    for identifier, data in state_data['compressed_data']['encrypted_storage'].items():
                        if 'combined_data' in data and isinstance(data['combined_data'], bytes):
                            data['combined_data'] = data['combined_data'].hex()
            
            state_file = Path("quantum_stack_state.enc")
            with open(state_file, 'wb') as f:
                pickle.dump(state_data, f)
        except Exception as e:
            print(f"Error saving state: {e}")

    def load_state(self):
        """Load encryption state from file"""
        try:
            state_file = Path("quantum_stack_state.enc")
            if not state_file.exists():
                return
                
            with open(state_file, 'rb') as f:
                state_data = pickle.load(f)
                
                # Convert lists back to sets
                if 'compressed_data' in state_data:
                    if 'active_identifiers' in state_data['compressed_data']:
                        state_data['compressed_data']['active_identifiers'] = set(
                            state_data['compressed_data']['active_identifiers']
                        )
                    
                    # Convert hex strings back to bytes
                    if 'encrypted_storage' in state_data['compressed_data']:
                        for identifier, data in state_data['compressed_data']['encrypted_storage'].items():
                            if 'combined_data' in data and isinstance(data['combined_data'], str):
                                data['combined_data'] = bytes.fromhex(data['combined_data'])
                
                # Restore main state data
                self.messages = [bytes.fromhex(m) if isinstance(m, str) else m for m in state_data.get('messages', [])]
                self.perfect_seeds = state_data.get('perfect_seeds', [])
                self.encryption_data = [(bytes.fromhex(iv), bytes.fromhex(ct), e) 
                                      for iv, ct, e in state_data.get('encryption_data', [])]
                self.entropy_history = state_data.get('entropy_history', [])
                self.used_seeds = set(state_data.get('used_seeds', []))
                
                # Store compressed data
                self.state = {
                    'compressed_data': state_data.get('compressed_data', {
                        'encrypted_storage': {},
                        'pattern_storage': {},
                        'active_identifiers': set()
                    })
                }
                
                # Restore timeline
                if 'timeline_data' in state_data:
                    self.timeline.restore_from_data(state_data['timeline_data'])
                    
        except Exception as e:
            print(f"Error loading state: {e}")
            self.state = {}

    def generate_adaptive_key(self, seed: int, message_length: int) -> bytes:
        """Generate encryption key"""
        rng = default_rng(seed)
        return rng.integers(0, 256, size=32, dtype=np.uint8).tobytes()
    
    
    def test_seed_entropy(self, message: bytes, seed: int) -> Dict[str, Union[float, bool]]:
        """
        Test if a given seed can achieve the required entropy without modifying the system state.
        
        Args:
            message: The message bytes to test
            seed: The seed to evaluate
            
        Returns:
            Dictionary containing entropy value and validation status
        """
        try:
            # Generate encryption key using the provided seed
            key = self.generate_adaptive_key(seed, len(message))

            # Create cipher and encrypt the message
            cipher = AES.new(key, AES.MODE_CBC)
            ciphertext = cipher.encrypt(pad(message, AES.block_size))

            # Convert ciphertext into bits for entropy calculation
            bits = np.unpackbits(np.frombuffer(ciphertext, dtype=np.uint8))

            # Calculate entropy using Shannon's formula
            unique, counts = np.unique(bits, return_counts=True)
            probabilities = counts / len(bits)
            entropy = -np.sum(probabilities * np.log2(probabilities)) / np.log2(2)

            # Check if entropy meets our requirement
            is_valid = entropy >= 0.9999

            return {
                'entropy': float(entropy),
                'valid': is_valid,
                'seed': seed,
                'message_size': len(message)
            }

        except Exception as e:
            print(f"Error testing seed entropy: {str(e)}")
            return {
                'entropy': 0.0,
                'valid': False,
                'seed': seed,
                'error': str(e)
            }
    
    
    
    def encrypt_with_seed(self, message: bytes, seed: int) -> Tuple[bytes, bytes]:
        """Encrypt message using seed"""
        key = self.generate_adaptive_key(seed, len(message))
        cipher = AES.new(key, AES.MODE_CBC)
        iv = cipher.iv
        ciphertext = cipher.encrypt(pad(message, AES.block_size))
        return iv, ciphertext

    def create_encrypted_segment(self, seed: int, iv: bytes, ciphertext: bytes, entropy: float) -> bytes:
        """Create an encrypted segment"""
        segment = struct.pack('>Q', seed)  # Seed as unsigned long long
        segment += struct.pack('>d', entropy)  # Entropy as double
        segment += struct.pack('>I', len(iv)) + iv  # IV length and IV
        segment += struct.pack('>I', len(ciphertext)) + ciphertext  # Ciphertext length and ciphertext
        return segment

    def decrypt_with_seed(self, ciphertext: bytes, seed: int, iv: bytes) -> bytes:
        """Decrypt message using seed"""
        key = self.generate_adaptive_key(seed, len(ciphertext))
        cipher = AES.new(key, AES.MODE_CBC, iv)
        return unpad(cipher.decrypt(ciphertext), AES.block_size)

    def calculate_entropy(self, bits: np.ndarray) -> float:
        """Calculate entropy of bit sequence"""
        _, counts = np.unique(bits, return_counts=True)
        probabilities = counts / len(bits)
        entropy = -np.sum(np.fromiter((p * np.log2(p) for p in probabilities if p > 0), dtype=float))
        return entropy / np.log2(2)

    def enhance_custom_seed(self, message: bytes, custom_seed: int) -> Tuple[int, float]:
        """Enhance custom seed to improve entropy while maintaining user preference"""
        # Calculate initial entropy
        iv, ciphertext = self.encrypt_with_seed(message, custom_seed)
        initial_bits = np.unpackbits(np.frombuffer(ciphertext, dtype=np.uint8))
        initial_entropy = self.calculate_entropy(initial_bits)
        
        if initial_entropy >= 0.9999:
            return custom_seed, initial_entropy
            
        # Generate entropy-enhanced seed while preserving user's seed
        enhanced_seed = custom_seed
        max_attempts = 1000
        best_entropy = initial_entropy
        
        for i in range(max_attempts):
            # Create variation of custom seed
            test_seed = custom_seed + (i * 1000)
            iv, ciphertext = self.encrypt_with_seed(message, test_seed)
            bits = np.unpackbits(np.frombuffer(ciphertext, dtype=np.uint8))
            entropy = self.calculate_entropy(bits)
            
            if entropy > best_entropy:
                enhanced_seed = test_seed
                best_entropy = entropy
                
            if entropy >= 0.9999:
                break
                
        return enhanced_seed, best_entropy

    def process_message_with_custom_seed(self, message: bytes, custom_seed: Optional[int],
                                       keyword: Optional[str] = None) -> Tuple[bool, float, str]:
        """Process message with custom seed and keyword support"""
        if custom_seed is not None:
            # Calculate initial entropy
            iv, ciphertext = self.encrypt_with_seed(message, custom_seed)
            initial_bits = np.unpackbits(np.frombuffer(ciphertext, dtype=np.uint8))
            initial_entropy = self.calculate_entropy(initial_bits)

            print(f"Initial entropy with seed {custom_seed}: {initial_entropy}")

            if initial_entropy >= 0.9999:
                # Generate or use custom keyword
                if not keyword:
                    keyword = ''.join(random.choices(string.ascii_letters, k=8))

                # Create pattern key
                pattern_key = f"{custom_seed}_{keyword}_{hashlib.sha256(message).hexdigest()[:8]}"
                return True, initial_entropy, pattern_key

            # Try to enhance the entropy
            enhanced_seed = custom_seed
            best_entropy = initial_entropy

            for i in range(100):  # Try 100 variations
                test_seed = custom_seed + i
                iv, ciphertext = self.encrypt_with_seed(message, test_seed)
                bits = np.unpackbits(np.frombuffer(ciphertext, dtype=np.uint8))
                entropy = self.calculate_entropy(bits)

                print(f"Variation {i}: entropy = {entropy}")

                if entropy > best_entropy:
                    enhanced_seed = test_seed
                    best_entropy = entropy

                if entropy >= 0.9999:
                    break

            if best_entropy >= 0.9999:
                if not keyword:
                    keyword = ''.join(random.choices(string.ascii_letters, k=8))

                pattern_key = f"{enhanced_seed}_{keyword}_{hashlib.sha256(message).hexdigest()[:8]}"
                return True, best_entropy, pattern_key

        return False, 0.0, ""


    def find_perfect_entropy_seed(self, message: bytes, max_attempts: int = 100000) -> Tuple[Optional[int], Optional[bytes], Optional[bytes], Optional[float]]:
        """Search for a perfect entropy seed"""
        print("\nSearching for a perfect entropy seed...")
        
        for attempt in range(1, max_attempts + 1):
            seed = random.randint(1, 2**64 - 1)
            if seed in self.used_seeds:
                continue

            iv, ciphertext = self.encrypt_with_seed(message, seed)
            ciphertext_bits = np.unpackbits(np.frombuffer(ciphertext, dtype=np.uint8))
            entropy = self.calculate_entropy(ciphertext_bits)

            if abs(entropy - 1.0) < 1e-9:
                print(f"Found perfect seed: {seed} with entropy {entropy:.10f}")
                self.used_seeds.add(seed)
                return seed, iv, ciphertext, entropy

        print("Perfect entropy seed not found within 100,000 seeds.")
        return None, None, None, None

    def add_message(self, message: bytes, seed: Optional[int] = None, keyword: Optional[str] = None) -> Tuple[bool, float]:
        """Add a message with support for custom seeds and keywords"""
        # Validate message
        if not message or len(message) == 0:
            print("Error: Empty message not allowed")
            return False, 0.0
        
        # Use a hardcoded max length if config_manager is not available
        max_length = 1048576  # 1MB default
        if hasattr(self, 'config_manager'):
            max_length = self.config_manager.get_value("max_message_length", max_length)
        
        if len(message) > max_length:
            print("Error: Message exceeds maximum allowed length")
            return False, 0.0
    
        message_id = len(self.messages)
    
        if seed is not None:
            # Use custom seed processing
            success, entropy, pattern_key = self.process_message_with_custom_seed(message, seed, keyword)
            if not success:
                print(f"Provided seed {seed} does not achieve sufficient entropy.")
                return False, entropy
            
            # Extract the enhanced seed from pattern key
            enhanced_seed = int(pattern_key.split('_')[0])
        
            # Encrypt with enhanced seed
            iv, ciphertext = self.encrypt_with_seed(message, enhanced_seed)
            self.store_message(message, enhanced_seed, iv, ciphertext, entropy, message_id)
            return True, entropy
        
        else:
            # Use standard perfect entropy seed finding
            seed_result = self.find_perfect_entropy_seed(message)
            if seed_result:
                seed, iv, ciphertext, entropy = seed_result
                self.store_message(message, seed, iv, ciphertext, entropy, message_id)
                return True, entropy
            
        return False, 0.0

    def store_message(self, message: bytes, seed: int, iv: bytes, ciphertext: bytes, entropy: float, message_id: int, original_message: Optional[bytes] = None) -> None:
        """Store message data and create timeline marker"""
        self.messages.append(message)
        self.perfect_seeds.append(seed)
        self.encryption_data.append((iv, ciphertext, entropy))
        self.timeline.create_marker(seed, message_id, message, entropy)
        self.entropy_history.append(entropy)

        # Store the original message if provided
        if original_message is not None:
            if not hasattr(self, 'original_messages'):
                self.original_messages = []
            self.original_messages.append(original_message)

    def extract_message(self, combined_data: bytes, seed: int) -> Tuple[Optional[bytes], Optional[int]]:
        """
        Extract message with enhanced error checking and binary format handling.
        """
        try:
            # Ensure combined_data is bytes
            if isinstance(combined_data, str):
                try:
                    combined_data = bytes.fromhex(combined_data)
                except ValueError:
                    print("Error: Failed to convert hex string to bytes")
                    return None, None

            data_length = len(combined_data)
            if data_length < 8:  # Minimum size for seed
                print("Error: Data too short for basic header")
                return None, None

            # Read message header
            current_pos = 0
            while current_pos + 8 <= data_length:
                # Read seed (8 bytes)
                msg_seed = struct.unpack('>Q', combined_data[current_pos:current_pos + 8])[0]
                current_pos += 8

                # Read entropy (8 bytes)
                if current_pos + 8 > data_length:
                    print("Error: Incomplete entropy field")
                    return None, None
                msg_entropy = struct.unpack('>d', combined_data[current_pos:current_pos + 8])[0]
                current_pos += 8

                # Read IV length (4 bytes)
                if current_pos + 4 > data_length:
                    print("Error: Incomplete IV length field")
                    return None, None
                iv_length = struct.unpack('>I', combined_data[current_pos:current_pos + 4])[0]
                current_pos += 4

                # Read IV
                if current_pos + iv_length > data_length:
                    print("Error: Incomplete IV field")
                    return None, None
                iv = combined_data[current_pos:current_pos + iv_length]
                current_pos += iv_length

                # Read ciphertext length (4 bytes)
                if current_pos + 4 > data_length:
                    print("Error: Incomplete ciphertext length field")
                    return None, None
                ct_length = struct.unpack('>I', combined_data[current_pos:current_pos + 4])[0]
                current_pos += 4

                # Read ciphertext
                if current_pos + ct_length > data_length:
                    print("Error: Incomplete ciphertext field")
                    return None, None
                ciphertext = combined_data[current_pos:current_pos + ct_length]
                current_pos += ct_length

                # Check if this is the message we're looking for
                if msg_seed == seed:
                    # Decrypt the message
                    message = self.decrypt_with_seed(ciphertext, seed, iv)
                    return message, 0  # Successfully extracted

                # If not the seed we're looking for, continue to the next message

            print(f"No message found for seed {seed}")
            return None, None

        except Exception as e:
            print(f"Error during extraction: {str(e)}")
            return None, None


    def combine_messages(self) -> bytes:
        """
        Combine all stored messages into a single byte stream.
        Returns combined data in bytes.
        """
        try:
            combined = bytearray()
        
            # Process each message in order
            for i in range(len(self.messages)):
                seed = self.perfect_seeds[i]
                iv, ciphertext, entropy = self.encryption_data[i]
            
                # Write seed (8 bytes)
                combined.extend(struct.pack('>Q', seed))
            
                # Write entropy (8 bytes, double)
                combined.extend(struct.pack('>d', entropy))
            
                # Write IV length (4 bytes) and IV
                combined.extend(struct.pack('>I', len(iv)))
                combined.extend(iv)
            
                # Write ciphertext length (4 bytes) and ciphertext
                combined.extend(struct.pack('>I', len(ciphertext)))
                combined.extend(ciphertext)
        
            return bytes(combined)
        
        except Exception as e:
            print(f"Error combining messages: {str(e)}")
            return bytes()



    def format_hash(self, combined_data: bytes) -> str:
        """Format hash for display"""
        return combined_data.hex()

    def verify_hash(self, hash_data: str) -> bool:
        """Verify hash integrity by recomputing from stored data"""
        try:
            combined_data = bytes.fromhex(hash_data)
            recomputed_hash = self.format_hash(self.combine_messages())
            return recomputed_hash == hash_data
        except Exception as e:
            print(f"Error verifying hash: {str(e)}")
            return False

    def monobit_test(self, data: np.ndarray) -> float:
        """Perform the Monobit Test on the binary data"""
        ones = np.count_nonzero(data)
        zeros = len(data) - ones
        total_bits = len(data)
        s = abs(ones - zeros) / np.sqrt(total_bits)
        p_value = erfc(s / np.sqrt(2))
        return p_value

    def runs_test(self, data: np.ndarray) -> float:
        """Perform the Runs Test on the binary data"""
        ones = np.count_nonzero(data)
        zeros = len(data) - ones
        pi = ones / len(data)

        if abs(pi - 0.5) >= (2 / np.sqrt(len(data))):
            return 0.0

        vobs = 1
        for i in range(1, len(data)):
            if data[i] != data[i - 1]:
                vobs += 1

        p_value = erfc(abs(vobs - (2 * len(data) * pi * (1 - pi))) /
                      (2 * np.sqrt(2 * len(data)) * pi * (1 - pi)))
        return p_value

    def chi_squared_test(self, data: np.ndarray) -> float:
        """Perform the Chi-Squared Test"""
        ones = np.count_nonzero(data)
        zeros = len(data) - ones
        expected = len(data) / 2
        chi_squared = ((zeros - expected) ** 2 + (ones - expected) ** 2) / expected
        p_value = 1 - chi2.cdf(chi_squared, df=1)
        return p_value

    def avalanche_test(self, message: bytes, seed: int) -> float:
        """Perform the Avalanche Test"""
        key = self.generate_adaptive_key(seed, len(message))
        cipher = AES.new(key, AES.MODE_CBC)
        iv = cipher.iv
        ciphertext1 = cipher.encrypt(pad(message, AES.block_size))

        flipped_message = bytearray(message)
        flipped_message[0] ^= 0x01
        cipher = AES.new(key, AES.MODE_CBC, iv)
        ciphertext2 = cipher.encrypt(pad(flipped_message, AES.block_size))

        bits1 = np.unpackbits(np.frombuffer(ciphertext1, dtype=np.uint8))
        bits2 = np.unpackbits(np.frombuffer(ciphertext2, dtype=np.uint8))
        differing_bits = np.sum(bits1 != bits2)
        total_bits = len(bits1)
        
        return differing_bits / total_bits