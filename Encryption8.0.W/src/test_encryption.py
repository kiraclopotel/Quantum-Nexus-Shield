import unittest
from core.encryption import QuantumStackEncryption
from core.timeline import TimelineManager
from core.mathematics import MathematicalCore
import numpy as np
import time
import hashlib
from typing import List, Tuple, Dict

class EnhancedTestSuite(unittest.TestCase):
    def setUp(self):
        """Initialize test environment"""
        self.encryption = QuantumStackEncryption()
        self.timeline = TimelineManager()
        self.math_core = MathematicalCore()
        self.test_messages = [
            b"Test message 1",
            b"Another test message",
            b"Third test message with more content",
            b"Fourth message with different length"
        ]

    def test_key_generation(self):
        """Test key generation uniqueness and properties"""
        keys = set()
        seeds = range(1000)  # Test with 1000 different seeds
        
        for seed in seeds:
            key = self.encryption.generate_adaptive_key(seed, 32)
            # Verify key length
            self.assertEqual(len(key), 32, "Key length should be 32 bytes")
            # Verify uniqueness
            self.assertNotIn(key, keys, "Generated key should be unique")
            keys.add(key)
            
            # Verify key consistency
            key2 = self.encryption.generate_adaptive_key(seed, 32)
            self.assertEqual(key, key2, "Same seed should generate same key")

    def test_entropy_analysis(self):
        """Test entropy calculation and perfect seed finding"""
        for message in self.test_messages:
            # Test perfect entropy seed finding
            seed_result = self.encryption.find_perfect_entropy_seed(message)
            self.assertIsNotNone(seed_result, "Should find perfect entropy seed")
            
            if seed_result:
                seed, iv, ciphertext, entropy = seed_result
                # Verify entropy is within valid range
                self.assertGreaterEqual(entropy, 0.9999, "Entropy should be >= 0.9999")
                self.assertLessEqual(entropy, 1.0, "Entropy should be <= 1.0")
                
                # Verify seed generates valid encryption
                key = self.encryption.generate_adaptive_key(seed, len(message))
                self.assertEqual(len(key), 32, "Generated key should be 32 bytes")

    def test_timeline_verification(self):
        """Test timeline management and verification"""
        messages = []
        seeds = []
        
        # Create timeline entries
        for i, message in enumerate(self.test_messages):
            seed_result = self.encryption.find_perfect_entropy_seed(message)
            if seed_result:
                seed, _, _, entropy = seed_result
                marker = self.timeline.create_marker(seed, i, message, entropy)
                messages.append(message)
                seeds.append(seed)
                
                # Verify marker properties
                self.assertIn('timestamp', marker)
                self.assertIn('layer', marker)
                self.assertIn('timeline', marker)
                self.assertIn('checksum', marker)
                
                # Verify marker integrity
                self.assertTrue(
                    self.timeline.verify_marker(seed, i, message),
                    f"Marker {i} verification failed"
                )

        # Test timeline continuity
        for i in range(1, len(seeds)):
            prev_marker = self.timeline.markers[seeds[i-1]]
            curr_marker = self.timeline.markers[seeds[i]]
            self.assertLessEqual(
                abs(prev_marker['layer'] - curr_marker['layer']), 1,
                "Layer difference should not exceed 1"
            )

    def test_layer_functions(self):
        """Test layer computation and transitions"""
        test_values = [
            (10, 2), (100, 3), (1000, 4), (10000, 5),
            (100000, 6), (1000000, 7)
        ]
        
        for value, expected_layer in test_values:
            computed_layer = self.math_core.compute_layer(value)
            self.assertEqual(
                computed_layer, expected_layer,
                f"Layer computation failed for value {value}"
            )
            
        # Test layer transitions
        for i in range(len(test_values) - 1):
            value1, layer1 = test_values[i]
            value2, layer2 = test_values[i + 1]
            transition = abs(layer2 - layer1)
            self.assertEqual(
                transition, 1,
                "Layer transitions should be continuous"
            )

    def test_encryption_decryption(self):
        """Test encryption and decryption functionality"""
        for message in self.test_messages:
            # Find perfect seed
            seed_result = self.encryption.find_perfect_entropy_seed(message)
            self.assertIsNotNone(seed_result, "Should find perfect entropy seed")
            
            if seed_result:
                seed, _, _, _ = seed_result
                # Test encryption
                iv, ciphertext = self.encryption.encrypt_with_seed(message, seed)
                self.assertNotEqual(ciphertext, message, "Ciphertext should differ from plaintext")
                
                # Test decryption
                decrypted = self.encryption.decrypt_with_seed(ciphertext, seed, iv)
                self.assertEqual(
                    decrypted, message,
                    "Decrypted message should match original"
                )
                
                # Test with wrong seed
                wrong_seed = seed + 1
                with self.assertRaises(Exception):
                    self.encryption.decrypt_with_seed(ciphertext, wrong_seed, iv)

    def test_performance_metrics(self):
        """Test performance and timing metrics"""
        encryption_times = []
        decryption_times = []
        
        for message in self.test_messages:
            # Measure encryption time
            start_time = time.time()
            seed_result = self.encryption.find_perfect_entropy_seed(message)
            if seed_result:
                seed, _, _, _ = seed_result
                iv, ciphertext = self.encryption.encrypt_with_seed(message, seed)
                encryption_time = time.time() - start_time
                encryption_times.append(encryption_time)
                
                # Measure decryption time
                start_time = time.time()
                self.encryption.decrypt_with_seed(ciphertext, seed, iv)
                decryption_time = time.time() - start_time
                decryption_times.append(decryption_time)
        
        # Verify performance metrics
        avg_encryption_time = np.mean(encryption_times)
        avg_decryption_time = np.mean(decryption_times)
        
        self.assertLess(avg_encryption_time, 1.0, "Average encryption time too high")
        self.assertLess(avg_decryption_time, 1.0, "Average decryption time too high")

    def test_randomness(self):
        """Test statistical properties of encrypted data"""
        test_message = b"Test message for randomness analysis"
        seed_result = self.encryption.find_perfect_entropy_seed(test_message)
        
        if seed_result:
            seed, _, ciphertext, _ = seed_result
            
            # Convert ciphertext to bits
            bits = np.unpackbits(np.frombuffer(ciphertext, dtype=np.uint8))
            
            # Test bit distribution
            ones = np.count_nonzero(bits)
            zeros = len(bits) - ones
            ratio = ones / len(bits)
            self.assertAlmostEqual(
                ratio, 0.5, delta=0.1,
                msg="Bit distribution should be approximately uniform"
            )
            
            # Test bit sequence randomness
            # Count runs of ones and zeros
            runs = 1
            for i in range(1, len(bits)):
                if bits[i] != bits[i-1]:
                    runs += 1
            
            # For truly random sequence, expect runs to be about half the length
            expected_runs = len(bits) / 2
            self.assertGreater(
                runs, expected_runs * 0.25,
                "Should have sufficient bit value transitions"
            )

    def test_hash_integrity(self):
        """Test hash generation and verification"""
        messages = []
        combined_data = None
        
        # Add messages and generate combined data
        for message in self.test_messages:
            success, entropy = self.encryption.add_message(message)
            self.assertTrue(success, "Message addition should succeed")
            messages.append(message)
        
        if self.encryption.messages:
            combined_data = self.encryption.combine_messages()
            self.assertIsNotNone(combined_data, "Should generate combined data")
            
            # Generate and verify hash
            hash_value = self.encryption.format_hash(combined_data)
            self.assertTrue(
                self.encryption.verify_hash(hash_value),
                "Hash verification should succeed"
            )
            
            # Test hash uniqueness
            different_hash = hashlib.sha256(b"different data").hexdigest()
            self.assertNotEqual(
                hash_value, different_hash,
                "Different data should produce different hashes"
            )

    def test_error_handling(self):
        """Test error handling and edge cases"""
        # Test with empty message
        result = self.encryption.add_message(b"")
        self.assertFalse(result[0], "Empty message should fail")
        
        # Test with very large message
        large_message = b"X" * (2**24)  # 16MB message
        result = self.encryption.add_message(large_message)
        self.assertFalse(result[0], "Oversized message should fail")
        
        # Test with invalid IV
        test_message = b"Test message"
        seed_result = self.encryption.find_perfect_entropy_seed(test_message)
        if seed_result:
            seed = seed_result[0]
            with self.assertRaises(ValueError):
                self.encryption.decrypt_with_seed(b"test", seed, b"")
        
        # Test timeline with invalid data
        with self.assertRaises(ValueError):
            self.timeline.create_marker(-1, 0, b"test", -1.0)
            
        # Test combine_messages with no data
        self.encryption.messages.clear()
        combined = self.encryption.combine_messages()
        self.assertEqual(combined, b"", "Empty message list should return empty bytes")

if __name__ == '__main__':
    unittest.main(verbosity=2)