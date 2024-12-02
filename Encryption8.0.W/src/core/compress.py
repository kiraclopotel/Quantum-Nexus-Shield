# compress.py

from typing import Dict, List, Optional
import time
from collections import OrderedDict
from dataclasses import dataclass


@dataclass
class CompressionResult:
    data: bytes
    compression_ratio: float
    patterns: List[bytes]
    pattern_map: Dict[bytes, int]
    processing_time: float


class LRUCache:
    """LRU Cache using OrderedDict for accurate LRU eviction"""

    def __init__(self, maxsize: int = 5000):
        self.cache = OrderedDict()
        self.maxsize = maxsize
        self.hits = 0
        self.misses = 0

    def get(self, key: bytes) -> Optional[int]:
        if key in self.cache:
            self.cache.move_to_end(key)  # Mark as recently used
            self.hits += 1
            return self.cache[key]
        self.misses += 1
        return None

    def put(self, key: bytes, value: int):
        if key in self.cache:
            self.cache.move_to_end(key)
        else:
            if len(self.cache) >= self.maxsize:
                self.cache.popitem(last=False)  # Remove least recently used
            self.cache[key] = value


class FastPatternMatcher:
    """Optimized pattern matching"""

    def __init__(self):
        self.cache = LRUCache()
        self.min_pattern_size = 2
        self.max_pattern_size = 256

    def find_patterns(self, data: bytes) -> Dict[bytes, int]:
        patterns = {}
        data_view = memoryview(data)
        length = len(data)

        for size in range(self.min_pattern_size,
                          min(self.max_pattern_size, length // 2 + 1)):
            pos = 0
            while pos + size <= length:
                pattern = bytes(data_view[pos:pos + size])

                # Check cache first
                cached_count = self.cache.get(pattern)
                if cached_count is not None:
                    if cached_count > 1:
                        patterns[pattern] = cached_count
                    pos += 1
                    continue

                # Count occurrences
                count = 0
                next_pos = pos + size
                while True:
                    next_pos = data.find(pattern, next_pos)
                    if next_pos == -1:
                        break
                    count += 1
                    next_pos += size

                if count > 1:
                    patterns[pattern] = count
                    self.cache.put(pattern, count)

                pos += 1

        return patterns


class OptimizedCompressor:
    """Main compression system"""

    def __init__(self):
        self.pattern_matcher = FastPatternMatcher()
        self.patterns = []  # List to store patterns for reference
        self.pattern_to_id = {}  # Dict to map patterns to their IDs

    def compress(self, data: bytes) -> CompressionResult:
        start_time = time.perf_counter()
        current_data = data
        self.patterns = []  # Reset patterns list for each compression
        self.pattern_to_id = {}

        # Find patterns
        patterns = self.pattern_matcher.find_patterns(current_data)
        if patterns:
            # Sort patterns by length descending to prioritize longer patterns
            sorted_patterns = sorted(patterns.keys(), key=lambda x: -len(x))
            # Update patterns and mapping
            for pattern in sorted_patterns:
                if pattern not in self.pattern_to_id:
                    self.pattern_to_id[pattern] = len(self.patterns)
                    self.patterns.append(pattern)

            # Compress using patterns
            compressed = self._compress_with_patterns(current_data)
            compression_ratio = len(compressed) / len(data)
            processing_time = time.perf_counter() - start_time

            return CompressionResult(
                data=compressed,
                compression_ratio=compression_ratio,
                patterns=self.patterns.copy(),
                pattern_map=self.pattern_to_id.copy(),
                processing_time=processing_time
            )
        else:
            # No patterns found, return original data
            compression_ratio = 1.0
            processing_time = time.perf_counter() - start_time
            return CompressionResult(
                data=data,
                compression_ratio=compression_ratio,
                patterns=[],
                pattern_map={},
                processing_time=processing_time
            )

    def _compress_with_patterns(self, data: bytes) -> bytes:
        """Compress data using patterns with proper byte range handling"""
        compressed = bytearray()
        pos = 0
        length = len(data)

        while pos < length:
            # Find longest matching pattern
            match = None
            match_length = 0

            for pattern in self.patterns:
                pat_len = len(pattern)
                if (pat_len > match_length and
                        pos + pat_len <= length and
                        data[pos:pos + pat_len] == pattern):
                    match = pattern
                    match_length = pat_len

            if match:
                # Store pattern reference
                pattern_id = self.pattern_to_id[match]
                compressed.append(0xFF)  # Pattern marker
                compressed.append(len(match))
                compressed.append(pattern_id >> 8)  # High byte
                compressed.append(pattern_id & 0xFF)  # Low byte
                pos += len(match)
            else:
                # Store literal byte
                compressed.append(data[pos])
                pos += 1

        return bytes(compressed)

    def decompress(self, compressed: bytes) -> bytes:
        """Decompress data with enhanced error handling"""
        decompressed = bytearray()
        pos = 0
        length = len(compressed)

        while pos < length:
            if compressed[pos] == 0xFF:  # Pattern marker
                if pos + 3 >= length:
                    raise ValueError("Invalid compressed data format.")
                pattern_length = compressed[pos + 1]
                pattern_id = (compressed[pos + 2] << 8) | compressed[pos + 3]
                if pattern_id >= len(self.patterns):
                    raise ValueError("Invalid pattern ID in compressed data.")
                pattern = self.patterns[pattern_id]
                decompressed.extend(pattern)
                pos += 4  # Move position past the pattern marker
            else:
                decompressed.append(compressed[pos])
                pos += 1

        return bytes(decompressed)

    def verify_compression(self, original: bytes, compressed: bytes,
                           decompressed: bytes) -> bool:
        """Verify compression integrity"""
        return original == decompressed
