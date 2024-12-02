from typing import Dict, List, Tuple, Optional
import time
import hashlib
import numpy as np
from .mathematics import MathematicalCore

class TimelineManager:
    def __init__(self):
        self.markers: Dict = {}
        self.checksum_history: List[str] = []
        self.mathematical_core = MathematicalCore()
        self.layer_history: List[int] = []
        self.timeline_entries: Dict[int, List[Tuple[int, int]]] = {}
        self.timeline_stats: Dict = {'total_entries': 0, 'max_depth': 0}
        self.previous_timeline = None
        self.last_timestamp = 0

    def create_marker(self, seed: int, message_id: int, message: bytes, entropy: float) -> Dict:
        """Create timeline marker with enhanced mathematical properties"""
        # Generate and validate timeline
        timeline = self.mathematical_core.generate_entry_timeline(seed)
        if self.previous_timeline:
            timeline = self._ensure_timeline_continuity(timeline)
        
        # Compute and validate layer
        layer = self.mathematical_core.compute_layer(seed)
        layer = self.mathematical_core._manage_layer_state(seed, layer)
        
        # Create timestamp with monotonic guarantee
        current_time = time.time()
        if current_time <= self.last_timestamp:
            current_time = self.last_timestamp + 0.001
        self.last_timestamp = current_time
        
        # Update statistics
        self._update_statistics(timeline)
        
        # Store timeline entry
        self.timeline_entries[seed] = timeline
        self.layer_history.append(layer)
        self.previous_timeline = timeline

        # Create marker
        marker = {
            'seed': seed,
            'id': message_id,
            'timestamp': current_time,
            'entropy': entropy,
            'layer': layer,
            'timeline': timeline,
            'depth': len(timeline),
            'checksum': self._generate_checksum(seed, message_id, message, timeline)
        }
        
        self.markers[seed] = marker
        return marker

    def _generate_checksum(self, seed: int, message_id: int, 
                         message: bytes, timeline: List[Tuple[int, int]]) -> str:
        """Generate enhanced checksum with timeline properties"""
        timeline_str = str(timeline)
        layer = self.mathematical_core.compute_layer(seed)
        layer_value = str(self.mathematical_core.layer_function(seed, layer))
        
        # Combine all components for checksum
        components = [
            str(seed).encode(),
            str(message_id).encode(),
            message,
            timeline_str.encode(),
            layer_value.encode(),
            str(time.time()).encode()
        ]
        
        checksum_data = b''.join(components)
        checksum = hashlib.sha256(checksum_data).hexdigest()
        self.checksum_history.append(checksum)
        return checksum

    def _ensure_timeline_continuity(self, new_timeline: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Ensure smooth transition between timelines"""
        if not self.previous_timeline:
            return new_timeline
            
        # Validate and adjust timeline connections
        last_depth = self.previous_timeline[-1][1]
        first_depth = new_timeline[0][1]
        
        if abs(last_depth - first_depth) > 1:
            # Adjust depth values for smooth transition
            depth_diff = last_depth - first_depth
            adjusted_timeline = [(digit, depth + depth_diff) 
                               for digit, depth in new_timeline]
            return adjusted_timeline
            
        return new_timeline

    def verify_marker(self, seed: int, message_id: int, message: bytes) -> bool:
        """Verify timeline marker integrity with enhanced validation"""
        if seed not in self.markers:
            return False
            
        marker = self.markers[seed]
        timeline = marker['timeline']
        
        # Verify timeline continuity
        if message_id > 0 and seed-1 in self.markers:
            prev_timeline = self.markers[seed-1]['timeline']
            if not self._verify_timeline_continuity(prev_timeline, timeline):
                return False
        
        # Verify checksum
        current_checksum = self._generate_checksum(seed, message_id, message, timeline)
        if marker['checksum'] != current_checksum:
            return False
        
        # Verify mathematical properties
        current_layer = self.mathematical_core.compute_layer(seed)
        stored_layer = marker['layer']
        if current_layer != stored_layer:
            return False
            
        return True

    def _verify_timeline_continuity(self, prev_timeline: List[Tuple[int, int]], 
                                  curr_timeline: List[Tuple[int, int]]) -> bool:
        """Verify continuity between two timelines"""
        if not prev_timeline or not curr_timeline:
            return True
            
        last_depth = prev_timeline[-1][1]
        first_depth = curr_timeline[0][1]
        return abs(last_depth - first_depth) <= 1

    def _update_statistics(self, timeline: List[Tuple[int, int]]) -> None:
        """Update timeline statistics"""
        self.timeline_stats['total_entries'] += 1
        depths = [depth for _, depth in timeline]
        max_depth = max(depths) if depths else 0
        self.timeline_stats['max_depth'] = max(
            self.timeline_stats['max_depth'], max_depth)

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

    def get_timeline_metrics(self, seed: int) -> Optional[Dict]:
        """Get detailed metrics for a specific timeline entry"""
        if seed not in self.markers:
            return None
            
        marker = self.markers[seed]
        timeline = marker['timeline']
        
        depths = [depth for _, depth in timeline]
        return {
            'max_depth': max(depths),
            'min_depth': min(depths),
            'avg_depth': np.mean(depths),
            'layer': marker['layer'],
            'timestamp': marker['timestamp'],
            'entropy': marker['entropy'],
            'timeline_length': len(timeline),
            'checksum_count': len(self.checksum_history)
        }

    def get_visualization_data(self) -> Dict:
        """Prepare data for visualization"""
        if not self.markers:
            return {
                'timestamps': [],
                'layers': [],
                'entropies': [],
                'depths': [],
                'seeds': []
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
            visualization_data['checksum_counts'].append(
                len([c for c in self.checksum_history 
                     if c == marker['checksum']]))
            
        return visualization_data

    def reset(self) -> None:
        """Reset timeline state"""
        self.markers.clear()
        self.checksum_history.clear()
        self.layer_history.clear()
        self.timeline_entries.clear()
        self.timeline_stats = {'total_entries': 0, 'max_depth': 0}
        self.previous_timeline = None
        self.last_timestamp = 0
        self.mathematical_core.clear_caches()