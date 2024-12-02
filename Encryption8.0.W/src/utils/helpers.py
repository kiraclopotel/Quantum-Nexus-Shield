import numpy as np
from typing import List, Dict, Any, Union
import json
import time
from pathlib import Path

class EncryptionHelper:
    @staticmethod
    def format_timestamp(timestamp: float) -> str:
        """Format timestamp for display"""
        return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp))

    @staticmethod
    def bytes_to_hex(data: bytes) -> str:
        """Convert bytes to hexadecimal string"""
        return data.hex()

    @staticmethod
    def hex_to_bytes(hex_str: str) -> bytes:
        """Convert hexadecimal string to bytes"""
        return bytes.fromhex(hex_str)

class DataValidator:
    @staticmethod
    def validate_seed(seed: Union[int, str, None]) -> bool:
        """Validate seed value"""
        if seed is None:
            return True
        try:
            seed_int = int(seed)
            return 0 <= seed_int <= 2**64 - 1
        except ValueError:
            return False

    @staticmethod
    def validate_message(message: Union[str, bytes]) -> bool:
        """Validate message format"""
        if isinstance(message, str):
            return len(message.encode()) > 0
        return len(message) > 0

class ConfigManager:
    def __init__(self, config_path: str = "config.json"):
        self.config_path = Path(config_path)
        self.default_config = {
            "max_attempts": 100000,
            "entropy_threshold": 0.9999,
            "min_message_length": 1,
            "max_message_length": 1048576,  # 1MB
            "visualization_options": {
                "auto_refresh": True,
                "default_view": "combined",
                "plot_dpi": 100
            },
            "performance_limits": {
                "max_encryption_time": 1.0,
                "max_decryption_time": 1.0
            }
        }
        self.config = self.load_config()

    def load_config(self) -> Dict:
        """Load configuration from file or create default"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    loaded_config = json.load(f)
                    # Merge with defaults to ensure all required fields exist
                    return {**self.default_config, **loaded_config}
            return self.default_config.copy()
        except Exception as e:
            print(f"Error loading config: {e}")
            return self.default_config.copy()

    def save_config(self) -> bool:
        """Save current configuration to file"""
        try:
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=4)
            return True
        except Exception as e:
            print(f"Error saving config: {e}")
            return False

    def get_value(self, key: str, default: Any = None) -> Any:
        """Get configuration value with default fallback"""
        return self.config.get(key, default)

    def set_value(self, key: str, value: Any) -> None:
        """Set configuration value"""
        self.config[key] = value

class MetricsCollector:
    def __init__(self):
        self.metrics: Dict[str, List[float]] = {
            'encryption_times': [],
            'decryption_times': [],
            'entropy_values': [],
            'layer_counts': [],
            'timeline_depths': [],
            'message_sizes': []
        }
        self.start_time = time.time()

    def add_metric(self, category: str, value: float) -> None:
        """Add a metric value to the specified category"""
        if category in self.metrics:
            self.metrics[category].append(value)

    def get_statistics(self) -> Dict[str, Dict[str, float]]:
        """Calculate statistics for all metrics"""
        stats = {}
        for category, values in self.metrics.items():
            if values:
                stats[category] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(min(values)),
                    'max': float(max(values)),
                    'count': len(values),
                    'total_time': time.time() - self.start_time
                }
        return stats

    def export_to_file(self, path: Path) -> bool:
        """Export metrics to JSON file"""
        try:
            stats = self.get_statistics()
            with open(path, 'w') as f:
                json.dump(stats, f, indent=4)
            return True
        except Exception as e:
            print(f"Error exporting metrics: {e}")
            return False

    def clear_metrics(self) -> None:
        """Clear all collected metrics"""
        for category in self.metrics:
            self.metrics[category] = []
        self.start_time = time.time()

    def get_performance_summary(self) -> Dict[str, float]:
        """Get summary of performance metrics"""
        return {
            'avg_encryption_time': np.mean(self.metrics['encryption_times']) if self.metrics['encryption_times'] else 0,
            'avg_decryption_time': np.mean(self.metrics['decryption_times']) if self.metrics['decryption_times'] else 0,
            'avg_entropy': np.mean(self.metrics['entropy_values']) if self.metrics['entropy_values'] else 0,
            'total_messages': len(self.metrics['message_sizes']),
            'total_runtime': time.time() - self.start_time
        }