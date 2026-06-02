"""
metrics.py
----------
Lightweight resource profiling module utilizing psutil.
Compatible with Raspberry Pi and resource-constrained edge systems.
"""

import time
import threading
import psutil
from typing import Dict, Any

class ResourceProfiler:
    """High-frequency resource profiler using a background thread."""
    
    def __init__(self, interval: float = 0.1):
        self.interval = interval
        self._stop_event = threading.Event()
        self._thread = None
        self.cpu_samples = []
        self.memory_samples = [] # In Bytes
        self.start_time = 0.0
        self.end_time = 0.0
        self.process = psutil.Process()

    def _profile_loop(self):
        # Warmup CPU reading
        self.process.cpu_percent(interval=None)
        
        while not self._stop_event.is_set():
            try:
                # Get CPU percent of the current process and its children
                cpu = self.process.cpu_percent(interval=None)
                # Get memory usage in bytes (RSS)
                mem = self.process.memory_info().rss
                
                # Sample child processes if any exist (e.g., subprocess runs)
                for child in self.process.children(recursive=True):
                    try:
                        cpu += child.cpu_percent(interval=None)
                        mem += child.memory_info().rss
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                
                self.cpu_samples.append(cpu)
                self.memory_samples.append(mem)
            except Exception:
                pass
            time.sleep(self.interval)

    def start(self):
        """Start the background resource profiling."""
        self.cpu_samples.clear()
        self.memory_samples.clear()
        self._stop_event.clear()
        self.start_time = time.perf_counter()
        
        self._thread = threading.Thread(target=self._profile_loop, daemon=True)
        self._thread.start()

    def stop(self) -> Dict[str, Any]:
        """Stop profiling and return aggregated stats."""
        self.end_time = time.perf_counter()
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)
            
        elapsed_time = self.end_time - self.start_time
        
        # Calculate stats
        avg_cpu = sum(self.cpu_samples) / len(self.cpu_samples) if self.cpu_samples else 0.0
        peak_cpu = max(self.cpu_samples) if self.cpu_samples else 0.0
        
        # Convert bytes to MegaBytes (MB)
        mem_mb = [m / (1024 * 1024) for m in self.memory_samples]
        avg_mem = sum(mem_mb) / len(mem_mb) if mem_mb else 0.0
        peak_mem = max(mem_mb) if mem_mb else 0.0
        
        return {
            "elapsed_time_sec": round(elapsed_time, 3),
            "avg_cpu_percent": round(avg_cpu, 2),
            "peak_cpu_percent": round(peak_cpu, 2),
            "avg_memory_mb": round(avg_mem, 2),
            "peak_memory_mb": round(peak_mem, 2)
        }
