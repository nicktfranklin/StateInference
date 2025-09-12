import gc
import os
import time
import tracemalloc
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import psutil
import torch
from memory_profiler import profile
from stable_baselines3.common.callbacks import BaseCallback
from torch.utils.tensorboard import SummaryWriter


class MemoryLeakDetector:
    """A utility class to detect memory leaks in Python code."""

    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.snapshots: List[tracemalloc.Snapshot] = []

    def start_monitoring(self):
        """Start memory monitoring."""
        tracemalloc.start()
        self.snapshots.clear()
        # Take initial snapshot
        self.snapshots.append(tracemalloc.take_snapshot())

    def stop_monitoring(self):
        """Stop memory monitoring and cleanup."""
        tracemalloc.stop()
        gc.collect()

    def take_snapshot(self, name: str = "") -> None:
        """Take a memory snapshot with an optional name."""
        snapshot = tracemalloc.take_snapshot()
        self.snapshots.append(snapshot)

    def compare_snapshots(
        self, snapshot1_idx: int = -2, snapshot2_idx: int = -1
    ) -> List[str]:
        """Compare two snapshots and return differences."""
        if len(self.snapshots) < 2:
            return ["Not enough snapshots to compare"]

        snapshot1 = self.snapshots[snapshot1_idx]
        snapshot2 = self.snapshots[snapshot2_idx]

        stats = snapshot2.compare_to(snapshot1, "lineno")
        return [str(stat) for stat in stats[:10]]  # Return top 10 differences

    def get_current_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        return self.process.memory_info().rss / 1024 / 1024

    @staticmethod
    def monitor_function(func: Callable) -> Callable:
        """Decorator to monitor memory usage of a function."""

        @profile
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper

    def monitor_object_counts(self) -> Dict[str, int]:
        """Monitor counts of existing objects by type."""
        counts = defaultdict(int)
        for obj in gc.get_objects():
            counts[type(obj).__name__] += 1
        return dict(counts)

    def find_memory_leaks(
        self, func: Callable, iterations: int = 5, threshold_mb: float = 1.0
    ) -> List[str]:
        """
        Run a function multiple times and check for memory leaks.
        Returns a list of warnings if memory usage consistently increases.
        """
        self.start_monitoring()
        initial_memory = self.get_current_memory_usage()
        warnings = []

        memory_usage = []
        for i in range(iterations):
            func()
            gc.collect()
            current_memory = self.get_current_memory_usage()
            memory_usage.append(current_memory)

            # Take a snapshot every other iteration
            if i % 2 == 0:
                self.take_snapshot(f"Iteration {i}")

            time.sleep(0.1)  # Small delay to stabilize

        # Check for consistent memory growth
        if all(
            memory_usage[i] < memory_usage[i + 1] for i in range(len(memory_usage) - 1)
        ):
            total_increase = memory_usage[-1] - initial_memory
            if total_increase > threshold_mb:
                warnings.append(
                    f"Potential memory leak detected! Memory increased by {total_increase:.2f}MB"
                )
                warnings.extend(self.compare_snapshots())

        self.stop_monitoring()
        return warnings


# Example usage
def example_leak_detection():
    detector = MemoryLeakDetector()

    # Example of a function with a memory leak
    @detector.monitor_function
    def leaky_function():
        data = []
        for i in range(1000):
            data.append("*" * 1000)  # Creating large strings
        return data

    # Monitor the function
    warnings = detector.find_memory_leaks(
        leaky_function, iterations=5, threshold_mb=1.0
    )

    # Print results
    if warnings:
        print("Memory leak warnings:")
        for warning in warnings:
            print(f"- {warning}")
    else:
        print("No significant memory leaks detected")

    # Print object counts
    print("\nObject counts:")
    counts = detector.monitor_object_counts()
    for obj_type, count in sorted(counts.items())[:10]:  # Show top 10 object types
        print(f"{obj_type}: {count}")


class MemoryMonitorCallback(BaseCallback):
    """
    Callback to monitor CPU and GPU memory usage during training.
    Supports logging to both console and TensorBoard.
    """

    def __init__(
        self,
        log_dir: Optional[str] = None,
        log_freq: int = 1,
        verbose: bool = True,
        gpu_id: Optional[int] = None,
    ):
        """
        Args:
            log_dir: Directory for TensorBoard logs. If None, TensorBoard logging is disabled
            log_freq: Log memory usage every N batches
            verbose: If True, print memory stats to console
            gpu_id: Specific GPU to monitor. If None, monitors all available GPUs
        """
        super().__init__()
        self.log_freq = log_freq
        self.verbose = verbose
        self.gpu_id = gpu_id
        self.process = psutil.Process(os.getpid())

        # Initialize TensorBoard writer if log_dir is provided
        self.writer = SummaryWriter(log_dir) if log_dir else None

        # Storage for memory statistics
        self.cpu_mem_history: List[float] = []
        self.gpu_mem_history: Dict[int, List[float]] = {}
        self.batch_history: List[int] = []

        # Initialize GPU tracking
        if torch.cuda.is_available():
            if gpu_id is not None:
                self.gpu_ids = [gpu_id]
            else:
                self.gpu_ids = list(range(torch.cuda.device_count()))

            for gpu_id in self.gpu_ids:
                self.gpu_mem_history[gpu_id] = []
        else:
            self.gpu_ids = []

    def _get_gpu_memory_usage(self, gpu_id: int) -> float:
        """Get GPU memory usage in MB."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated(gpu_id) / 1024 / 1024
        return 0.0

    def _get_cpu_memory_usage(self) -> float:
        """Get CPU memory usage in MB."""
        return self.process.memory_info().rss / 1024 / 1024

    def _log_memory_usage(self, batch_idx: int, epoch: int):
        """Log memory usage to console and/or TensorBoard."""
        cpu_mem = self._get_cpu_memory_usage()
        self.cpu_mem_history.append(cpu_mem)

        # Log CPU memory
        if self.writer:
            self.writer.add_scalar("Memory/CPU_MB", cpu_mem, batch_idx)

        memory_info = [f"CPU Memory: {cpu_mem:.1f}MB"]

        # Log GPU memory
        for gpu_id in self.gpu_ids:
            gpu_mem = self._get_gpu_memory_usage(gpu_id)
            self.gpu_mem_history[gpu_id].append(gpu_mem)

            if self.writer:
                self.writer.add_scalar(f"Memory/GPU_{gpu_id}_MB", gpu_mem, batch_idx)

            memory_info.append(f"GPU{gpu_id} Memory: {gpu_mem:.1f}MB")

        if self.verbose:
            print(f"Epoch {epoch}, Batch {batch_idx}: " + " | ".join(memory_info))

    def update_locals(self, batch_idx: int, epoch: int):
        """Called at the end of each batch."""
        if batch_idx % self.log_freq == 0:
            self._log_memory_usage(batch_idx, epoch)
            self.batch_history.append(batch_idx)

    def _on_training_end(self, epoch: int):
        """Called at the end of each epoch."""
        # Calculate and log memory statistics for the epoch
        cpu_mean = np.mean(self.cpu_mem_history[-self.log_freq :])
        cpu_peak = np.max(self.cpu_mem_history[-self.log_freq :])

        if self.writer:
            self.writer.add_scalar("Memory/CPU_Mean_MB", cpu_mean, epoch)
            self.writer.add_scalar("Memory/CPU_Peak_MB", cpu_peak, epoch)

        stats = [
            f"Epoch {epoch} Summary:",
            f"CPU Memory - Mean: {cpu_mean:.1f}MB, Peak: {cpu_peak:.1f}MB",
        ]

        for gpu_id in self.gpu_ids:
            gpu_mean = np.mean(self.gpu_mem_history[gpu_id][-self.log_freq :])
            gpu_peak = np.max(self.gpu_mem_history[gpu_id][-self.log_freq :])

            if self.writer:
                self.writer.add_scalar(f"Memory/GPU_{gpu_id}_Mean_MB", gpu_mean, epoch)
                self.writer.add_scalar(f"Memory/GPU_{gpu_id}_Peak_MB", gpu_peak, epoch)

            stats.append(
                f"GPU{gpu_id} Memory - Mean: {gpu_mean:.1f}MB, Peak: {gpu_peak:.1f}MB"
            )

        if self.verbose:
            print("\n".join(stats))

    def close(self):
        """Clean up resources."""
        if self.writer:
            self.writer.close()


# Example usage with PyTorch training loop
def example_training_loop():
    # Initialize model, optimizer, data loader, etc.
    model = torch.nn.Linear(10, 2).cuda()
    optimizer = torch.optim.Adam(model.parameters())

    # Initialize memory monitor
    memory_monitor = MemoryMonitorCallback(
        log_dir="runs/experiment_1", log_freq=10, verbose=True
    )

    num_epochs = 3
    for epoch in range(num_epochs):
        for batch_idx in range(100):
            # Your training step here
            x = torch.randn(32, 10).cuda()
            y = model(x)
            loss = y.mean()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Call memory monitor
            memory_monitor.on_batch_end(batch_idx, epoch)

        # End of epoch logging
        memory_monitor.on_epoch_end(epoch)

    # Cleanup
    memory_monitor.close()


if __name__ == "__main__":
    example_leak_detection()
