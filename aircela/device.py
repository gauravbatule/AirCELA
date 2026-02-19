"""
CELA Device Manager — Auto-detect GPU, CPU, RAM, SSD.

Automatically detects hardware capabilities and plans
optimal layer placement for inference.

Developed by Gaurav Batule | 🤖 AI-assisted vibe code
"""

import os
import sys
import platform


def _detect_gpu():
    """Detect NVIDIA GPU and VRAM."""
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            return {
                "name": torch.cuda.get_device_name(0),
                "vram_bytes": props.total_memory,
                "vram_free_bytes": props.total_memory - torch.cuda.memory_allocated(0),
                "compute_capability": (props.major, props.minor),
                "is_ampere": props.major >= 8,
            }
    except ImportError:
        pass
    return None


class DeviceManager:
    """
    Auto-detect hardware and plan optimal layer placement.

    Usage:
        dm = DeviceManager()
        print(dm.summary())
        plan = dm.plan_layer_placement(n_layers=56, layer_size_bytes=209*1024**2)
    """

    def __init__(self):
        self.gpu_info = _detect_gpu()
        self.has_cuda = self.gpu_info is not None
        self.device = "cuda" if self.has_cuda else "cpu"

        # RAM
        try:
            import psutil
            mem = psutil.virtual_memory()
            self.ram_total = mem.total
            self.ram_free = mem.available
        except ImportError:
            self.ram_total = 0
            self.ram_free = 0

    # ------------------------------------------------------------------
    @property
    def vram_total(self) -> int:
        return self.gpu_info["vram_bytes"] if self.gpu_info else 0

    @property
    def vram_free(self) -> int:
        return self.gpu_info["vram_free_bytes"] if self.gpu_info else 0

    @property
    def gpu_name(self) -> str:
        return self.gpu_info["name"] if self.gpu_info else "None"

    @property
    def is_ampere(self) -> bool:
        return self.gpu_info.get("is_ampere", False) if self.gpu_info else False

    # ------------------------------------------------------------------
    def summary(self) -> str:
        parts = []
        if self.has_cuda:
            cc = self.gpu_info["compute_capability"]
            parts.append(
                f"GPU: {self.gpu_name} | "
                f"VRAM: {self.vram_free/1024**3:.1f}/{self.vram_total/1024**3:.1f} GB | "
                f"RAM: {self.ram_free/1024**3:.1f}/{self.ram_total/1024**3:.1f} GB | "
                f"SM {cc[0]}.{cc[1]}"
            )
        else:
            parts.append(
                f"CPU only | RAM: {self.ram_free/1024**3:.1f}/{self.ram_total/1024**3:.1f} GB"
            )
        return "\n".join(parts)

    # ------------------------------------------------------------------
    def plan_layer_placement(self, n_layers: int, layer_size_bytes: int) -> dict:
        """
        Decide how many layers go on GPU, CPU RAM, or disk.

        Returns dict with gpu_layers, cpu_layers, disk_layers, layer_size_mb.
        """
        layer_mb = layer_size_bytes / (1024 ** 2)

        # Reserve ~500 MB for KV cache + overhead
        usable_vram = max(0, self.vram_free - 500 * 1024 ** 2) if self.has_cuda else 0
        usable_ram = max(0, self.ram_free - 2 * 1024 ** 3)  # keep 2 GB for OS

        gpu_layers = min(n_layers, int(usable_vram / layer_size_bytes)) if layer_size_bytes > 0 else 0
        remaining = n_layers - gpu_layers
        cpu_layers = min(remaining, int(usable_ram / layer_size_bytes)) if layer_size_bytes > 0 else 0
        disk_layers = remaining - cpu_layers

        return {
            "gpu_layers": gpu_layers,
            "cpu_layers": cpu_layers,
            "disk_layers": disk_layers,
            "layer_size_mb": layer_mb,
        }
