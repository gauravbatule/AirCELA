"""
AirCELA — Optimized AirLLM Alternative
=============================================

A high-performance LLM inference engine for consumer hardware.
The fastest way to run 7B-70B models on low-VRAM GPUs.

Developed by Gaurav Batule | https://github.com/gauravbatule/AirCELA
"""

__version__ = "1.0.0"
__author__ = "Gaurav Batule"
__email__ = "gauravbatule@gmail.com"
__license__ = "CELA Proprietary License"
__url__ = "https://github.com/gauravbatule/AirCELA"
__linkedin__ = "https://www.linkedin.com/in/gaurav-batule/"
__support__ = "https://buymeacoffee.com/gauravbatule"

# Lazy imports — torch takes 30-120s to import on some systems.
# Only import heavy modules when actually used.

def __getattr__(name):
    if name == "CELAEngine":
        from aircela.engine import CELAEngine
        return CELAEngine
    elif name == "DeviceManager":
        from aircela.device import DeviceManager
        return DeviceManager
    elif name == "GGUFReader":
        from aircela.gguf import GGUFReader
        return GGUFReader
    elif name == "Dequantizer":
        from aircela.quantize import Dequantizer
        return Dequantizer
    elif name == "CELATransformerLayer":
        from aircela.transformer import CELATransformerLayer
        return CELATransformerLayer
    elif name == "RMSNorm":
        from aircela.transformer import RMSNorm
        return RMSNorm
    elif name == "LayerPrefetcher":
        from aircela.prefetch import LayerPrefetcher
        return LayerPrefetcher
    raise AttributeError(f"module 'aircela' has no attribute {name}")

__all__ = [
    "CELAEngine",
    "DeviceManager",
    "GGUFReader",
    "Dequantizer",
    "CELATransformerLayer",
    "RMSNorm",
    "LayerPrefetcher",
]
