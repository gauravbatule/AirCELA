"""
AirCELA — Optimized AirLLM Alternative
=============================================

A high-performance LLM inference engine for consumer hardware.
The fastest way to run 7B-70B models on low-VRAM GPUs.

Developed by Gaurav Batule | https://github.com/gauravbatule
"""

__version__ = "1.0.0"
__author__ = "Gaurav Batule"
__email__ = "gauravbatule@cela-engine.dev"
__license__ = "CELA Proprietary License"
__url__ = "https://github.com/gauravbatule/aircela"
__linkedin__ = "https://www.linkedin.com/in/gaurav-batule/"
__support__ = "https://buymeacoffee.com/gauravbatule"

from aircela.device import DeviceManager
from aircela.gguf import GGUFReader
from aircela.quantize import Dequantizer
from aircela.transformer import CELATransformerLayer, RMSNorm
from aircela.prefetch import LayerPrefetcher
from aircela.engine import CELAEngine

__all__ = [
    "CELAEngine",
    "DeviceManager",
    "GGUFReader",
    "Dequantizer",
    "CELATransformerLayer",
    "RMSNorm",
    "LayerPrefetcher",
]
