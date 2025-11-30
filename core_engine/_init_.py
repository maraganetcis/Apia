# core_engine/__init__.py
"""
Apia Core Engine
세계 최고 수준 프로그래밍 AI의 핵심 엔진
"""

from .apia_core import ApiaCore, ApiaManager, get_apia
from .model_loader import ApiaModelLoader, get_model_loader
from .code_generator import ApiaCodeGenerator, create_code_generator

__version__ = "1.0.0"
__author__ = "Apia Team"
__start_date__ = "2025-11-29"

__all__ = [
    "ApiaCore",
    "ApiaManager", 
    "ApiaModelLoader",
    "ApiaCodeGenerator",
    "get_apia",
    "get_model_loader",
    "create_code_generator"
]

print(f"🚀 Apia Core Engine v{__version__} initialized")
print(f"📅 Project Start: {__start_date__}")
