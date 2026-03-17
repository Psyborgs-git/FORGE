"""Hardware detection and advisory service."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

try:
    import pynvml

    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

import psutil

from config import settings


@dataclass
class HardwareProfile:
    """Hardware profile with detected capabilities."""

    # GPU info
    gpu_count: int
    gpu_names: list[str]
    gpu_vram_total_gb: list[float]
    gpu_vram_free_gb: list[float]
    cuda_available: bool
    cuda_version: str | None

    # CPU/RAM info
    cpu_count: int
    cpu_model: str
    ram_total_gb: float
    ram_available_gb: float

    # Timestamp
    detected_at: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "gpu_count": self.gpu_count,
            "gpu_names": self.gpu_names,
            "gpu_vram_total_gb": self.gpu_vram_total_gb,
            "gpu_vram_free_gb": self.gpu_vram_free_gb,
            "cuda_available": self.cuda_available,
            "cuda_version": self.cuda_version,
            "cpu_count": self.cpu_count,
            "cpu_model": self.cpu_model,
            "ram_total_gb": self.ram_total_gb,
            "ram_available_gb": self.ram_available_gb,
            "detected_at": self.detected_at,
        }


@dataclass
class FeasibleOps:
    """Operations feasible with current hardware."""

    # Training methods
    can_train_full_ft: bool
    can_train_lora: bool
    can_train_qlora: bool
    max_trainable_params_b: float  # Billions

    # Inference backends
    can_use_gpu_inference: bool
    can_use_vllm: bool
    recommended_inference: str  # 'ollama' | 'llamacpp' | 'vllm'

    # Warnings
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "can_train_full_ft": self.can_train_full_ft,
            "can_train_lora": self.can_train_lora,
            "can_train_qlora": self.can_train_qlora,
            "max_trainable_params_b": self.max_trainable_params_b,
            "can_use_gpu_inference": self.can_use_gpu_inference,
            "can_use_vllm": self.can_use_vllm,
            "recommended_inference": self.recommended_inference,
            "warnings": self.warnings,
        }


class HardwareAdvisor:
    """Hardware detection and advisory service."""

    def __init__(self) -> None:
        """Initialize hardware advisor."""
        self._cached_profile: HardwareProfile | None = None
        self._last_check: float = 0.0
        self._check_interval = settings.HARDWARE_CHECK_INTERVAL_SECONDS

        # Initialize NVML if available
        if NVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
            except Exception:
                pass

    def get_hardware_profile(self, force_refresh: bool = False) -> HardwareProfile:
        """Get current hardware profile with caching."""
        current_time = time.time()

        if (
            not force_refresh
            and self._cached_profile is not None
            and (current_time - self._last_check) < self._check_interval
        ):
            return self._cached_profile

        profile = self._detect_hardware()
        self._cached_profile = profile
        self._last_check = current_time

        return profile

    def _detect_hardware(self) -> HardwareProfile:
        """Detect hardware capabilities."""
        # GPU detection
        gpu_count = 0
        gpu_names: list[str] = []
        gpu_vram_total_gb: list[float] = []
        gpu_vram_free_gb: list[float] = []
        cuda_available = False
        cuda_version: str | None = None

        if NVML_AVAILABLE:
            try:
                gpu_count = pynvml.nvmlDeviceGetCount()
                for i in range(gpu_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    name = pynvml.nvmlDeviceGetName(handle)
                    if isinstance(name, bytes):
                        name = name.decode("utf-8")
                    gpu_names.append(name)

                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    gpu_vram_total_gb.append(mem_info.total / 1024**3)
                    gpu_vram_free_gb.append(mem_info.free / 1024**3)
            except Exception:
                pass

        if TORCH_AVAILABLE:
            cuda_available = torch.cuda.is_available()
            if cuda_available:
                try:
                    cuda_version = torch.version.cuda
                    if not gpu_count:  # Fallback to torch if NVML not available
                        gpu_count = torch.cuda.device_count()
                        for i in range(gpu_count):
                            gpu_names.append(torch.cuda.get_device_name(i))
                            props = torch.cuda.get_device_properties(i)
                            gpu_vram_total_gb.append(props.total_memory / 1024**3)
                            # Free memory requires context
                            gpu_vram_free_gb.append(props.total_memory / 1024**3)
                except Exception:
                    pass

        # CPU/RAM detection
        cpu_count = psutil.cpu_count(logical=False) or 0
        cpu_model = "Unknown"
        try:
            with open("/proc/cpuinfo", "r") as f:
                for line in f:
                    if "model name" in line:
                        cpu_model = line.split(":")[1].strip()
                        break
        except Exception:
            pass

        ram = psutil.virtual_memory()
        ram_total_gb = ram.total / 1024**3
        ram_available_gb = ram.available / 1024**3

        return HardwareProfile(
            gpu_count=gpu_count,
            gpu_names=gpu_names,
            gpu_vram_total_gb=gpu_vram_total_gb,
            gpu_vram_free_gb=gpu_vram_free_gb,
            cuda_available=cuda_available,
            cuda_version=cuda_version,
            cpu_count=cpu_count,
            cpu_model=cpu_model,
            ram_total_gb=ram_total_gb,
            ram_available_gb=ram_available_gb,
            detected_at=time.time(),
        )

    def get_feasible_operations(self, profile: HardwareProfile | None = None) -> FeasibleOps:
        """Determine what operations are feasible with current hardware."""
        if profile is None:
            profile = self.get_hardware_profile()

        warnings = []

        # Determine max VRAM available
        max_vram_gb = max(profile.gpu_vram_total_gb) if profile.gpu_vram_total_gb else 0

        # Training capabilities based on VRAM thresholds from README Section 11
        can_train_full_ft = False
        can_train_lora = False
        can_train_qlora = False
        max_trainable_params_b = 0.0

        if max_vram_gb < 4:
            warnings.append("GPU VRAM < 4GB: All training and GPU inference disabled")
            can_train_qlora = False
            max_trainable_params_b = 0
        elif max_vram_gb < 8:
            can_train_qlora = True
            max_trainable_params_b = 7.0
            warnings.append("GPU VRAM 4-8GB: Limited to QLoRA 3B-7B models")
        elif max_vram_gb < 16:
            can_train_lora = True
            can_train_qlora = True
            max_trainable_params_b = 13.0
            warnings.append("GPU VRAM 8-16GB: LoRA 7B-13B, QLoRA 30B supported")
        elif max_vram_gb < 24:
            can_train_lora = True
            can_train_qlora = True
            can_train_full_ft = True  # For smaller models only
            max_trainable_params_b = 34.0
            warnings.append("GPU VRAM 16-24GB: LoRA 13B-34B, QLoRA 70B, Full FT 7B")
        else:
            can_train_lora = True
            can_train_qlora = True
            can_train_full_ft = True
            max_trainable_params_b = 70.0

        # Inference capabilities
        can_use_gpu_inference = max_vram_gb >= 4
        can_use_vllm = max_vram_gb >= 8

        # Recommended inference backend
        if max_vram_gb >= 24:
            recommended_inference = "vllm"
        elif max_vram_gb >= 4:
            recommended_inference = "ollama"
        else:
            recommended_inference = "llamacpp"
            warnings.append("No GPU available: CPU-only inference via llama.cpp")

        return FeasibleOps(
            can_train_full_ft=can_train_full_ft,
            can_train_lora=can_train_lora,
            can_train_qlora=can_train_qlora,
            max_trainable_params_b=max_trainable_params_b,
            can_use_gpu_inference=can_use_gpu_inference,
            can_use_vllm=can_use_vllm,
            recommended_inference=recommended_inference,
            warnings=warnings,
        )

    def __del__(self) -> None:
        """Clean up NVML."""
        if NVML_AVAILABLE:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass


# Global instance
hardware_advisor = HardwareAdvisor()
