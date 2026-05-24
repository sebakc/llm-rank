"""Detect host RAM/VRAM and check model fit."""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
from dataclasses import dataclass

import psutil

from .models import Quant

log = logging.getLogger(__name__)

# Bytes per parameter at a given quantization.
QUANT_BYTES: dict[Quant, float] = {
    "fp16": 2.0,
    "q8": 1.0,
    "q5_k_m": 0.7,
    "q4_k_m": 0.55,
}
OVERHEAD_GB = 1.5  # KV cache + runtime slop


@dataclass
class HardwareInfo:
    ram_gb: float
    vram_gb: float | None
    gpu_name: str | None


def _detect_vram_via_torch() -> tuple[float | None, str | None]:
    try:
        import torch  # type: ignore
    except ImportError:
        return None, None
    if not torch.cuda.is_available():
        return None, None
    props = torch.cuda.get_device_properties(0)
    return props.total_memory / 1024**3, props.name


def _detect_vram_via_nvidia_smi() -> tuple[float | None, str | None]:
    if shutil.which("nvidia-smi") is None:
        return None, None
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            text=True,
            timeout=5,
        )
    except (subprocess.SubprocessError, OSError) as e:
        log.debug("nvidia-smi failed: %s", e)
        return None, None
    line = out.strip().splitlines()[0] if out.strip() else ""
    if not line:
        return None, None
    parts = [p.strip() for p in line.split(",")]
    name = parts[0] if parts else None
    mem_mib = None
    if len(parts) > 1:
        m = re.search(r"\d+", parts[1])
        if m:
            mem_mib = float(m.group(0))
    vram_gb = mem_mib / 1024 if mem_mib is not None else None
    return vram_gb, name


def detect_ram_gb() -> float:
    return psutil.virtual_memory().total / 1024**3


def detect() -> HardwareInfo:
    ram_gb = detect_ram_gb()
    vram, name = _detect_vram_via_torch()
    if vram is None:
        vram, name = _detect_vram_via_nvidia_smi()
    return HardwareInfo(ram_gb=ram_gb, vram_gb=vram, gpu_name=name)


def required_gb(params_b: float, quant: Quant) -> float:
    bytes_per = QUANT_BYTES.get(quant, 0.7)
    return params_b * bytes_per + OVERHEAD_GB


def fits(params_b: float, quant: Quant, available_gb: float) -> bool:
    return required_gb(params_b, quant) <= available_gb
