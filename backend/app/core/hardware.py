"""Hardware capability detection for Quantiva Model Studio.

Inspects the local machine to determine:
- CPU model, physical/logical core counts
- System RAM (total and available)
- Disk free space in QUANTIVA_DATA_DIR
- GPU name and VRAM (via ``nvidia-smi`` subprocess with CREATE_NO_WINDOW
  so no black cmd window flashes on Windows, with ``torch.cuda`` fallback)

Returns a structured :class:`HardwareProfile` and :class:`ModelTierRecommendation`
based on VRAM / RAM thresholds defined in the prompt spec.
"""
from __future__ import annotations

import logging
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import psutil

logger = logging.getLogger(__name__)

# Suppress noisy startup messages from torch / transformers when only probing
_TORCH_AVAILABLE: bool | None = None

# Windows flag to suppress the console window for nvidia-smi
_CREATE_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class GPUInfo:
    """Details about a single detected GPU."""

    name: str
    vram_total_mb: int
    vram_free_mb: int | None = None

    @property
    def vram_total_gb(self) -> float:
        return round(self.vram_total_mb / 1024, 2)


@dataclass(frozen=True)
class HardwareProfile:
    """Snapshot of relevant hardware capabilities for model selection."""

    cpu_model: str
    physical_cores: int
    logical_cores: int
    ram_total_gb: float
    ram_available_gb: float
    disk_free_gb: float
    gpus: list[GPUInfo] = field(default_factory=list)
    detection_errors: list[str] = field(default_factory=list)

    @property
    def has_gpu(self) -> bool:
        return bool(self.gpus)

    @property
    def total_vram_mb(self) -> int:
        return sum(g.vram_total_mb for g in self.gpus)

    @property
    def total_vram_gb(self) -> float:
        return round(self.total_vram_mb / 1024, 2)

    def to_dict(self) -> dict[str, Any]:
        return {
            "cpu_model": self.cpu_model,
            "physical_cores": self.physical_cores,
            "logical_cores": self.logical_cores,
            "ram_total_gb": self.ram_total_gb,
            "ram_available_gb": self.ram_available_gb,
            "disk_free_gb": self.disk_free_gb,
            "gpus": [
                {
                    "name": g.name,
                    "vram_total_gb": g.vram_total_gb,
                    "vram_total_mb": g.vram_total_mb,
                    "vram_free_mb": g.vram_free_mb,
                }
                for g in self.gpus
            ],
            "has_gpu": self.has_gpu,
            "total_vram_gb": self.total_vram_gb,
            "detection_errors": self.detection_errors,
        }


@dataclass(frozen=True)
class ModelTierRecommendation:
    """Recommended local model tiers based on detected hardware resources.

    Tier definitions (from P4.1 spec):
      - ``ultra_light`` → < 8 GB RAM / No GPU  (or very limited VRAM < 4 GB)
      - ``balanced``    → 8–16 GB RAM / 4–6 GB VRAM
      - ``powerful``    → 16+ GB RAM / 8+ GB VRAM
    """

    tier: str  # "cloud_api" | "ultra_light" | "balanced" | "powerful"
    description: str
    llm_models: list[str]
    embedding_models: list[str]
    reranker_models: list[str]
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "tier": self.tier,
            "description": self.description,
            "llm_models": self.llm_models,
            "embedding_models": self.embedding_models,
            "reranker_models": self.reranker_models,
            "rationale": self.rationale,
        }


# ── Tier classification ───────────────────────────────────────────────────────

def _classify_tier(profile: HardwareProfile) -> ModelTierRecommendation:
    ram_gb = profile.ram_total_gb
    vram_mb = profile.total_vram_mb
    vram_gb = profile.total_vram_gb

    if ram_gb >= 16 and (vram_mb >= 8 * 1024 or (not profile.has_gpu and ram_gb >= 24)):
        # Powerful: 16+ GB RAM with 8+ GB VRAM (or 24+ GB RAM no-GPU)
        return ModelTierRecommendation(
            tier="powerful",
            description="High-performance local inference with large models and reranker",
            llm_models=["qwen2.5:7b", "qwen2.5:14b", "llama3.1:8b", "mistral:7b"],
            embedding_models=["intfloat/multilingual-e5-base", "BAAI/bge-base-en-v1.5"],
            reranker_models=["cross-encoder/ms-marco-MiniLM-L-6-v2"],
            rationale=(
                f"RAM: {ram_gb:.1f} GB, VRAM: {vram_gb:.1f} GB — "
                f"sufficient for 7–14B parameter quantised models with reranker."
            ),
        )

    if ram_gb >= 8 and (vram_mb >= 4 * 1024 or (not profile.has_gpu and ram_gb >= 12)):
        # Balanced: 8–16 GB RAM / 4–6 GB VRAM
        return ModelTierRecommendation(
            tier="balanced",
            description="Balanced local inference with mid-size quantised models",
            llm_models=["qwen2.5:3b", "qwen2.5:7b-q4_K_M", "llama3.2:3b"],
            embedding_models=["intfloat/multilingual-e5-small", "BAAI/bge-small-en-v1.5"],
            reranker_models=[],
            rationale=(
                f"RAM: {ram_gb:.1f} GB, VRAM: {vram_gb:.1f} GB — "
                f"suitable for 3–7B quantised models; reranker optional."
            ),
        )

    # Ultra-light / Cloud: < 8 GB RAM or < 4 GB VRAM
    if ram_gb >= 4:
        return ModelTierRecommendation(
            tier="ultra_light",
            description="Ultra-light local models or cloud API (limited local resources)",
            llm_models=["qwen2.5:1.5b", "llama3.2:1b", "phi3:mini"],
            embedding_models=["sentence-transformers/all-MiniLM-L6-v2"],
            reranker_models=[],
            rationale=(
                f"RAM: {ram_gb:.1f} GB, VRAM: {vram_gb:.1f} GB — "
                f"only ultra-light models feasible; consider Cloud API for best quality."
            ),
        )

    return ModelTierRecommendation(
        tier="cloud_api",
        description="Cloud API strongly recommended (insufficient local resources)",
        llm_models=[],
        embedding_models=["sentence-transformers/all-MiniLM-L6-v2"],
        reranker_models=[],
        rationale=(
            f"RAM: {ram_gb:.1f} GB — insufficient for meaningful local LLM inference. "
            f"Use Gemini, OpenAI, or Anthropic cloud APIs."
        ),
    )


# ── GPU detection helpers ─────────────────────────────────────────────────────

def _detect_gpus_nvidia_smi() -> list[GPUInfo]:
    """Query NVIDIA GPUs via ``nvidia-smi`` subprocess (no console window)."""
    smi_path = shutil.which("nvidia-smi")
    if not smi_path:
        return []
    try:
        result = subprocess.run(
            [
                smi_path,
                "--query-gpu=name,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=8,
            creationflags=_CREATE_NO_WINDOW,  # No black cmd window on Windows
        )
        if result.returncode != 0:
            return []
        gpus: list[GPUInfo] = []
        for line in result.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 2:
                continue
            name = parts[0]
            try:
                vram_total = int(parts[1])
            except (ValueError, IndexError):
                vram_total = 0
            vram_free: int | None = None
            if len(parts) >= 3:
                try:
                    vram_free = int(parts[2])
                except ValueError:
                    pass
            gpus.append(GPUInfo(name=name, vram_total_mb=vram_total, vram_free_mb=vram_free))
        return gpus
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return []


def _detect_gpus_torch() -> list[GPUInfo]:
    """Probe NVIDIA GPU info through PyTorch CUDA API (silent import attempt)."""
    try:
        import torch  # type: ignore[import-untyped]

        if not torch.cuda.is_available():
            return []
        gpus: list[GPUInfo] = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            vram_total_mb = props.total_memory // (1024 * 1024)
            gpus.append(GPUInfo(name=props.name, vram_total_mb=vram_total_mb))
        return gpus
    except Exception:
        return []


def _get_cpu_model() -> str:
    """Return a human-readable CPU model string."""
    if sys.platform == "win32":
        try:
            result = subprocess.run(
                ["wmic", "cpu", "get", "name", "/value"],
                capture_output=True,
                text=True,
                timeout=5,
                creationflags=_CREATE_NO_WINDOW,
            )
            for line in result.stdout.splitlines():
                if line.startswith("Name=") and len(line) > 5:
                    return line.split("=", 1)[1].strip()
        except Exception:
            pass
    return platform.processor() or platform.machine() or "Unknown CPU"


# ── HardwareInspector ─────────────────────────────────────────────────────────

class HardwareInspector:
    """Inspect local machine hardware to guide Model Studio recommendations.

    Usage::

        inspector = HardwareInspector()
        profile = inspector.detect()
        recommendation = inspector.recommend(profile)
        print(recommendation.tier)          # "balanced"
        print(recommendation.llm_models)    # ["qwen2.5:7b-q4_K_M", ...]
    """

    def __init__(self, data_dir: Path | str | None = None) -> None:
        if data_dir is not None:
            self._data_dir = Path(data_dir).expanduser().resolve()
        else:
            try:
                from app.config import settings
                self._data_dir = Path(settings.QUANTIVA_DATA_DIR).expanduser().resolve()
            except Exception:
                self._data_dir = Path.home() / "Quantiva"

    def detect(self) -> HardwareProfile:
        """Gather a full hardware snapshot."""
        errors: list[str] = []

        # CPU
        try:
            cpu_model = _get_cpu_model()
        except Exception as exc:
            cpu_model = "Unknown"
            errors.append(f"CPU detection failed: {exc}")

        physical = psutil.cpu_count(logical=False) or 1
        logical = psutil.cpu_count(logical=True) or physical

        # RAM
        try:
            vm = psutil.virtual_memory()
            ram_total_gb = round(vm.total / 1024**3, 2)
            ram_available_gb = round(vm.available / 1024**3, 2)
        except Exception as exc:
            ram_total_gb = 0.0
            ram_available_gb = 0.0
            errors.append(f"RAM detection failed: {exc}")

        # Disk
        try:
            self._data_dir.mkdir(parents=True, exist_ok=True)
            usage = shutil.disk_usage(str(self._data_dir))
            disk_free_gb = round(usage.free / 1024**3, 2)
        except Exception as exc:
            disk_free_gb = 0.0
            errors.append(f"Disk detection failed: {exc}")

        # GPU — try nvidia-smi first, fall back to torch
        gpus: list[GPUInfo] = []
        try:
            gpus = _detect_gpus_nvidia_smi()
            if not gpus:
                gpus = _detect_gpus_torch()
        except Exception as exc:
            errors.append(f"GPU detection failed: {exc}")

        return HardwareProfile(
            cpu_model=cpu_model,
            physical_cores=physical,
            logical_cores=logical,
            ram_total_gb=ram_total_gb,
            ram_available_gb=ram_available_gb,
            disk_free_gb=disk_free_gb,
            gpus=gpus,
            detection_errors=errors,
        )

    def recommend(self, profile: HardwareProfile | None = None) -> ModelTierRecommendation:
        """Return model tier recommendation, detecting hardware if *profile* is None."""
        if profile is None:
            profile = self.detect()
        return _classify_tier(profile)

    def full_report(self) -> dict[str, Any]:
        """Return both hardware profile and recommendation as a serialisable dict."""
        profile = self.detect()
        recommendation = self.recommend(profile)
        return {
            "hardware": profile.to_dict(),
            "recommendation": recommendation.to_dict(),
        }


__all__ = [
    "GPUInfo",
    "HardwareInspector",
    "HardwareProfile",
    "ModelTierRecommendation",
]
