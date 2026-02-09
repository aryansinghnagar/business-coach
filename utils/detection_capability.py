"""
Detection Capability Module

Evaluates device specs and network speed to recommend whether to use
Azure Face API (cloud) or MediaPipe (local) for face/emotion detection.
Used for dynamic switching when detection_method is "auto".
"""

import os
import time
from typing import Optional, Tuple

try:
    import requests
except ImportError:
    requests = None

try:
    import psutil
except ImportError:
    psutil = None

import config

# Device tier: low = prefer MediaPipe, high = can use Azure
DEVICE_TIER_LOW = "low"
DEVICE_TIER_MEDIUM = "medium"
DEVICE_TIER_HIGH = "high"

# Default latency threshold (ms): above this, prefer MediaPipe over Azure
DEFAULT_AZURE_LATENCY_THRESHOLD_MS = 500.0

# Minimum CPU count to consider "high" tier
HIGH_TIER_CPU_COUNT = 4
# Minimum memory (GB) for "high" tier if psutil available
HIGH_TIER_MEMORY_GB = 8.0


def get_cpu_count() -> int:
    """Return number of CPUs (logical)."""
    try:
        return os.cpu_count() or 2
    except Exception:
        return 2


def get_memory_gb() -> Optional[float]:
    """Return total system memory in GB if psutil available, else None."""
    if psutil is None:
        return None
    try:
        return psutil.virtual_memory().total / (1024 ** 3)
    except Exception:
        return None


def get_device_tier() -> str:
    """
    Estimate device capability: low, medium, or high.
    Low/medium → prefer MediaPipe (local); high → can use Azure (cloud).
    """
    cpus = get_cpu_count()
    mem_gb = get_memory_gb()
    if cpus >= HIGH_TIER_CPU_COUNT and (mem_gb is None or mem_gb >= HIGH_TIER_MEMORY_GB):
        return DEVICE_TIER_HIGH
    if cpus >= 2 and (mem_gb is None or mem_gb >= 4.0):
        return DEVICE_TIER_MEDIUM
    return DEVICE_TIER_LOW


def get_azure_latency_ms(
    base_url: Optional[str] = None,
    timeout_sec: float = 5.0,
) -> Optional[float]:
    """
    Measure round-trip latency (ms) to the backend Azure config endpoint.
    Used to infer network speed to Azure Face API (same region as backend).
    Returns None if request fails or requests not available.
    """
    if requests is None:
        return None
    url = (base_url or getattr(config, "BACKEND_BASE_URL", None) or "http://localhost:5000").rstrip("/")
    config_url = url + "/config/face-detection"
    try:
        start = time.perf_counter()
        r = requests.get(config_url, timeout=timeout_sec)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        if r.status_code == 200:
            return round(elapsed_ms, 2)
        return None
    except Exception:
        return None


def recommend_detection_method(
    azure_available: bool = True,
    device_tier: Optional[str] = None,
    azure_latency_ms: Optional[float] = None,
    latency_threshold_ms: Optional[float] = None,
    prefer_local: bool = False,
    return_unified_when_viable: bool = True,
) -> str:
    """
    Recommend detection method: "mediapipe", "azure_face_api", or "unified" based on
    device tier and Azure/network latency.

    Rules:
    - If Azure not available or prefer_local → mediapipe.
    - If device is low tier → mediapipe (reduce cloud dependency on weak devices).
    - If Azure latency is above threshold → mediapipe (slow network).
    - When return_unified_when_viable is True and tier is medium/high with latency
      below threshold → "unified" (MediaPipe + Azure fusion). Otherwise → "azure_face_api".

    Returns:
        "mediapipe" | "unified" | "azure_face_api"
    """
    if not azure_available or prefer_local:
        return "mediapipe"
    tier = device_tier or get_device_tier()
    threshold = latency_threshold_ms if latency_threshold_ms is not None else getattr(
        config, "AZURE_LATENCY_THRESHOLD_MS", DEFAULT_AZURE_LATENCY_THRESHOLD_MS
    )
    if tier == DEVICE_TIER_LOW:
        return "mediapipe"
    if azure_latency_ms is not None and azure_latency_ms > threshold:
        return "mediapipe"
    if tier == DEVICE_TIER_HIGH:
        if return_unified_when_viable:
            return "unified"
        return "azure_face_api"
    # Medium: prefer MediaPipe unless latency is very low (favor stability)
    if azure_latency_ms is not None and azure_latency_ms > threshold * 0.6:
        return "mediapipe"
    if return_unified_when_viable:
        return "unified"
    return "azure_face_api"


def evaluate_capability(
    base_url: Optional[str] = None,
) -> Tuple[str, str, Optional[float], str]:
    """
    Evaluate current capability and return recommended method.

    Returns:
        (device_tier, recommended_method, azure_latency_ms, reason)
    """
    tier = get_device_tier()
    try:
        cfg = config.get_face_detection_config()
        azure_available = cfg.get("azureFaceApiAvailable", False)
    except Exception:
        azure_available = False
    latency_ms = get_azure_latency_ms(base_url=base_url) if requests else None
    threshold = getattr(config, "AZURE_LATENCY_THRESHOLD_MS", DEFAULT_AZURE_LATENCY_THRESHOLD_MS)
    method = recommend_detection_method(
        azure_available=bool(azure_available),
        device_tier=tier,
        azure_latency_ms=latency_ms,
        latency_threshold_ms=threshold,
        return_unified_when_viable=True,
    )
    if not azure_available:
        reason = "Azure Face API not configured or unavailable"
    elif tier == DEVICE_TIER_LOW:
        reason = "Device tier is low; using local MediaPipe"
    elif latency_ms is not None and latency_ms > threshold:
        reason = f"Azure latency {latency_ms:.0f} ms above threshold; using MediaPipe"
    elif method == "unified":
        reason = f"Device tier {tier}, latency {latency_ms} ms; using unified (MediaPipe + Azure)"
    else:
        reason = f"Device tier {tier}, latency {latency_ms} ms; using Azure Face API"
    return tier, method, latency_ms, reason
