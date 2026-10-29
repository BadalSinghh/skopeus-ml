"""Device detection and GPU management utilities.

Optimized for NVIDIA GeForce RTX 3050 Laptop GPU and general CUDA environments
with automatic fallback to CPU.
"""

from typing import Dict, Any, Optional
import torch


def get_device(requested_device: Optional[str] = None) -> torch.device:
    """Return torch.device based on availability and optional user request.

    Args:
        requested_device: Optional device string ('cuda', 'cpu', 'cuda:0', etc.).

    Returns:
        torch.device instance.
    """
    if requested_device is not None:
        req = requested_device.lower()
        if req.startswith("cuda") and not torch.cuda.is_available():
            print("WARNING: CUDA requested but not available. Falling back to CPU.")
            return torch.device("cpu")
        return torch.device(req)

    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_gpu_info() -> Dict[str, Any]:
    """Retrieve detailed GPU and CUDA diagnostics.

    Returns:
        Dictionary containing CUDA availability, device names, VRAM, and version info.
    """
    cuda_available = torch.cuda.is_available()
    info: Dict[str, Any] = {
        "cuda_available": cuda_available,
        "device_count": torch.cuda.device_count() if cuda_available else 0,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda if cuda_available else None,
        "cudnn_version": torch.backends.cudnn.version() if cuda_available and torch.backends.cudnn.is_available() else None,
        "gpu_name": None,
        "total_memory_gb": 0.0,
        "free_memory_gb": 0.0,
        "reserved_memory_gb": 0.0,
        "allocated_memory_gb": 0.0,
    }

    if cuda_available and torch.cuda.device_count() > 0:
        current_idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(current_idx)
        total_mem = props.total_memory / (1024 ** 3)
        reserved_mem = torch.cuda.memory_reserved(current_idx) / (1024 ** 3)
        allocated_mem = torch.cuda.memory_allocated(current_idx) / (1024 ** 3)

        info["gpu_name"] = props.name
        info["total_memory_gb"] = round(total_mem, 2)
        info["reserved_memory_gb"] = round(reserved_mem, 3)
        info["allocated_memory_gb"] = round(allocated_mem, 3)
        info["free_memory_gb"] = round(total_mem - reserved_mem, 3)

    return info


def print_device_info(requested_device: Optional[str] = None) -> torch.device:
    """Print standard GPU diagnostic banner and return selected torch.device.

    Args:
        requested_device: Optional user override string.

    Returns:
        Selected torch.device.
    """
    device = get_device(requested_device)
    info = get_gpu_info()

    print("=" * 60)
    print("DEVICE & ACCELERATION DIAGNOSTICS")
    print("=" * 60)
    print(f"Selected Device       : {device}")
    print(f"CUDA Available        : {info['cuda_available']}")
    print(f"PyTorch Version       : {info['torch_version']}")
    print(f"PyTorch CUDA Version  : {info['cuda_version']}")

    if info["cuda_available"]:
        print(f"GPU Name              : {info['gpu_name']}")
        print(f"CUDA Device Count     : {info['device_count']}")
        print(f"Total VRAM            : {info['total_memory_gb']} GB")
        print(f"Allocated VRAM        : {info['allocated_memory_gb']} GB")
        print(f"Reserved VRAM         : {info['reserved_memory_gb']} GB")
    else:
        print("GPU                   : None (Running on CPU)")
    print("=" * 60)

    return device


def get_memory_stats() -> Dict[str, float]:
    """Return current memory consumption for CUDA device in MB."""
    if not torch.cuda.is_available():
        return {"allocated_mb": 0.0, "reserved_mb": 0.0, "max_allocated_mb": 0.0}

    current = torch.cuda.current_device()
    return {
        "allocated_mb": round(torch.cuda.memory_allocated(current) / (1024 ** 2), 2),
        "reserved_mb": round(torch.cuda.memory_reserved(current) / (1024 ** 2), 2),
        "max_allocated_mb": round(torch.cuda.max_memory_allocated(current) / (1024 ** 2), 2),
    }
