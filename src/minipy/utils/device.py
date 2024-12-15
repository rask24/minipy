def cuda_is_available():
    try:
        import cupy  # noqa

        return True
    except ImportError:
        return False


def get_device(device="auto"):
    if device == "auto":
        return "cpu"
    if device == "gpu":
        if not cuda_is_available():
            raise ValueError("GPU is not available")
        return "gpu"
    if device not in ["cpu", "gpu"]:
        raise ValueError(f"Unsupported device: {device}")
    return device
