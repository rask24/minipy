def cuda_is_available():
    try:
        import cupy  # noqa

        return True
    except ImportError:
        return False


def get_device(device="auto"):
    if device == "auto":
        return "cpu"
    if device not in ["cpu", "cuda"]:
        raise ValueError(f"Unsupported device: {device}")
    return device
