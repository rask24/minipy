def cuda_is_available():
    try:
        import cupy  # noqa

        return True
    except ImportError:
        return False


def get_device(device="auto"):
    if device == "auto":
        return "cuda" if cuda_is_available() else "cpu"
    return device
