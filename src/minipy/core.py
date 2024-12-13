from minipy.utils import get_device


class MiniArray:
    def __init__(self, data, device="auto"):
        self.device = get_device(device)
        self._data = data
