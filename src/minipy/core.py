class MiniArray:
    def __init__(self, data, device="auto"):
        self.device = self_get_device(device)
        self._data = None
