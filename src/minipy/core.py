import numpy as np
from minipy.utils import get_device


class MiniArray:
    def __init__(self, data, device="auto"):
        self.device = get_device(device)
        self._data = np.asarray(data)
        self.shape = self._data.shape
        self.dtype = self._data.dtype

    def __repr__(self):
        return f"MiniArray({self._data!r}, device={self.device!r})"

    @staticmethod
    def _get_device(device):
        if device == "auto":
            return "cpu"
        if device not in ["cpu", "cuda"]:
            raise ValueError(f"Unsupported device: {device}")
        return device

    def __add__(self, other):
        if isinstance(other, MiniArray):
            return MiniArray(self._data + other._data, device=self.device)
        return MiniArray(self._data + other, device=self.device)

    def __mul__(self, other):
        if isinstance(other, MiniArray):
            return MiniArray(self._data * other._data, device=self.device)
        return MiniArray(self._data * other, device=self.device)
