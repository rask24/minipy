from minipy.utils import get_device
from minipy.backend import array_ops


class MiniArray:
    def __init__(self, data, device="auto"):
        self.device = get_device(device)
        self._data = self._validate_data(data)
        self.shape = (len(self._data),)

    def _validate_data(self, data):
        if not isinstance(data, (list, tuple)):
            raise TypeError("Data must be a list or tuple")
        if not all(isinstance(x, (int, float)) for x in data):
            raise ValueError("All elements must be numbers")
        return list(data)

    def __add__(self, other):
        if not isinstance(other, MiniArray):
            raise TypeError("Unsupported operand type")
        if self.shape != other.shape:
            raise ValueError("Arrays must have the same shape")
        result = array_ops.add(self._data, other._data)
        return MiniArray(result, device=self.device)
