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
        return list(map(float, data))

    def __add__(self, other):
        if not isinstance(other, MiniArray):
            raise TypeError("Unsupported operand type")
        if self.shape != other.shape:
            raise ValueError("Arrays must have the same shape")

        if self.device == "gpu":
            result = array_ops.add_gpu(self._data, other._data)
        else:
            result = array_ops.add_cpu(self._data, other._data)

        return MiniArray(result, device=self.device)

    def to(self, device):
        if device not in ["cpu", "gpu"]:
            raise ValueError("Invalid device")
        return MiniArray(self._data, device=device)
