import numpy as np
from minipy import MiniArray


def test_array_creation():
    data = [1, 2, 3]
    arr = MiniArray(data)
    assert arr.device == "cpu"
    assert arr.shape == (3,)
    assert np.array_equal(arr._data, np.array(data))


def test_array_operations():
    a = MiniArray([1, 2, 3])
    b = MiniArray([4, 5, 6])

    c = a + b
    assert np.array_equal(c._data, np.array([5, 7, 9]))

    d = a * 2
    assert np.array_equal(d._data, np.array([2, 4, 6]))
