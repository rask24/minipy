import pytest
from minipy import MiniArray


def test_array_creation():
    arr = MiniArray([1, 2, 3])
    assert arr.device == "cpu"


def test_device_validation():
    with pytest.raises(ValueError):
        MiniArray([1, 2, 3], device="cuda")
