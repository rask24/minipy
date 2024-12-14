import pytest
from minipy import MiniArray


def test_array_creation():
    # Basic array creation test
    arr = MiniArray([1.0, 2.0, 3.0])
    assert arr.shape == (3,)
    assert arr._data == [1.0, 2.0, 3.0]


def test_array_validation():
    # Validation test
    MiniArray([1, 2, 3])
    MiniArray([1.0, 2.0, 3.0])

    # Validation errors
    with pytest.raises(ValueError):
        MiniArray([1, "2", 3])

    with pytest.raises(TypeError):
        MiniArray("123")


def test_array_addition():
    # Addition test
    a = MiniArray([1.0, 2.0, 3.0])
    b = MiniArray([4.0, 5.0, 6.0])
    c = a + b
    assert c._data == [5.0, 7.0, 9.0]


def test_addition_validation():
    # Validation tests for addition
    a = MiniArray([1.0, 2.0])
    b = MiniArray([1.0, 2.0, 3.0])

    with pytest.raises(ValueError):
        _ = a + b  # Not the same shape

    with pytest.raises(TypeError):
        _ = a + [1.0, 2.0]  # Unsupported operand type


def test_device_support():
    # Device support test
    a = MiniArray([1.0, 2.0, 3.0], device="cpu")
    assert a.device == "cpu"

    with pytest.raises(ValueError):
        MiniArray([1.0, 2.0], device="invalid")
