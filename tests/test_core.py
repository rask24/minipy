import pytest
from minipy import MiniArray


def test_array_validation():
    arr1 = MiniArray([1, 2, 3])
    assert arr1._data == [1, 2, 3]
    arr2 = MiniArray([1.0, 2.0, 3.0])
    assert arr2._data == [1.0, 2.0, 3.0]

    with pytest.raises(ValueError):
        MiniArray([1, "2", 3])

    with pytest.raises(TypeError):
        MiniArray("123")


def test_array_operations():
    a = MiniArray([1, 2, 3])
    b = MiniArray([4, 5, 6])
    c = a + b
    assert c._data == [5, 7, 9]
