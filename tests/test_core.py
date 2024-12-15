import pytest

from minipy import MiniArray


def test_array_creation():
    arr = MiniArray([1.0, 2.0, 3.0])
    assert arr.shape == (3,)
    assert arr._data == [1.0, 2.0, 3.0]

    arr_cpu = MiniArray([1.0, 2.0, 3.0], device="cpu")
    assert arr_cpu.device == "cpu"

    arr_gpu = MiniArray([1.0, 2.0, 3.0], device="gpu")
    assert arr_gpu.device == "gpu"


def test_validation():
    MiniArray([1, 2, 3])
    MiniArray([1.0, 2.0, 3.0])
    MiniArray([1.0, 2.0, 3.0], device="cpu")
    MiniArray([1.0, 2.0, 3.0], device="gpu")

    with pytest.raises(ValueError):
        MiniArray([1, "2", 3])

    with pytest.raises(TypeError):
        MiniArray("123")

    with pytest.raises(ValueError):
        MiniArray([1.0, 2.0], device="invalid_device")


def test_cpu_add():
    a = MiniArray([1.0, 2.0, 3.0], device="cpu")
    b = MiniArray([4.0, 5.0, 6.0], device="cpu")
    c = a + b
    assert c.device == "cpu"
    assert c._data == [5.0, 7.0, 9.0]


def test_gpu_add():
    a = MiniArray([1.0, 2.0, 3.0], device="gpu")
    b = MiniArray([4.0, 5.0, 6.0], device="gpu")
    c = a + b
    assert c.device == "gpu"
    assert c._data == [5.0, 7.0, 9.0]


def test_device_transfer():
    a = MiniArray([1.0, 2.0, 3.0], device="cpu")
    a_gpu = a.to("gpu")
    assert a_gpu.device == "gpu"
    assert a_gpu._data == a._data

    a_cpu = a_gpu.to("cpu")
    assert a_cpu.device == "cpu"
    assert a_cpu._data == a._data


def test_add_validation():
    a = MiniArray([1.0, 2.0, 3.0])
    b = MiniArray([1.0, 2.0])

    with pytest.raises(ValueError):
        _ = a + b

    with pytest.raises(TypeError):
        _ = a + [1.0, 2.0, 3.0]


def test_cpu_dot():
    a = MiniArray([1.0, 2.0, 3.0], device="cpu")
    b = MiniArray([4.0, 5.0, 6.0], device="cpu")
    c = a.dot(b)
    assert c.device == "cpu"
    assert c._data == [32.0]  # 1*4 + 2*5 + 3*6 = 32


def test_gpu_dot():
    a = MiniArray([1.0, 2.0, 3.0], device="gpu")
    b = MiniArray([4.0, 5.0, 6.0], device="gpu")
    c = a.dot(b)
    assert c.device == "gpu"
    assert c._data == [32.0]


def test_dot_validation():
    a = MiniArray([1.0, 2.0, 3.0])
    b = MiniArray([1.0, 2.0])

    with pytest.raises(ValueError):
        _ = a.dot(b)

    with pytest.raises(TypeError):
        _ = a.dot([1.0, 2.0, 3.0])


def test_dot_device_transfer():
    a = MiniArray([1.0, 2.0, 3.0], device="cpu")
    b = MiniArray([4.0, 5.0, 6.0], device="gpu")

    # Transfer to same device before operation
    a_gpu = a.to("gpu")
    c = a_gpu.dot(b)
    assert c.device == "gpu"
    assert c._data == [32.0]
