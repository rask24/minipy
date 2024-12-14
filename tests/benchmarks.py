import timeit
import random
from minipy import MiniArray


def generate_test_data(size):
    return [random.random() for _ in range(size)]


def benchmark_python_add(data1, data2):
    return [a + b for a, b in zip(data1, data2)]


def benchmark_minipy_add(data1, data2):
    arr1 = MiniArray(data1)
    arr2 = MiniArray(data2)
    return (arr1 + arr2)._data


def run_benchmarks():
    sizes = [100, 1000, 10000, 100000]

    print("Benchmarking list addition operations")
    print("Size | Python (ms) | MiniPy (ms) | Speedup")
    print("-" * 45)

    for size in sizes:
        # データ生成
        data1 = generate_test_data(size)
        data2 = generate_test_data(size)

        # Pythonのリスト操作のベンチマーク
        python_time = timeit.timeit(
            lambda: benchmark_python_add(data1, data2), number=1000
        )

        # MiniPyのベンチマーク
        minipy_time = timeit.timeit(
            lambda: benchmark_minipy_add(data1, data2), number=1000
        )

        # ミリ秒に変換
        python_ms = python_time * 1000
        minipy_ms = minipy_time * 1000
        speedup = python_ms / minipy_ms

        print(f"{size:5d} | {python_ms:10.2f} | {minipy_ms:10.2f} | {speedup:7.2f}x")


if __name__ == "__main__":
    run_benchmarks()
