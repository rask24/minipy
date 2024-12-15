import random
import timeit
from dataclasses import dataclass
from statistics import mean, stdev
from typing import List, Callable, Dict, Tuple
from minipy import MiniArray


@dataclass
class BenchmarkConfig:
    sizes: List[int] = None
    devices: List[str] = None
    warmup_iterations: int = 5
    test_iterations: int = 100

    def __post_init__(self):
        self.sizes = self.sizes or [1000, 10000, 100000]
        self.devices = self.devices or ["cpu", "gpu"]


@dataclass
class StatResult:
    mean: float
    std: float
    min: float
    max: float

    def __str__(self):
        return f"mean={self.mean:.2f}ms ± {self.std:.2f}ms [min={self.min:.2f}ms, max={self.max:.2f}ms]"


def run_benchmark(func: Callable, warmup: int, iterations: int) -> StatResult:
    # Warmup phase
    for _ in range(warmup):
        func()

    # Measurement phase
    times = []
    for _ in range(iterations):
        start = timeit.default_timer()
        func()
        end = timeit.default_timer()
        times.append((end - start) * 1000)  # Convert to ms

    return StatResult(
        mean=mean(times),
        std=stdev(times) if len(times) > 1 else 0,
        min=min(times),
        max=max(times),
    )


def generate_test_data(size: int) -> List[float]:
    return [random.random() for _ in range(size)]


def warmup_gpu():
    warmup_size = 1000
    data = generate_test_data(warmup_size)
    arr = MiniArray(data, device="gpu")
    _ = arr + arr
    return


def benchmark_array_creation(data, device):
    return timeit.timeit(lambda: MiniArray(data, device=device), number=1) * 1000


def benchmark_computation(arr1, arr2, repeat=1):
    return timeit.timeit(lambda: arr1 + arr2, number=repeat) * 1000


def benchmark_dot(arr1, arr2, repeat=1):
    return timeit.timeit(lambda: arr1.dot(arr2), number=repeat) * 1000


def benchmark_operation(
    arr1: MiniArray, arr2: MiniArray, op: str, config: BenchmarkConfig
) -> StatResult:
    if op == "add":
        op_func = lambda: arr1 + arr2
    else:  # dot
        op_func = lambda: arr1.dot(arr2)

    return run_benchmark(op_func, config.warmup_iterations, config.test_iterations)


def run_detailed_benchmarks():
    config = BenchmarkConfig()
    results: Dict[Tuple[int, str], Dict[str, StatResult]] = {}

    print("\nDetailed Performance Benchmarks")
    print("=" * 80)

    # Warm up GPU
    warmup_data = generate_test_data(1000)
    warmup_arr = MiniArray(warmup_data, device="gpu")
    _ = warmup_arr + warmup_arr
    _ = warmup_arr.dot(warmup_arr)

    for size in config.sizes:
        print(f"\nArray Size: {size}")
        print("-" * 40)

        data1 = generate_test_data(size)
        data2 = generate_test_data(size)

        for device in config.devices:
            print(f"\nDevice: {device.upper()}")

            arr1 = MiniArray(data1, device=device)
            arr2 = MiniArray(data2, device=device)

            add_stats = benchmark_operation(arr1, arr2, "add", config)
            print(f"Addition:     {add_stats}")

            dot_stats = benchmark_operation(arr1, arr2, "dot", config)
            print(f"Dot Product:  {dot_stats}")

            results[(size, device)] = {"add": add_stats, "dot": dot_stats}


def run_comparative_benchmarks():
    print("\nComparative Benchmarks")
    print("=" * 80)

    sizes = [1000, 10000, 100000, 1000000, 10000000]
    print(f"{'Size':>10} | {'CPU (ms)':>12} | {'GPU (ms)':>12} | {'Speedup':>8}")
    print("-" * 50)

    for size in sizes:
        data1 = generate_test_data(size)
        data2 = generate_test_data(size)

        arr1_cpu = MiniArray(data1, device="cpu")
        arr2_cpu = MiniArray(data2, device="cpu")
        cpu_time = benchmark_computation(arr1_cpu, arr2_cpu)

        arr1_gpu = MiniArray(data1, device="gpu")
        arr2_gpu = MiniArray(data2, device="gpu")
        gpu_time = benchmark_computation(arr1_gpu, arr2_gpu)

        speedup = cpu_time / gpu_time if gpu_time > 0 else float("inf")

        print(f"{size:10d} | {cpu_time:12.2f} | {gpu_time:12.2f} | {speedup:8.2f}x")


if __name__ == "__main__":
    run_detailed_benchmarks()
    run_comparative_benchmarks()
