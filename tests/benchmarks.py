import random
import timeit

from minipy import MiniArray


def generate_test_data(size):
    return [random.random() for _ in range(size)]


def warmup_gpu():
    warmup_size = 1000
    data = generate_test_data(warmup_size)
    arr = MiniArray(data, device="gpu")
    _ = arr + arr
    return


def benchmark_array_creation(data, device):
    return timeit.timeit(lambda: MiniArray(data, device=device), number=1) * 1000


def benchmark_computation(arr1, arr2, repeat=1000):
    return timeit.timeit(lambda: arr1 + arr2, number=repeat) * 1000


def run_detailed_benchmarks():
    print("\nDetailed Performance Benchmarks")
    print("=" * 80)

    sizes = [1000, 10000, 100000]
    devices = ["cpu", "gpu"]

    print("Warming up GPU...")
    warmup_gpu()

    for size in sizes:
        print(f"\nArray Size: {size}")
        print("-" * 40)

        data1 = generate_test_data(size)
        data2 = generate_test_data(size)

        for device in devices:
            print(f"\nDevice: {device.upper()}")

            creation_time = benchmark_array_creation(data1, device)

            arr1 = MiniArray(data1, device=device)
            arr2 = MiniArray(data2, device=device)

            compute_time = benchmark_computation(arr1, arr2)

            print(f"Array Creation Time: {creation_time:.2f} ms")
            print(f"Computation Time (1000 iterations): {compute_time:.2f} ms")
            print(f"Average Time per Operation: {compute_time/1000:.2f} ms")

            result = arr1 + arr2
            expected = [a + b for a, b in zip(data1, data2)]
            first_elements = all(
                abs(r - e) < 1e-10 for r, e in zip(result._data[:5], expected[:5])
            )
            print(f"First 5 elements correct: {first_elements}")


def run_comparative_benchmarks():
    print("\nComparative Benchmarks")
    print("=" * 80)

    sizes = [1000, 10000, 100000, 1000000]
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
