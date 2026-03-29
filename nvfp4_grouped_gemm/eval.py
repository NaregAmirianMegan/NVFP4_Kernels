import base64
import dataclasses
import multiprocessing
import re
import time
import os
import sys
import math
import torch

# Disable CuTe DSL file caching for more stable benchmarking
os.environ["CUTE_DSL_DISABLE_FILE_CACHING"] = "1"


def _init_worker():
    """Initialize worker process with correct env vars."""
    os.environ["CUTE_DSL_DISABLE_FILE_CACHING"] = "1"


from pathlib import Path
from typing import Any, Optional


try:
    from task import TestSpec
except ImportError:
    TestSpec = dict

import modal

CURRENT_DIR = Path(__file__).parent
REMOTE_DIR = Path("/my_extension")

image = (
    modal.Image.from_registry("nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04", add_python="3.12")
    .entrypoint([])  # remove verbose logging by base image on entry
    .uv_pip_install("torch==2.9.1", index_url="https://download.pytorch.org/whl/cu130")
    .uv_pip_install("ninja")
    .uv_pip_install("nvidia-cutlass")
    .uv_pip_install("nvidia-cutlass-dsl")
    .add_local_dir(CURRENT_DIR, remote_path=REMOTE_DIR, ignore=["env"])
)
app = modal.App("sm100-nvfp4-grouped-gemm", image=image)


NUM_ITERATIONS_PER_BENCHMARK = 15
# NUM_ITERATIONS_PER_BENCHMARK = 1



@dataclasses.dataclass
class TestCase:
    args: dict
    spec: str


def _combine(a: int, b: int) -> int:
    # combine two integers into one:
    # we need this to generate a secret seed based on the test-level seed and
    # the global secret seed.
    # the test-level seeds are public knowledge, and typically relatively small numbers,
    # so we need to make sure they don't provide any useful info for the full seed.
    # This Cantor construction ensures that if the secret seed is a large number,
    # then so is the overall seed.
    return int(a + (a + b) * (a + b + 1) // 2)


@dataclasses.dataclass
class Stats:
    runs: int
    mean: float
    std: float
    err: float
    best: float
    worst: float


def calculate_stats(durations: list[int]):
    """
    Calculate statistical data from a list of durations.

    @param durations: A list of durations in nanoseconds.
    @return: A Stats object containing the number of runs, mean, standard deviation, error, best, and worst durations.
    """
    runs = len(durations)
    total = sum(durations)
    best = min(durations)
    worst = max(durations)

    avg = total / runs
    variance = sum(map(lambda x: (x - avg) ** 2, durations))
    std = math.sqrt(variance / (runs - 1))
    err = std / math.sqrt(runs)

    return Stats(
        runs=runs, mean=avg, std=std, err=err, best=float(best), worst=float(worst)
    )


def _clone_data(data):
    """
    Recursively goes through data and clones all tensors.
    """
    if isinstance(data, tuple):
        return tuple(_clone_data(x) for x in data)
    elif isinstance(data, list):
        return [_clone_data(x) for x in data]
    elif isinstance(data, dict):
        return {k: _clone_data(v) for k, v in data.items()}
    elif isinstance(data, torch.Tensor):
        return data.clone()
    else:
        return data


def _run_single_test(test: TestCase):
    """
    Runs a single test case. Do not call directly
    """
    import torch.cuda
    from cutlass.cute.nvgpu.common import OpError
    from torch.cuda.nvtx import range as nvtx_range
    from utils import clear_l2_cache_large as clear_l2_cache
    from reference import check_implementation, generate_input
    from v2 import custom_kernel

    data = generate_input(**test.args)
    torch.cuda.synchronize()
    try:
        submission_output = custom_kernel(_clone_data(data))

    except OpError as E:
        print(f"Encountered {E}", file=sys.stderr)
        return False, str(E)
    torch.cuda.synchronize()
    return check_implementation(data, submission_output)


def run_single_test(test: TestCase):
    """
    Runs a single test in another process.
    """
    return _run_single_test(test)

@app.function(gpu="B200")
def run_testing(
    tests: list[TestCase]
):
    """
    Executes the actual test case code and checks for correctness.

    @param logger: A PopcornOutput object used for logging test results.
    @param tests: A list of TestCase objects representing the test cases to be executed.
    @return: An integer representing the exit status: 0 if all tests pass, otherwise 112.
    """
    import sys
    sys.path.insert(0, "/my_extension")

    passed = True
    print("test-count", len(tests))
    for idx, test in enumerate(tests):
        print(f"test.{idx}.spec", test.spec)
        good, message = run_single_test(test)
        if not good:
            print(f"test.{idx}.status", "fail")
            print(f"test.{idx}.error", message)
            passed = False
        else:
            print(f"test.{idx}.status", "pass")
            if message:
                print(f"test.{idx}.message", message)

    if passed:
        print("check", "pass")
        return 0
    else:
        print("check", "fail")
        return 112


def _run_single_benchmark(
    test: TestCase, recheck: bool, max_repeats: int, max_time_ns: float
) -> Stats | Any:
    """
    Runs one benchmark. Do not call directly.
    """
    import torch.cuda
    from cutlass.cute.nvgpu.common import OpError
    from torch.cuda.nvtx import range as nvtx_range
    from utils import clear_l2_cache_large as clear_l2_cache
    from reference import check_implementation, generate_input
    from v2 import custom_kernel

    durations = []
    data_list = []
    # generate input data once

    for i in range(NUM_ITERATIONS_PER_BENCHMARK):
        if "seed" in test.args:
            test.args["seed"] += 42
        data = generate_input(**test.args)
        data_list.append(data)

    check_copy = _clone_data(data_list)

    # first, one obligatory correctness check
    outputs = []
    try:
        for data in data_list:
            output = custom_kernel(_clone_data(data))
            outputs.append(output)
    except OpError as E:
        return f"Encountered {E}"
    for reference_output, custom_output in zip(check_copy, outputs):
        good, message = check_implementation(reference_output, custom_output)
        if not good:
            return message

    # now, do multiple timing runs without further correctness testing
    # there is an upper bound of 200 runs, and a lower bound of 3 runs;
    # otherwise, we repeat until we either measure at least 10 full seconds,
    # or the relative error of the mean is below 1%.

    bm_start_time = time.perf_counter_ns()
    for i in range(max_repeats):
        torch.cuda.synchronize()

        outputs = []
        clear_l2_cache()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        for data in data_list:
            output = custom_kernel(data)
            outputs.append(output)
        end_event.record()
        torch.cuda.synchronize()
        duration = (
            start_event.elapsed_time(end_event) / NUM_ITERATIONS_PER_BENCHMARK
        ) * 1e6  # Convert ms to ns

        if recheck:
            for reference_output, custom_output in zip(check_copy, outputs):
                good, message = check_implementation(reference_output, custom_output)
            if not good:
                return message

        durations.append(duration)

        total_bm_duration = time.perf_counter_ns() - bm_start_time
        if (
            i > 1 and total_bm_duration > 1e8
        ):  # at least 2 runs, and at least 100 ms total time
            stats = calculate_stats(durations)
            # stop if either
            # a) relative error dips below 0.1%
            # b) we exceed the total time limit for benchmarking the kernel
            # c) we exceed 2 minutes of total wallclock time.
            if (
                stats.err / stats.mean < 0.001
                or stats.mean * stats.runs > max_time_ns
                or total_bm_duration > 120e9
            ):
                break

    return calculate_stats(durations)


def run_single_benchmark(
    test: TestCase,
    recheck: bool,
    max_repeats: int,
    max_time_ns: float,
):
    """
    For a particular test case, check correctness (if applicable) and grab runtime results.

    @param pool: Process on which the benchmark will be launched.
    @param test: TestCase object.
    @param recheck: Flag for whether to explicitly check functional correctness.
    @param max_repeats: Number of trials to repeat.
    @param max_time_ns: Timeout time in nanoseconds.
    @return: A Stats object for this particular benchmark case or an error if the test fails.
    """
    return _run_single_benchmark(test, recheck, max_repeats, max_time_ns)

@app.function(gpu="B200")
def run_benchmarking(
    tests: list[TestCase],
    max_repeats: int
):
    """
    Executes benchmarking code for a CUDA Kernel and logs runtimes.

    @param logger: A PopcornOutput object used for logging benchmark results.
    @param pool: Process on which the benchmarks will be launched.
    @param tests: A list of TestCase objects representing the test cases to be benchmarked.
    @return: An integer representing the exit status: 0 if all benchmarks pass, otherwise 112.
    """
    import sys
    sys.path.insert(0, "/my_extension")

    run_single_benchmark(tests[0], False, max_repeats, 10e7)

    passed = True
    print("benchmark-count", len(tests))
    for idx, test in enumerate(tests):
        print(f"benchmark.{idx}.spec", test.spec)
        result = run_single_benchmark(test, False, max_repeats, 10e9)
        if isinstance(result, Stats):
            for field in dataclasses.fields(Stats):
                print(f"benchmark.{idx}.{field.name}", getattr(result, field.name))
        else:
            passed = False
            print(f"benchmark.{idx}.status", "fail")
            print(f"benchmark.{idx}.error", result)

    if passed:
        print("check", "pass")
        return 0
    else:
        print("check", "fail")
        return 112

@app.local_entrypoint()
def main(mode: str):
    from utils import set_seed
    set_seed(42)

    _tests_benchmark = [
                # {"g": 2, "k": [1536, 1536], "m": [80, 80], "n": [4096, 4096], "seed": 1111},
                # {"g": 1, "k": [1536], "m": [80], "n": [8192], "seed": 1111}
                # {"g": 8, "k": [1536, 1536, 1536, 1536, 1536, 1536, 1536, 1536], "m": [80, 176, 128, 72, 64, 248, 96, 160], "n": [4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096], "seed": 1111}
                # {"g": 1, "k": [7168], "m": [128], "n": [4096], "seed": 1111},
                # {"g": 2, "k": [7168, 7168], "m": [128, 128], "n": [4096, 4096], "seed": 1111},
                # {"g": 4, "k": [7168, 7168, 7168, 7168], "m": [128, 128, 128, 128], "n": [4096, 4096, 4096, 4096], "seed": 1111},
                # {"g": 5, "k": [7168, 7168, 7168, 7168, 7168], "m": [128, 128, 128, 128, 128], "n": [4096, 4096, 4096, 4096, 4096], "seed": 1111},
                # {"g": 8, "k": [7168, 7168, 7168, 7168, 7168, 7168, 7168, 7168], "m": [128, 128, 128, 128, 128, 128, 128, 128], "n": [4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096], "seed": 1111}
                # {"g": 2, "k": [8192, 8192], "m": [192, 320], "n": [4096, 4096], "seed": 1111},
                # {"g": 2, "k": [4096, 4096], "m": [192, 320], "n": [4096, 4096], "seed": 1111},
                # {"g": 3, "k": [2048, 2048, 2048], "m": [128, 256, 256], "n": [4096, 4096, 4096], "seed": 1111},
                # {"g": 2, "k": [2048, 2048], "m": [128, 512], "n": [4096, 4096], "seed": 1111},
                # {"g": 2, "k": [2048, 2048], "m": [128, 512], "n": [4096, 4096], "seed": 1111},
                # {"g": 3, "k": [2048, 2048, 2048], "m": [512, 512, 250], "n": [4096, 4096, 4096], "seed": 1111}
                # {"g": 8, "k": [7168, 7168, 7168, 7168, 7168, 7168, 7168, 7168], "m": [80, 176, 128, 72, 64, 248, 96, 160], "n": [4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096], "seed": 1111},
                # {"g": 2, "k": [2048, 2048], "m": [128, 256], "n": [9728, 9728], "seed": 1111}
                # {"g": 2, "k": [4096, 4096], "m": [192, 320], "n": [3072, 3072], "seed": 1111},
                # {"g": 2, "k": [2048, 2048], "m": [192, 320], "n": [3072, 3072], "seed": 1111},
                # {"g": 2, "k": [1024, 1024], "m": [192, 320], "n": [3072, 3072], "seed": 1111},
                # {"g": 2, "k": [512, 512], "m": [192, 320], "n": [3072, 3072], "seed": 1111}
                # {"g": 2, "k": [8192, 8192], "m": [192, 320], "n": [3072, 3072], "seed": 1111},
                # {"g": 2, "k": [4096, 4096], "m": [192, 320], "n": [3072, 3072], "seed": 1111},
                # {"g": 2, "k": [2048, 2048], "m": [192, 320], "n": [3072, 3072], "seed": 1111}
                # {"g": 8, "k": [2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048], "m": [40, 126, 128, 128, 128, 128, 128, 128], "n": [7168, 7168, 7168, 7168, 7168, 7168, 7168, 7168], "seed": 1111},
                # {"g": 2, "k": [2048, 2048], "m": [192, 320], "n": [3072, 3072], "seed": 1111},

                {"k": [7168, 7168, 7168, 7168, 7168, 7168, 7168, 7168], "m": [80, 176, 128, 72, 64, 248, 96, 160], "n": [4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096], "seed": 1111},
                {"g": 8, "k": [2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048], "m": [40, 76, 168, 72, 164, 148, 196, 160], "n": [7168, 7168, 7168, 7168, 7168, 7168, 7168, 7168], "seed": 1111},
                {"g": 2, "k": [4096, 4096], "m": [192, 320], "n": [3072, 3072], "seed": 1111},
                {"g": 2, "k": [1536, 1536], "m": [128, 384], "n": [4096, 4096], "seed": 1111}
             ]

    _tests_correctness = [
                {"g": 2, "k": [256, 512], "m": [96, 128], "n": [128, 256], "seed": 1111},
                {"g": 2, "k": [256, 256], "m": [256, 72], "n": [512, 384], "seed": 1111},
                {"g": 2, "k": [512, 256], "m": [128, 128], "n": [128, 256], "seed": 1111},
                {"g": 3, "k": [256, 512, 256], "m": [80, 128, 256], "n": [384, 256, 128], "seed": 1111},
                {"g": 3, "k": [512, 512, 256], "m": [64, 72, 96], "n": [128, 384, 512], "seed": 1111},
                {"g": 3, "k": [512, 256, 512], "m": [64, 256, 128], "n": [768, 128, 256], "seed": 1111},
                {"g": 3, "k": [768, 256, 768], "m": [128, 128, 64], "n": [256, 512, 512], "seed": 1111},
                {"g": 4, "k": [512, 256, 512, 256], "m": [128, 128, 128, 128], "n": [128, 128, 128, 128], "seed": 1111},
                {"g": 4, "k": [256, 256, 256, 256], "m": [40, 56, 384, 512], "n": [512, 384, 256, 128], "seed": 1111},
                {"g": 4, "k": [512, 768, 512, 768], "m": [512, 384, 256, 128], "n": [256, 256, 256, 256], "seed": 1111},
             ]

    tests_benchmark = [TestCase(spec=str(case), args=case) for case in _tests_benchmark]
    tests_correctness = [TestCase(spec=str(case), args=case) for case in _tests_correctness]

    max_repeats = 100

    if mode == "test":
        return run_testing.remote(tests_correctness)
    elif mode == "benchmark":
        return run_benchmarking.remote(tests_benchmark, max_repeats)
    else:
        return 2


if __name__ == "__main__":
    sys.exit(main())
