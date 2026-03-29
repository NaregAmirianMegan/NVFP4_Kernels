import base64
import dataclasses
import multiprocessing
import re
import time
import os
import sys
import math
from pathlib import Path
from typing import Any, Optional

import torch

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
app = modal.App("sm100-nvfp4", image=image)

NUM_ITERATIONS_PER_BENCHMARK = 50


@dataclasses.dataclass
class TestCase:
    args: dict
    spec: str


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
    from reference import check_implementation, generate_input
    from submission_ptx import custom_kernel

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
    from reference import check_implementation, generate_input
    from utils import clear_l2_cache
    from submission_ptx import custom_kernel

    durations = []
    data_list = []
    # generate input data once

    for i in range(NUM_ITERATIONS_PER_BENCHMARK):
        if "seed" in test.args:
            test.args["seed"] += 42
        data = generate_input(**test.args)
        data_list.append(data)

    check_copy = _clone_data(data_list)

    #  first, one obligatory correctness check
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

    @param test: TestCase object.
    @param recheck: Flag for whether to explicitly check functional correctness.
    @param max_repeats: Number of trials to repeat.
    @param max_time_ns: Timeout time in nanoseconds.
    @return: A Stats object for this particular benchmark case or an error if the test fails.
    """
    return _run_single_benchmark(test, recheck, max_repeats, max_time_ns)


@app.function(gpu="B200")
def run_benchmarking(
    tests: list[TestCase]
):
    """
    Executes benchmarking code for a CUDA Kernel and logs runtimes.

    @param tests: A list of TestCase objects representing the test cases to be benchmarked.
    @return: An integer representing the exit status: 0 if all benchmarks pass, otherwise 112.
    """
    import sys
    sys.path.insert(0, "/my_extension")

    run_single_benchmark(tests[0], False, 200, 10e7)

    passed = True
    print("benchmark-count", len(tests))
    for idx, test in enumerate(tests):
        print(f"benchmark.{idx}.spec", test.spec)
        result = run_single_benchmark(test, False, 200, 10e9)
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


def _run_single_profile_torch(test: TestCase) -> str:
    """
    Profiles a single benchmark using the torch profiler.
    """
    import torch.cuda
    from torch.cuda.nvtx import range as nvtx_range
    from torch.profiler import profile, ProfilerActivity
    from reference import generate_input
    from submission import custom_kernel

    with nvtx_range("generate input"):
        data = generate_input(**test.args)
        torch.cuda.synchronize()

    cloned = _clone_data(data)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        with nvtx_range("custom_kernel"):
            submission_output = custom_kernel(cloned)
            torch.cuda.synchronize()

    return prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=20)


def _run_single_profile_ncu(test: TestCase) -> str:
    """
    Profiles a single benchmark using ncu. Note: this does not
    invoke NCU; instead, it is expected that eval is launched
    under NCU, and this function will rurnthe kernel excactly
    once in the 'custom_kernel' nvtx range.
    """
    import torch.cuda
    from torch.cuda.nvtx import range as nvtx_range
    from reference import generate_input
    from submission import custom_kernel

    with nvtx_range("generate input"):
        data = generate_input(**test.args)
        torch.cuda.synchronize()

    cloned = _clone_data(data)
    with nvtx_range("custom_kernel"):
        submission_output = custom_kernel(cloned)
        torch.cuda.synchronize()

    return ""


def _combine_traces(traces: list["EventList"]) -> "EventList":
    """
    Combine multiple event traces obtained from multiple (distributed) torch.profiler
    activities. This function simply aggregates the data as like `prof.key_averages()`,
    except over multiple traces. Most of this function is reimplemented
    from `torch.autograd.profiler_util.EventList.key_averages()`.
    """
    from torch.autograd.profiler_util import FunctionEventAvg, EventList
    from collections import defaultdict

    def get_key(event) -> tuple[str, ...]:
        return (
            str(event.key),
            str(event.node_id),
            str(event.device_type),
            str(event.is_legacy),
            str(event.is_user_annotation),
        )

    stats: dict[tuple[str, ...], FunctionEventAvg] = defaultdict(FunctionEventAvg)

    for events in traces:
        for event in events:
            stats[get_key(event)].add(event)

    avg_list = EventList(stats.values())
    for event in avg_list:
        event.stack = []
        event.input_shapes = ""
        event.overload_name = ""

    return avg_list


def run_single_profile(test: TestCase) -> str:
    """
    Runs a single profiling activity in another process.
    """
    if bool(os.getenv("POPCORN_NCU", "0")):
        return _run_single_profile_ncu(test)
    else:
        return _run_single_profile_torch(test)

@app.function(gpu="B200")
def run_profiling(
    tests: list[TestCase]
):
    import sys
    sys.path.insert(0, "/my_extension")

    print("benchmark-count", len(tests))
    for idx, test in enumerate(tests):
        print(f"benchmark.{idx}.spec", test.spec)
        report = run_single_profile(test)
        print(
            f"benchmark.{idx}.report",
            base64.b64encode(report.encode("utf-8"), b"+*").decode("utf-8"),
        )
    print("check", "pass")
    return 0


@app.function(gpu="B200")
def find_cutlass_headers():
    """Debug function to locate CUTLASS headers in the container."""
    import subprocess
    import sys

    print("Searching for cutlass headers...")

    # Search for command_line.h
    print("\n=== Searching for command_line.h ===")
    try:
        result = subprocess.run(
            ["find", "/usr/local/lib/python3.12/site-packages", "-name", "command_line.h", "-type", "f"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.stdout.strip():
            print("Found command_line.h at:")
            print(result.stdout)
        else:
            print("command_line.h not found in site-packages")
    except Exception as e:
        print(f"Error searching: {e}")

    # List directory structure
    print("\n=== Directory structure of cutlass_library/source/include/cutlass ===")
    try:
        result = subprocess.run(
            ["find", "/usr/local/lib/python3.12/site-packages/cutlass_library/source/include/cutlass", "-type", "d"],
            capture_output=True,
            text=True,
            timeout=10
        )
        print(result.stdout)
    except Exception as e:
        print(f"Error: {e}")

    return 0


@app.local_entrypoint()
def main(mode: str):
    from utils import set_seed

    set_seed(42)

    _tests = [
                {"m": 128, "n": 7168, "k": 16384, "l": 1, "seed": 1111},
                {"m": 128, "n": 4096, "k": 7168, "l": 1, "seed": 1111},
                {"m": 128, "n": 7168, "k": 2048, "l": 1, "seed": 1111}
             ]
    tests = [TestCase(spec=str(case), args=case) for case in _tests]

    if mode == "test":
        return run_testing.remote(tests)
    elif mode == "benchmark":
        return run_benchmarking.remote(tests)
    elif mode == "profile":
        return run_profiling.remote(tests)
    elif mode == "find-cutlass":
        return find_cutlass_headers.remote()
    else:
        return 2


if __name__ == "__main__":
    sys.exit(main())
