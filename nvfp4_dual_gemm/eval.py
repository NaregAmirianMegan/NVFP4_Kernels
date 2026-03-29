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
    from torch.cuda.nvtx import range as nvtx_range
    from utils import clear_l2_cache_large as clear_l2_cache
    from reference import check_implementation, generate_input
    from sub_ptx_v3 import custom_kernel

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

def run_testing(
    tests: list[TestCase]
):
    """
    Executes the actual test case code and checks for correctness.

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


def main():
    from utils import set_seed

    set_seed(42)

    _tests = [
                {"m": 256, "n": 4096, "k": 7168, "l": 1, "seed": 1111},
                {"m": 512, "n": 4096, "k": 7168, "l": 1, "seed": 1111},
                {"m": 256, "n": 3072, "k": 4096, "l": 1, "seed": 1111},
                {"m": 512, "n": 3072, "k": 7168, "l": 1, "seed": 1111}
             ]

    tests = [TestCase(spec=str(case), args=case) for case in _tests]

    return run_testing(tests)


if __name__ == "__main__":
    sys.exit(main())

