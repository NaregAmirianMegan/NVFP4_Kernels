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
    from submission_v5_static_overlap import custom_kernel

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

def main():
    import os
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    from utils import set_seed
    set_seed(42)

    _tests_correctness = [
                {"g": 8, "k": [7168, 7168, 7168, 7168, 7168, 7168, 7168, 7168], "m": [80, 176, 128, 72, 64, 248, 96, 160], "n": [4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096], "seed": 1111},
                # {"g": 2, "k": [256, 512], "m": [96, 128], "n": [128, 256], "seed": 1111},
                # {"g": 2, "k": [256, 256], "m": [256, 72], "n": [512, 384], "seed": 1111},
                # {"g": 2, "k": [512, 256], "m": [128, 128], "n": [128, 256], "seed": 1111},
                # {"g": 3, "k": [256, 512, 256], "m": [80, 128, 256], "n": [384, 256, 128], "seed": 1111},
                # {"g": 3, "k": [512, 512, 256], "m": [64, 72, 96], "n": [128, 384, 512], "seed": 1111},
                # {"g": 3, "k": [512, 256, 512], "m": [64, 256, 128], "n": [768, 128, 256], "seed": 1111},
                # {"g": 3, "k": [768, 256, 768], "m": [128, 128, 64], "n": [256, 512, 512], "seed": 1111},
                # {"g": 4, "k": [512, 256, 512, 256], "m": [128, 128, 128, 128], "n": [128, 128, 128, 128], "seed": 1111},
                # {"g": 4, "k": [256, 256, 256, 256], "m": [40, 56, 384, 512], "n": [512, 384, 256, 128], "seed": 1111},
                # {"g": 4, "k": [512, 768, 512, 768], "m": [512, 384, 256, 128], "n": [256, 256, 256, 256], "seed": 1111},
             ]

    tests_correctness = [TestCase(spec=str(case), args=case) for case in _tests_correctness]

    max_repeats = 100

    return run_testing(tests_correctness)


if __name__ == "__main__":
    sys.exit(main())
