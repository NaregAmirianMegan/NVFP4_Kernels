import modal
import subprocess
from pathlib import Path

CURRENT_DIR = Path(__file__).parent
REMOTE_DIR = Path("/root/my_extension")

image = (
    modal.Image.from_registry("nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04", add_python="3.12")
    .entrypoint([])  # remove verbose logging by base image on entry
    .uv_pip_install("torch==2.9.1", index_url="https://download.pytorch.org/whl/cu130")
    .uv_pip_install("ninja")
    .uv_pip_install("nvidia-cutlass")
    .uv_pip_install("nvidia-cutlass-dsl")
    .add_local_dir(CURRENT_DIR, remote_path=REMOTE_DIR, ignore=["env"])
)
app = modal.App("sm100-nvfp4-debug", image=image)

@app.function(gpu="B200")
def run_memcheck():
    import sys
    sys.path.insert(0, "/my_extension")

    cmd = [
        "compute-sanitizer",
        "--tool", "memcheck",
        "--kernel-name", "kne=nvfp4_group_gemm_kernel",
        "python", "my_extension/eval_debug.py",
    ]

    res = subprocess.run(cmd, check=True)
