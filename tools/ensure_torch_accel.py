import os
import platform
import subprocess
import sys


def _run(cmd: list[str]) -> int:
    print("[ensure_torch_accel] " + " ".join(cmd))
    return subprocess.call(cmd)


def _has_nvidia_smi() -> bool:
    try:
        r = subprocess.run(
            ["nvidia-smi", "-L"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return r.returncode == 0
    except Exception:
        return False


def main() -> int:
    # macOS 不支持 CUDA；Apple Silicon 走 MPS（torch 自己处理），这里直接跳过
    sysname = platform.system().lower()
    if sysname == "darwin":
        print("[ensure_torch_accel] macOS detected, skip CUDA torch install (use MPS if available).")
        return 0

    # 只在检测到 NVIDIA 环境时尝试安装 CUDA torch（避免在无 GPU 机器上误装）
    if not _has_nvidia_smi():
        print("[ensure_torch_accel] nvidia-smi not found; assume no NVIDIA CUDA environment. Skip.")
        return 0

    try:
        import torch  # type: ignore
    except Exception:
        print("[ensure_torch_accel] torch not installed yet; skip (requirements will install it).")
        return 0

    ver = getattr(torch, "__version__", "")
    cuda_ok = bool(getattr(torch, "cuda", None) and torch.cuda.is_available())
    print(f"[ensure_torch_accel] torch={ver}, cuda_available={cuda_ok}")

    if cuda_ok:
        print("[ensure_torch_accel] CUDA already available. Nothing to do.")
        return 0

    # 若已是 CPU 版 / CUDA 不可用：尝试安装 CUDA wheel。
    # 2026 推荐：优先尝试 cu126（PyTorch 官方近期推荐的 CUDA 12.x），失败则尝试 cu121。
    pip = [sys.executable, "-m", "pip"]
    candidates = [
        "https://download.pytorch.org/whl/cu126",
        "https://download.pytorch.org/whl/cu121",
    ]

    for idx_url in candidates:
        print(f"[ensure_torch_accel] Trying PyTorch CUDA wheel via index-url: {idx_url}")
        code = _run(pip + ["install", "--upgrade", "torch", "torchvision", "torchaudio", "--index-url", idx_url])
        if code != 0:
            print(f"[ensure_torch_accel] install failed for {idx_url} (code={code})")
            continue
        # Re-check
        try:
            import importlib

            importlib.invalidate_caches()
            import torch as torch2  # type: ignore

            cuda_ok2 = bool(getattr(torch2, "cuda", None) and torch2.cuda.is_available())
            print(f"[ensure_torch_accel] after install: torch={torch2.__version__}, cuda_available={cuda_ok2}")
            if cuda_ok2:
                print("[ensure_torch_accel] CUDA torch enabled successfully.")
                return 0
        except Exception as e:
            print(f"[ensure_torch_accel] re-check failed: {e}")

    print("[ensure_torch_accel] Could not enable CUDA torch. Keep CPU torch.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

