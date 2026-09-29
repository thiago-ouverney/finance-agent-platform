#!/usr/bin/env python3
"""Fail-fast validation for the CUDA environment used by vLLM."""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import sysconfig
import tempfile
from datetime import datetime, timezone
from importlib.metadata import (
    PackageNotFoundError,
    distribution,
    version as package_version,
)
from typing import Any


VERSION_RE = re.compile(r"\d+")
CUDA_DIR_RE = re.compile(r"^cu(?P<major>\d{2,})$")
# CUDA Toolkit 13.0 release notes, table 3 (Linux x86_64 GA drivers):
# https://docs.nvidia.com/cuda/archive/13.0.0/cuda-toolkit-release-notes/
CUDA_12_MINIMUM_DRIVERS = {
    0: (525, 60, 13),
    1: (530, 30, 2),
    2: (535, 54, 3),
    3: (545, 23, 6),
    4: (550, 54, 14),
    5: (555, 42, 2),
    6: (560, 28, 3),
    8: (570, 26),
    9: (575, 51, 3),
}


def version_tuple(value: str) -> tuple[int, ...]:
    """Return the numeric components of a version string."""
    return tuple(int(part) for part in VERSION_RE.findall(value))


def cuda_version_from_directory(directory: str) -> str | None:
    """Convert NVIDIA package directories such as cu13 or cu129 to versions."""
    match = CUDA_DIR_RE.match(directory)
    if not match:
        return None
    digits = match.group("major")
    if len(digits) == 2:
        return f"{int(digits)}.0"
    return f"{int(digits[:2])}.{int(digits[2:])}"


def minimum_driver_for_cuda(cuda_version: str) -> tuple[int, ...]:
    """Return the minimum Linux driver family accepted for a CUDA major."""
    parsed = version_tuple(cuda_version)
    if not parsed:
        raise ValueError(f"versão CUDA inválida: {cuda_version!r}")
    major = parsed[0]
    if major >= 13:
        return (580, 65, 6)
    if major == 12:
        minor = parsed[1] if len(parsed) > 1 else 0
        if minor in CUDA_12_MINIMUM_DRIVERS:
            return CUDA_12_MINIMUM_DRIVERS[minor]
        return (525, 60, 13)
    if major == 11:
        return (450, 80, 2)
    raise ValueError(f"família CUDA não suportada pelo preflight: {cuda_version}")


def validate_driver_compatibility(driver: str, cuda_versions: list[str]) -> None:
    """Reject a driver older than any CUDA runtime visible to vLLM."""
    if not cuda_versions:
        raise RuntimeError(
            "não foi possível detectar a família do runtime CUDA do vLLM"
        )
    required = max(minimum_driver_for_cuda(value) for value in cuda_versions)
    current = version_tuple(driver)
    if current < required:
        families = ", ".join(sorted(set(cuda_versions)))
        required_text = ".".join(str(part) for part in required)
        raise RuntimeError(
            "driver NVIDIA incompatível com o runtime CUDA do vLLM: "
            f"driver={driver}, cuda={families}, mínimo={required_text}. "
            "Use uma imagem/runtime CUDA compatível ou atualize o driver do host."
        )


def discover_bundled_cuda_versions(
    explicit_runtime_lib: str = "",
) -> tuple[list[str], list[str]]:
    """Find CUDA runtimes bundled in the active Python environment."""
    versions: set[str] = set()
    libraries: set[str] = set()

    candidates: list[Path] = []
    if explicit_runtime_lib:
        explicit = Path(explicit_runtime_lib)
        if explicit.is_dir():
            candidates.extend(explicit.glob("libcudart.so.*"))
        else:
            candidates.append(explicit)

    purelib = Path(sysconfig.get_paths()["purelib"])
    candidates.extend(purelib.glob("nvidia/cu*/lib/libcudart.so.*"))

    for candidate in candidates:
        if not candidate.is_file():
            continue
        libraries.add(str(candidate.resolve()))
        for parent in candidate.parents:
            directory_version = cuda_version_from_directory(parent.name)
            if directory_version:
                versions.add(directory_version)
                break
        soname = re.search(r"libcudart\.so\.(\d+)", candidate.name)
        if soname:
            versions.add(f"{int(soname.group(1))}.0")

    return sorted(versions, key=version_tuple), sorted(libraries)


def query_gpu() -> tuple[str, str]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version",
            "--format=csv,noheader",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    rows = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not rows:
        raise RuntimeError("nvidia-smi não retornou nenhuma GPU")
    gpu, separator, driver = rows[0].rpartition(",")
    if not separator:
        raise RuntimeError(f"saída inesperada do nvidia-smi: {rows[0]!r}")
    return gpu.strip(), driver.strip()


def installed_version(name: str) -> str | None:
    try:
        return package_version(name)
    except PackageNotFoundError:
        return None


def wheel_metadata(name: str) -> dict[str, Any]:
    """Capture the installed wheel identity, including platform tags and origin."""
    try:
        installed = distribution(name)
    except PackageNotFoundError:
        return {}
    wheel_text = installed.read_text("WHEEL") or ""
    tags = [
        line.removeprefix("Tag:").strip()
        for line in wheel_text.splitlines()
        if line.startswith("Tag:")
    ]
    direct_url_text = installed.read_text("direct_url.json")
    direct_url = json.loads(direct_url_text) if direct_url_text else None
    return {
        "distribution": installed.metadata.get("Name", name),
        "version": installed.version,
        "wheel_tags": tags,
        "direct_url": direct_url,
    }


def run_uva_preflight(torch_module: Any) -> str:
    from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

    cpu_tensor = torch_module.empty(4096, dtype=torch_module.uint8, pin_memory=True)
    cuda_view = get_accelerator_view_from_cpu_tensor(cpu_tensor)
    torch_module.cuda.synchronize()
    return str(cuda_view.device)


def write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("managed", "existing"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vllm-package", default="")
    parser.add_argument("--vllm-wheel", default="")
    parser.add_argument("--torch-package", default="")
    parser.add_argument("--torch-backend", default="")
    parser.add_argument("--expected-vllm-version", default="")
    parser.add_argument("--expected-torch-version", default="")
    parser.add_argument("--expected-cuda-version", default="")
    parser.add_argument("--plugin-revision", default="")
    parser.add_argument(
        "--plugin-state", choices=("expected", "require"), default="expected"
    )
    parser.add_argument("--driver-only", action="store_true")
    parser.add_argument(
        "--cuda-runtime-lib", default=os.environ.get("VLLM_CUDA_RUNTIME_LIB", "")
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    gpu, driver = query_gpu()
    detected_cuda, runtime_libraries = discover_bundled_cuda_versions(
        args.cuda_runtime_lib
    )
    if args.mode == "managed" and args.expected_cuda_version:
        detected_cuda.append(args.expected_cuda_version)
    detected_cuda = sorted(set(detected_cuda), key=version_tuple)

    # Fail before loading vLLM native code when the CUDA family is already known.
    if detected_cuda:
        validate_driver_compatibility(driver, detected_cuda)
    if args.driver_only:
        if not detected_cuda:
            raise RuntimeError("--driver-only exige uma versão CUDA esperada ou detectada")
        print(
            "VLLM DRIVER PREFLIGHT OK: "
            f"cuda={','.join(detected_cuda)} driver={driver} gpu={gpu}"
        )
        return 0

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch não encontrou uma GPU CUDA disponível")
    if torch.version.cuda:
        detected_cuda = sorted(
            set([*detected_cuda, torch.version.cuda]), key=version_tuple
        )
    validate_driver_compatibility(driver, detected_cuda)

    # Only load vLLM native code after the CUDA/driver check succeeds.
    import vllm

    if args.mode == "managed":
        expected = {
            "vLLM": (args.expected_vllm_version, vllm.__version__),
            "PyTorch": (args.expected_torch_version, torch.__version__),
            "CUDA do PyTorch": (args.expected_cuda_version, torch.version.cuda or ""),
        }
        mismatches = [
            f"{name}: esperado={wanted}, observado={actual}"
            for name, (wanted, actual) in expected.items()
            if wanted and wanted != actual
        ]
        if mismatches:
            raise RuntimeError("ambiente vLLM fora do lock: " + "; ".join(mismatches))

    plugin_version = installed_version("vllm-gguf-plugin")
    if args.plugin_state == "require":
        importlib.import_module("vllm_gguf_plugin")
        if not plugin_version:
            raise RuntimeError(
                "vllm_gguf_plugin importa, mas a distribuição vllm-gguf-plugin "
                "não possui metadata instalada"
            )

    cuda_view = run_uva_preflight(torch)

    payload = {
        "schema_version": 1,
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "python_executable": sys.executable,
        "requested": {
            "vllm_package": args.vllm_package,
            "vllm_wheel": args.vllm_wheel,
            "torch_package": args.torch_package,
            "torch_backend": args.torch_backend,
            "vllm_version": args.expected_vllm_version,
            "torch_version": args.expected_torch_version,
            "cuda_version": args.expected_cuda_version,
            "gguf_plugin_revision": args.plugin_revision,
        },
        "observed": {
            "vllm_version": vllm.__version__,
            "vllm_wheel": wheel_metadata("vllm"),
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "cuda_runtime_families": detected_cuda,
            "cuda_runtime_libraries": runtime_libraries,
            "driver_version": driver,
            "gpu": gpu,
            "gguf_plugin_version": plugin_version,
            "uva_cuda_view": cuda_view,
        },
        "status": "ok",
    }
    write_manifest(args.output, payload)
    print(
        "VLLM PREFLIGHT OK: "
        f"mode={args.mode} vllm={vllm.__version__} torch={torch.__version__} "
        f"cuda={torch.version.cuda} driver={driver} gpu={gpu} uva={cuda_view}"
    )
    print(f"manifest={args.output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"VLLM PREFLIGHT FALHOU: {error}", file=sys.stderr)
        raise SystemExit(2) from error
