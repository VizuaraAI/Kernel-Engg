#!/usr/bin/env python3
"""
Generate Nsight Compute reports for every .cu file in this lecture directory.

Sources live in   <project>/lecture21/*.cu
Binaries live in  <project>/cmake-build-debug/lecture21-<stem>

For each kernel01.cu this runs, from inside cmake-build-debug:

    sudo env "PATH=$PATH" ncu -f -o kernel01.ncu-rep ./lecture21-kernel01

Usage:
    python generate_ncu_reports.py                      # sources = this script's dir
    python generate_ncu_reports.py --build-dir /path/to/cmake-build-debug
    python generate_ncu_reports.py --src-dir ../lecture20
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

DEFAULT_BUILD_DIR_NAME = "cmake-build-debug"


def find_build_dir(src_dir: Path) -> Path | None:
    """Walk up from src_dir looking for a cmake-build-debug directory."""
    for parent in [src_dir, *src_dir.parents]:
        candidate = parent / DEFAULT_BUILD_DIR_NAME
        if candidate.is_dir():
            return candidate
    return None


def profile(src_dir: Path, build_dir: Path, cu_file: Path) -> bool:
    stem = cu_file.stem                    # kernel01
    binary = f"{src_dir.name}-{stem}"      # lecture21-kernel01
    report = f"{stem}.ncu-rep"             # kernel01.ncu-rep

    if not (build_dir / binary).is_file():
        print(f"[skip] binary not found: {build_dir / binary} (build it first)")
        return False

    cmd = [
        "sudo", "env", f"PATH={os.environ['PATH']}",
        "ncu", "-f", "-o", report, f"./{binary}",
    ]
    print(f"[run ] (cwd={build_dir}) {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=build_dir)

    if result.returncode != 0:
        print(f"[fail] {binary} exited with code {result.returncode}")
        return False

    print(f"[ok  ] wrote {build_dir / report}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Run ncu on binaries built from .cu files.")
    parser.add_argument("--src-dir", default=Path(__file__).resolve().parent, type=Path,
                        help="directory containing the .cu files (default: this script's directory)")
    parser.add_argument("--build-dir", type=Path,
                        help=f"directory containing the binaries (default: nearest parent '{DEFAULT_BUILD_DIR_NAME}')")
    args = parser.parse_args()

    src_dir = args.src_dir.resolve()
    if not src_dir.is_dir():
        print(f"Not a directory: {src_dir}", file=sys.stderr)
        return 1

    build_dir = args.build_dir.resolve() if args.build_dir else find_build_dir(src_dir)
    if build_dir is None or not build_dir.is_dir():
        print(f"Could not find '{DEFAULT_BUILD_DIR_NAME}'; pass --build-dir.", file=sys.stderr)
        return 1

    cu_files = sorted(src_dir.glob("*.cu"))
    if not cu_files:
        print(f"No .cu files found in {src_dir}")
        return 1

    results = [profile(src_dir, build_dir, f) for f in cu_files]
    print(f"\nDone: {sum(results)}/{len(results)} profiled successfully.")
    return 0 if all(results) else 2


if __name__ == "__main__":
    sys.exit(main())