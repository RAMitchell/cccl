#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 Rory Mitchell
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Compare the paged radix prototype with Thrust's integer radix path."""

import argparse
import csv
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--full",
        action="store_true",
        help="Include one billion keys and structured inputs",
    )
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument(
        "--tuning",
        action="store_true",
        help="Also benchmark digit variants built by run_radix_validation.sh",
    )
    args = parser.parse_args()
    results = ROOT / os.environ.get("RESULTS_DIR", "results/development/radix-current")
    results.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.pop("LMS_PROFILE", None)
    env["LMS_WARMUPS"] = "5"
    compiler = env.get(
        "NVCC",
        str(Path(env["CUDA_ENV"]) / "bin/nvcc") if env.get("CUDA_ENV") else "nvcc",
    )
    cccl = Path(env.get("CCCL_ROOT", ROOT.parent.parent))
    if not cccl.is_absolute():
        cccl = ROOT / cccl
    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "cuda_visible_devices": env.get("CUDA_VISIBLE_DEVICES", ""),
        "gpu": subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,uuid,driver_version,memory.total",
                "--format=csv",
            ],
            text=True,
        ).strip(),
        "compiler": subprocess.check_output([compiler, "--version"], text=True).strip(),
        "architecture": env.get("CUDA_ARCH", "sm_120"),
        "extra_compiler_flags": env.get("NVCC_FLAGS", ""),
        "cccl_commit": subprocess.check_output(
            ["git", "-C", str(cccl), "rev-parse", "HEAD"], text=True
        ).strip(),
        "warmups_per_algorithm": 5,
        "timed_repeats": args.repeats,
        "summary": "median CUDA event milliseconds; alternating trial order; reset copies and allocation growth excluded",
        "memory_accounting": "input array plus allocator high-water capacity; CUDA context and harness copies excluded",
        "seed": 42,
        "sources_sha256": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in [
                "paged_radix_sort.cuh",
                "paged_stable_sort.cuh",
                "benchmark_radix.cu",
                "build.sh",
            ]
        },
    }
    (results / "environment.json").write_text(json.dumps(metadata, indent=2) + "\n")
    sizes = [1 << 20, 1 << 24, 1 << 26, 1 << 28] + ([1 << 30] if args.full else [])
    with (results / "final.csv").open("w", newline="") as f:
        writer = None
        for n in sizes:
            distributions = ["random", "duplicates"]
            if args.full and n == 1 << 26:
                distributions += ["equal", "sorted", "reverse", "alternating"]
            for distribution in distributions:
                result = subprocess.run(
                    [
                        str(ROOT / "benchmark-radix"),
                        str(n),
                        str(args.repeats),
                        "0",
                        distribution,
                    ],
                    env=env,
                    text=True,
                    capture_output=True,
                    timeout=300,
                    check=True,
                )
                row = next(csv.DictReader(result.stdout.splitlines()))
                if writer is None:
                    writer = csv.DictWriter(f, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
                f.flush()
                print(n, distribution, row["throughput_ratio"], flush=True)

    if args.tuning:
        with (results / "tuning.csv").open("w", newline="") as f:
            writer = None
            for bits in [4, 6, 7, 8]:
                binary = (
                    "benchmark-radix" if bits == 8 else f"benchmark-radix-bits{bits}"
                )
                result = subprocess.run(
                    [
                        str(ROOT / binary),
                        str(1 << 26),
                        str(args.repeats),
                        "0",
                        "random",
                    ],
                    env=env,
                    text=True,
                    capture_output=True,
                    timeout=300,
                    check=True,
                )
                row = next(csv.DictReader(result.stdout.splitlines()))
                if writer is None:
                    writer = csv.DictWriter(f, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
                f.flush()
                print("digit bits", bits, row["throughput_ratio"], flush=True)
    result = subprocess.run(
        [str(ROOT / "benchmark-radix"), str(1 << 26), str(args.repeats), "0", "random"],
        env={**env, "LMS_PROFILE": "1"},
        text=True,
        capture_output=True,
        timeout=300,
        check=True,
    )
    (results / "profile.txt").write_text(
        "Milliseconds, separate profiled run:\n" + result.stderr
    )


if __name__ == "__main__":
    main()
