#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 Rory Mitchell
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Run the reproducible comparison-sort study, serially on the selected GPU."""

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--full",
        action="store_true",
        help="Include one billion records and structured inputs",
    )
    parser.add_argument("--repeats", type=int, default=9)
    args = parser.parse_args()
    results = ROOT / os.environ.get("RESULTS_DIR", "results/development/current")
    results.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.pop("LMS_PROFILE", None)
    env["LMS_WARMUPS"] = "5"
    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "gpu": subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,driver_version,memory.total",
                "--format=csv",
            ],
            text=True,
        ).strip(),
        "compiler": subprocess.check_output(
            [
                os.environ.get(
                    "NVCC",
                    (
                        str(Path(os.environ["CUDA_ENV"]) / "bin/nvcc")
                        if os.environ.get("CUDA_ENV")
                        else "nvcc"
                    ),
                ),
                "--version",
            ],
            text=True,
        ).strip(),
        "architecture": os.environ.get("CUDA_ARCH", "sm_120"),
        "warmups_per_algorithm": 5,
        "timed_repeats": args.repeats,
        "summary": "median CUDA event milliseconds; reset copies and allocation growth excluded",
        "seed": 42,
        "sources_sha256": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in [
                "paged_stable_sort.cuh",
                "benchmark.cu",
                "benchmark_scalar.cu",
                "correctness_extra.cu",
            ]
        },
    }
    cccl = Path(os.environ.get("CCCL_ROOT", ROOT.parent.parent))
    if not cccl.is_absolute():
        cccl = ROOT / cccl
    if cccl.exists():
        metadata["cccl_commit"] = subprocess.check_output(
            ["git", "-C", str(cccl), "rev-parse", "HEAD"], text=True
        ).strip()
    with (results / "final.csv").open("w", newline="") as f:
        writer = None
        for binary, sizes in [
            (
                "benchmark",
                [1 << 24, 1 << 26, 1 << 28] + ([1 << 30] if args.full else []),
            ),
            ("benchmark-record16", [1 << 26, 1 << 28]),
        ]:
            for n in sizes:
                distributions = ["random", "duplicates"]
                if args.full and n == 1 << 26:
                    distributions += [
                        "equal",
                        "sorted",
                        "reverse",
                        "runs",
                        "alternating",
                    ]
                for dist in distributions:
                    command = [
                        str(ROOT / binary),
                        str(n),
                        "0",
                        "0",
                        dist,
                        str(args.repeats),
                    ]
                    result = subprocess.run(
                        command,
                        env=env,
                        text=True,
                        capture_output=True,
                        timeout=300,
                        check=True,
                    )
                    version = int(re.search(r"Thrust=(\d+)", result.stderr).group(1))
                    row = next(csv.DictReader(result.stdout.splitlines()))
                    row = {"thrust_version": version, **row}
                    if writer is None:
                        writer = csv.DictWriter(f, fieldnames=list(row))
                        writer.writeheader()
                    writer.writerow(row)
                    f.flush()
                    print(binary, n, dist, row["throughput_ratio"], flush=True)
    scalar = subprocess.run(
        [str(ROOT / "benchmark-scalar"), str(1 << 26), str(args.repeats)],
        env=env,
        text=True,
        capture_output=True,
        timeout=300,
        check=True,
    )
    (results / "scalar-final.csv").write_text(scalar.stdout)
    (results / "environment.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
