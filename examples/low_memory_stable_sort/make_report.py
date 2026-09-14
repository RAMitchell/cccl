#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 Rory Mitchell
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Generate the table and standalone figure from the recorded measurements."""

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def main():
    rows = list(csv.DictReader((RESULTS / "final.csv").open()))
    scalar = next(csv.DictReader((RESULTS / "scalar-final.csv").open()))
    env = json.loads((RESULTS / "environment.json").read_text())
    random = [r for r in rows if r["distribution"] == "random"]
    original = {
        (r["n"], r["record_bytes"]): r
        for r in csv.DictReader((RESULTS / "original.csv").open())
        if r["distribution"] == "random"
    }
    text = [
        "# Measured results",
        "",
        "RTX PRO 6000 Blackwell Workstation Edition, 188 SMs; CUDA 12.9; CCCL 3.4.2.",
        f"{env['timed_repeats']}-trial medians after {env['warmups_per_algorithm']} warmups per algorithm. Recorded {env['timestamp_utc']}.",
        "",
        "The strongest case is large comparison sorting of records. The complete output is returned to the original iterator range, including the final physical-page permutation.",
        "",
        "| Records | Record bytes | Thrust scratch MB | Paged scratch MB | Thrust peak MB | Paged peak MB | Thrust ms | Paged ms | Thrust throughput retained | Scratch reduction |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in random:
        old, new = int(r["baseline_temp_bytes"]), int(r["candidate_temp_bytes"])
        input_bytes = int(r["n"]) * int(r["record_bytes"])
        text.append(
            f"| {int(r['n']):,} | {r['record_bytes']} | {old / 1e6:.2f} | {new / 1e6:.2f} | {(input_bytes + old) / 1e6:.2f} | {(input_bytes + new) / 1e6:.2f} | {float(r['baseline_ms']):.2f} | {float(r['candidate_ms']):.2f} | {float(r['throughput_ratio']):.1%} | {1 - new / old:.2%} |"
        )
    text += [
        "",
        "MB means 1,000,000 bytes. Peak memory is calculated as the input array plus measured scratch; it excludes CUDA runtime/context overhead and benchmark harness allocations. Throughput retained is `Thrust time / paged time`; 90% throughput is about 11% longer runtime.",
        "",
        "![Storage and throughput](summary.png)",
        "",
        "## Refinement versus the initial prototype",
        "",
        "The initial prototype is commit `a3df667d9`; its original nine-trial measurements and metadata are preserved in `original.csv` and `original-environment.json`. The table below compares those measurements with this implementation on the same GPU and CCCL version. These are separate benchmark runs, not interleaved original/refined trials.",
        "",
        "The retained changes remove two full device-property queries, release single-tile inputs before allocating their output, bound the free queue by worker count, and share metadata between phases. Other transpose and register experiments did not give consistent enough gains to retain.",
        "",
        "| Records | Record bytes | Initial ms | Refined ms | Runtime reduction | Initial scratch MB | Refined scratch MB |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in random:
        old = original[(r["n"], r["record_bytes"])]
        before, after = float(old["candidate_ms"]), float(r["candidate_ms"])
        text.append(
            f"| {int(r['n']):,} | {r['record_bytes']} | {before:.2f} | {after:.2f} | {1 - after / before:.1%} | {float(old['candidate_temp_bytes']) / 1e6:.2f} | {float(r['candidate_temp_bytes']) / 1e6:.2f} |"
        )
    text += [
        "",
        "## Distribution sensitivity",
        "",
        "Both algorithms use the same key-only comparator over records that also carry original positions and payloads. Ordered and separated runs can use page-map fast paths; their speedups should not be generalized to random data.",
        "",
        "| Input, 67,108,864 records | Record bytes | Thrust ms | Paged ms | Speed relative to Thrust |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in rows:
        if int(r["n"]) == 1 << 26:
            text.append(
                f"| {r['distribution']} | {r['record_bytes']} | {float(r['baseline_ms']):.2f} | {float(r['candidate_ms']):.2f} | {float(r['throughput_ratio']):.2f}x |"
            )
    text += [
        "",
        "## Plain 32-bit keys",
        "",
        f"For {int(scalar['n']):,} unsigned keys, the paged sort takes {float(scalar['candidate_ms']):.2f} ms with {float(scalar['candidate_temp_bytes']) / 1e6:.2f} MB of scratch. The explicit comparison baseline takes {float(scalar['merge_ms']):.2f} ms, and the default Thrust radix baseline takes {float(scalar['radix_ms']):.2f} ms. Retained throughput is {float(scalar['vs_merge']):.1%} and {float(scalar['vs_radix']):.1%}, respectively. This case does **not** meet the near-original-throughput target.",
        "",
        "## Validation",
        "",
    ]
    for name in [
        "correctness-final-8.txt",
        "correctness-final-16.txt",
        "correctness-stress.txt",
    ]:
        text.append(f"- `{name}`: {(RESULTS / name).read_text().strip()}")
    for tool in ["memcheck", "initcheck", "racecheck", "synccheck"]:
        lines = (RESULTS / f"{tool}-final.txt").read_text().splitlines()
        summaries = [
            line.removeprefix("========= ") for line in lines if "SUMMARY:" in line
        ]
        text.append(
            f"- `{tool}-final.txt`: {len(summaries)} configurations; "
            + "; ".join(dict.fromkeys(summaries))
        )
    text += [
        "",
        "Tests compare 8- and 16-byte records exactly with CPU `std::stable_sort`, covering ascending/descending comparators, duplicate/ordered/adversarial inputs, tails, every supported page size, and empty inputs. Separate stress cases exercise guarded subranges, worker counts from 1 to 1,024, workspace reuse, a non-default stream, and signed-zero stability. Large benchmark outputs are additionally compared exactly with Thrust.",
        "",
        "Each sanitizer covers early release with 8- and 16-byte records and the multi-tile release path, all with partial pages. Sanitizer timing is instrumentation overhead and is not used in the performance tables. Sanitizers cover the specified finite cases, not all possible schedules or global-memory interleavings.",
        "",
        "## Measurement boundaries",
        "",
        "- Both sorts use reusable scratch buffers, so allocation growth is outside timed trials. CUDA event intervals include the complete sort, including setup kernels, merge levels, and final materialization. Each input is reset outside its timed interval; run order alternates.",
        "- The baseline's custom allocator records its complete temporary allocation. The paged implementation uses one allocation containing all data pages, maps, counts, queue storage, locks, and counters. Scratch figures exclude the input array; peak figures add its size to the scratch allocation.",
        "- The harness also retains a pristine input copy and reference output for resetting/verifying both implementations. Those harness allocations are excluded equally and are not required by the paged algorithm.",
        "- Each size/distribution uses a deterministic input; timings are repeat trials of that input, not independent random datasets. No GPU clocks or power limits were changed. Hardware/version generality has not been established.",
        "- Small and medium inputs are slower and can use more scratch than ordinary Thrust because of the hardware-sized spare reserve. The report does not claim a universal replacement for all stable-sort workloads.",
        "",
        "Raw data: `final.csv`, `scalar-final.csv`. Reproduction metadata and exact source hashes: `environment.json`. Build, validation, and benchmark commands are in the parent README.",
    ]
    (RESULTS / "report.md").write_text("\n".join(text) + "\n")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
    for size, color in [(8, "#2563eb"), (16, "#9333ea")]:
        subset = sorted(
            (r for r in random if int(r["record_bytes"]) == size),
            key=lambda r: int(r["n"]),
        )
        x = [int(r["n"]) / 1e6 for r in subset]
        axes[0].plot(
            x,
            [float(r["throughput_ratio"]) * 100 for r in subset],
            "o-",
            color=color,
            label=f"{size}-byte records",
        )
        axes[1].plot(
            x,
            [float(r["candidate_temp_bytes"]) / 1e6 for r in subset],
            "o-",
            color=color,
            label=f"Paged, {size} bytes",
        )
        axes[1].plot(
            x,
            [float(r["baseline_temp_bytes"]) / 1e6 for r in subset],
            "--",
            color=color,
            label=f"Thrust, {size} bytes",
        )
    axes[0].axhline(100, color="#777", linestyle="--", linewidth=1)
    axes[0].set(
        xscale="log",
        ylim=(0, 110),
        ylabel="Thrust comparison throughput retained (%)",
        title="Random records: throughput",
    )
    axes[1].set(
        xscale="log",
        yscale="log",
        ylabel="Auxiliary device storage (MB)",
        title="Complete sort workspace",
    )
    for ax in axes:
        ax.set_xlabel("Records (millions)")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.savefig(RESULTS / "summary.png", dpi=180)
    fig.savefig(RESULTS / "summary.svg")


if __name__ == "__main__":
    main()
