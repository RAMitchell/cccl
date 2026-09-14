#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 Rory Mitchell
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Build the illustrated guide from the committed measurements; no GPU work is run."""

import csv
import json
import math
import os
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = ROOT / "results"
BUILD = RESULTS / "development" / "pdf"
BUILD.mkdir(parents=True, exist_ok=True)
PDF = HERE / "paged_stable_sort_explained.pdf"
ROWS = list(csv.DictReader((RESULTS / "final.csv").open()))
RANDOM = [r for r in ROWS if r["distribution"] == "random"]
ORIGINAL = {
    (r["n"], r["record_bytes"]): r
    for r in csv.DictReader((RESULTS / "original.csv").open())
    if r["distribution"] == "random"
}
SCALAR = next(csv.DictReader((RESULTS / "scalar-final.csv").open()))
ENV = json.loads((RESULTS / "environment.json").read_text())
# This document describes the saved CCCL 3.4.2 study.
assert {row["thrust_version"] for row in ROWS} == {"300402"}
assert SCALAR["thrust_version"] == "300402"
assert len(ROWS) == 22
BIG = next(r for r in RANDOM if int(r["n"]) == 1 << 30)

FONTS = Path("/usr/share/fonts/truetype/noto")
for name, filename in [
    ("Sans", "NotoSans-Regular.ttf"),
    ("Bold", "NotoSans-Bold.ttf"),
    ("Italic", "NotoSans-Italic.ttf"),
    ("Serif", "NotoSerif-Regular.ttf"),
    ("SerifBold", "NotoSerif-Bold.ttf"),
    ("Mono", "NotoSansMono-Regular.ttf"),
]:
    pdfmetrics.registerFont(TTFont(name, str(FONTS / filename)))
pdfmetrics.registerFontFamily(
    "Sans", normal="Sans", bold="Bold", italic="Italic", boldItalic="Bold"
)
pdfmetrics.registerFont(
    TTFont("Symbols", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
)
pdfmetrics.registerFont(
    TTFont("SymbolsBold", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
)
pdfmetrics.registerFontFamily(
    "Symbols",
    normal="Symbols",
    bold="SymbolsBold",
    italic="Symbols",
    boldItalic="SymbolsBold",
)

W, H = A4
M = 48
CW = W - 2 * M
INK = "#152C40"
MUTED = "#526371"
TEAL = "#087F75"
BLUE = "#315CB5"
AMBER = "#B76723"
PALE = "#F1F5F6"
LINE = "#DCE4E8"
GREEN_PALE = "#E4F3ED"
BLUE_PALE = "#EAF0FC"
WHITE = "#FFFFFF"

os.environ["SOURCE_DATE_EPOCH"] = str(
    int(datetime.fromisoformat(ENV["timestamp_utc"]).timestamp())
)
C = canvas.Canvas(str(PDF), pagesize=A4, pageCompression=1, invariant=1)
C.setTitle("Stable GPU sorting with recycled pages")
C.setCreator("Illustrated prototype report / docs/make_pdf.py")
C.setAuthor("GPU sorting prototype study")
C.setSubject(
    "An illustrated explanation and measured results for a low-memory stable CUDA merge sort"
)
C.setKeywords(
    "CUDA, GPU, stable sort, merge sort, page recycling, Thrust, CCCL, bounded workspace"
)
PAGE = 0


def fill(col):
    C.setFillColor(colors.HexColor(col))


def stroke(col):
    C.setStrokeColor(colors.HexColor(col))


def box(x, y, w, h, col=PALE, r=8, border=None):
    fill(col)
    if border:
        stroke(border)
    C.roundRect(x, H - y - h, w, h, r, fill=1, stroke=bool(border))


def label(txt, x, y, size=10, font="Sans", color=INK, align="left"):
    fill(color)
    segments = []
    for char in txt:
        chosen = (
            font
            if ord(char) in pdfmetrics.getFont(font).face.charToGlyph
            else ("SymbolsBold" if font in ("Bold", "SerifBold") else "Symbols")
        )
        assert ord(char) in pdfmetrics.getFont(chosen).face.charToGlyph, (char, font)
        if segments and segments[-1][0] == chosen:
            segments[-1] = (chosen, segments[-1][1] + char)
        else:
            segments.append((chosen, char))
    total = sum(pdfmetrics.stringWidth(s, f, size) for f, s in segments)
    if align == "right":
        x -= total
    elif align == "center":
        x -= total / 2
    for f, s in segments:
        C.setFont(f, size)
        C.drawString(x, H - y - size * 0.86, s)
        x += pdfmetrics.stringWidth(s, f, size)


def math_glyphs(txt):
    out = []
    for char in txt:
        if (
            ord(char) > 127
            and ord(char) not in pdfmetrics.getFont("Sans").face.charToGlyph
        ):
            assert ord(char) in pdfmetrics.getFont("Symbols").face.charToGlyph, char
            out.append(f'<font name="Symbols">{char}</font>')
        else:
            out.append(char)
    return "".join(out)


def para(
    txt, x, y, w=CW, size=10.5, leading=15.5, color=INK, font="Sans", align=TA_LEFT
):
    style = ParagraphStyle(
        "body",
        fontName=font,
        fontSize=size,
        leading=leading,
        textColor=colors.HexColor(color),
        alignment=align,
        spaceAfter=0,
    )
    p = Paragraph(math_glyphs(txt), style)
    _, h = p.wrap(w, H)
    assert y + h < H - 48, (PAGE, txt[:80], y, h)
    p.drawOn(C, x, H - y - h)
    return y + h


def line(x1, y1, x2, y2, col=LINE, width=1):
    stroke(col)
    C.setLineWidth(width)
    C.line(x1, H - y1, x2, H - y2)


def arrow(x1, y1, x2, y2, col=MUTED, width=1.2):
    line(x1, y1, x2, y2, col, width)
    theta = math.atan2(y2 - y1, x2 - x1)
    p = C.beginPath()
    p.moveTo(x2, H - y2)
    for off in [0.48, -0.48]:
        p.lineTo(x2 - 6 * math.cos(theta + off), H - (y2 - 6 * math.sin(theta + off)))
    p.close()
    fill(col)
    C.drawPath(p, fill=1, stroke=0)


def title(kicker, heading, sub=None):
    global PAGE
    PAGE += 1
    fill(WHITE)
    C.rect(0, 0, W, H, fill=1, stroke=0)
    C.bookmarkPage(f"p{PAGE}")
    C.addOutlineEntry(heading, f"p{PAGE}", 0, False)
    box(M, 31, 19, 4, TEAL, 2)
    label(kicker.upper(), M + 29, 26, 8.5, "Bold", MUTED)
    label(heading, M, 63, 26, "Bold")
    if sub:
        para(sub, M, 103, CW, 10.5, 15, color=MUTED)
    line(M, H - 41, W - M, H - 41)
    label("RECYCLED-PAGE STABLE SORT  /  PROTOTYPE STUDY", M, H - 28, 7, "Bold", MUTED)
    label(f"{PAGE:02d} / 08", W - M, H - 29, 8, "Sans", MUTED, "right")


def finish():
    C.showPage()


def minor(txt, x, y):
    label(txt, x, y, 12, "Bold")


def pill(txt, x, y, w, col=TEAL):
    box(x, y, w, 22, col, 11)
    label(txt, x + w / 2, y + 5, 8, "Bold", WHITE, "center")


def table(headers, rows, widths, y, rowh=27, hh=34, fs=9, highlight=None):
    assert abs(sum(widths) - CW) < 0.1
    x = M
    box(M, y, CW, hh, INK, 5)
    for i, (h, w) in enumerate(zip(headers, widths)):
        para(
            h,
            x + 9,
            y + 7,
            w - 18,
            8.1,
            10.5,
            WHITE,
            "Bold",
            TA_LEFT if i == 0 else TA_RIGHT,
        )
        x += w
    y += hh
    for ri, row in enumerate(rows):
        if ri == highlight:
            box(M, y, CW, rowh, GREEN_PALE, 0)
        elif ri % 2 == 0:
            box(M, y, CW, rowh, "#F5F7F8", 0)
        x = M
        for i, (cell, w) in enumerate(zip(row, widths)):
            label(
                str(cell),
                x + 9 if i == 0 else x + w - 9,
                y + (rowh - fs) / 2 - 1,
                fs,
                "Bold" if i == 0 or ri == highlight else "Sans",
                TEAL if ri == highlight else INK,
                "left" if i == 0 else "right",
            )
            x += w
        y += rowh
        line(M, y, W - M, y, LINE, 0.4)
    return y


def chips(values, x, y, w=50, h=33, color=TEAL, selected=None):
    for i, value in enumerate(values):
        bg = GREEN_PALE if color == TEAL else BLUE_PALE
        if selected is not None and i not in selected:
            bg = PALE
        box(x + i * (w + 5), y, w, h, bg, 5)
        label(value, x + i * (w + 5) + w / 2, y + 9, 11, "Bold", color, "center")


def workload(r):
    n = int(r["n"])
    return f"{n / 1e6:,.1f}M × {r['record_bytes']} B"


def make_plot():
    font_manager.fontManager.addfont(str(FONTS / "NotoSans-Regular.ttf"))
    plt.rcParams.update(
        {
            "font.family": "Noto Sans",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelcolor": MUTED,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.edgecolor": LINE,
            "text.color": INK,
            "axes.titleweight": "bold",
        }
    )
    fig, ax = plt.subplots(figsize=(6.93, 2.55), layout="constrained")
    for size, col, marker in [(8, TEAL, "o"), (16, BLUE, "s")]:
        r = [r for r in RANDOM if int(r["record_bytes"]) == size]
        ax.plot(
            [int(x["n"]) / 1e6 for x in r],
            [100 * float(x["throughput_ratio"]) for x in r],
            color=col,
            marker=marker,
            linewidth=2,
            markersize=6,
            label=f"{size}-byte records",
        )
    ax.axhline(100, color=MUTED, linestyle=(0, (4, 4)), linewidth=1)
    ax.text(16.8, 102, "Thrust comparison baseline", fontsize=8, color=MUTED)
    ax.set(
        xscale="log",
        ylim=(0, 110),
        xlim=(13, 1350),
        ylabel="Thrust throughput retained (%)",
        xlabel="Records (millions; logarithmic scale)",
    )
    ax.set_xticks(
        [16.777216, 67.108864, 268.435456, 1073.741824],
        ["16.8", "67.1", "268.4", "1,073.7"],
    )
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.grid(axis="y", alpha=0.22)
    ax.legend(loc="lower right", frameon=False, ncol=2, fontsize=8)
    fig.savefig(BUILD / "throughput.png", dpi=300)
    fig.savefig(HERE / "throughput.pdf")
    plt.close(fig)


def cover():
    global PAGE
    PAGE = 1
    C.bookmarkPage("p1")
    C.addOutlineEntry("Stable GPU sorting with recycled pages", "p1", 0, False)
    fill(INK)
    C.rect(0, H - 363, W, 363, fill=1, stroke=0)
    label("GPU ALGORITHMS  /  EXPERIMENTAL STUDY", M, 43, 9, "Bold", "#9FD7D0")
    label("Stable GPU sorting", M, 99, 34, "SerifBold", WHITE)
    label("with recycled pages", M, 147, 34, "SerifBold", WHITE)
    para(
        "An approachable guide to a merge sort that keeps the result in the original array and uses a small reusable workspace.",
        M,
        214,
        465,
        13,
        20,
        "#DCE9EE",
    )
    pill("STABILITY REQUIRED", M, 311, 132)
    label(
        "13 SEPTEMBER 2026  •  CCCL 3.4.2",
        M + 148,
        317,
        8,
        "Sans",
        "#C5D7E1",
    )
    facts = [
        ("32.14 MB", "temporary storage"),
        ("86.2%", "Thrust throughput retained"),
        ("8.62 GB", "accounted peak memory"),
    ]
    for i, (value, desc) in enumerate(facts):
        x = M + i * (CW + 14) / 3
        box(x, 390, (CW - 28) / 3, 94, PALE, 8)
        label(value, x + 13, 409, 24, "Bold", TEAL)
        para(desc, x + 13, 449, (CW - 28) / 3 - 26, 9, 12, MUTED)
    para(
        "Measured for <b>1,073,741,824 random 8-byte records</b> on one RTX PRO 6000 Blackwell. The comparison is against CCCL 3.4.2 through Thrust's stable comparison-sort path.",
        M,
        505,
        CW,
        10.5,
        15.5,
    )
    minor("The memory difference", M, 568)
    maxpeak = (int(BIG["n"]) * 8 + int(BIG["baseline_temp_bytes"])) / 1e9
    inputgb = int(BIG["n"]) * 8 / 1e9
    for y, name, scratch, col in [
        (601, "Thrust", int(BIG["baseline_temp_bytes"]) / 1e9, BLUE),
        (654, "Paged sort", int(BIG["candidate_temp_bytes"]) / 1e9, TEAL),
    ]:
        label(name, M, y, 10, "Bold")
        x = M + 88
        available = CW - 88
        inputw = available * inputgb / maxpeak
        fill("#CFD8E0")
        C.rect(x, H - y - 20, inputw, 22, fill=1, stroke=0)
        fill(col)
        C.rect(
            x + inputw, H - y - 20, available * scratch / maxpeak, 22, fill=1, stroke=0
        )
        label(
            f"{inputgb + scratch:.2f} GB",
            x + available,
            y + 27,
            9,
            "Bold",
            col,
            "right",
        )
    box(M, 715, 8, 8, "#CFD8E0", 0)
    label("input", M + 14, 713, 8, "Sans", MUTED)
    box(M + 63, 715, 8, 8, TEAL, 0)
    label("scratch (paged)", M + 77, 713, 8, "Sans", MUTED)
    box(M + 175, 715, 8, 8, BLUE, 0)
    label("scratch (Thrust)", M + 189, 713, 8, "Sans", MUTED)
    para(
        "Peak = input + measured scratch. CUDA runtime/context and benchmark-only copies are excluded. Bars use a common linear scale; the 32 MB paged workspace is a thin sliver beside the 8.59 GB input.",
        M,
        745,
        CW,
        8.2,
        11,
        MUTED,
    )
    label(
        "ALGORITHM  02–04     RESULTS  05–06     VALIDATION & SOURCES  07–08",
        M,
        803,
        7.5,
        "Bold",
        MUTED,
    )
    finish()


def overview():
    title(
        "01 / The idea",
        "Keep the order in a page map",
        "A run is a sorted sequence. A page is a fixed-size chunk of that sequence. The map tells us where each chunk currently lives.",
    )
    y = para(
        "The full-buffer comparison baseline allocates approximately another array of records while sorting. [2] This prototype keeps a small pool of spare pages instead. As merging consumes an input page, that storage becomes available for an output page.",
        M,
        155,
        CW,
        11,
        16,
    )
    box(M, 237, CW, 212, PALE)
    label("LOGICAL ORDER OF A SORTED RUN", M + 18, 254, 8.5, "Bold", MUTED)
    positions = [M + 74 + i * 104 for i in range(4)]
    for i, x in enumerate(positions):
        box(x, 281, 66, 38, GREEN_PALE, 5)
        label(f"page {i}", x + 33, 294, 10, "Bold", TEAL, "center")
    physical = [2, 0, 3, 1]
    for i, j in enumerate(physical):
        arrow(positions[i] + 33, 321, positions[j] + 33, 375, "#8B9BA7", 1)
    for j, x in enumerate(positions):
        box(x, 379, 66, 38, WHITE, 5, border=LINE)
        label(f"slot {j}", x + 33, 392, 10, "Bold", INK, "center")
    label("PHYSICAL STORAGE", M + 18, 429, 8, "Bold", MUTED)
    para(
        "Here the map is [2, 0, 3, 1]. Records remain contiguous <i>within</i> each page. These are software-managed chunks; the implementation does not remap hardware virtual memory.",
        M,
        464,
        CW,
        9.7,
        14,
        MUTED,
    )
    steps = [
        ("Sort locally", "CUB stably sorts the initial shared-memory-sized runs."),
        (
            "Plan each merge",
            "Find input cuts for each output page and count its readers.",
        ),
        (
            "Merge and recycle",
            "Workers merge pages and release input storage after its last read.",
        ),
        (
            "Swap the maps",
            "Double the run size and repeat using the new logical order.",
        ),
        (
            "Restore the array",
            "Move pages into their final physical positions once, at the end.",
        ),
    ]
    y = 526
    for i, (head, body) in enumerate(steps):
        box(M, y + 1, 22, 22, TEAL, 11)
        label(str(i + 1), M + 11, y + 6, 9, "Bold", WHITE, "center")
        label(head, M + 34, y, 10.5, "Bold")
        para(body, M + 34, y + 16, CW - 34, 9.4, 13, MUTED)
        y += 43
    box(M, 750, CW, 38, GREEN_PALE, 6)
    para(
        "<b>Worker = CUDA thread block.</b> The record benchmarks launch 376 blocks of 256 threads on 188 SMs. CUDA schedules the blocks; two per SM is an average, not a pinning rule.",
        M + 11,
        757,
        CW - 22,
        8.4,
        11.5,
        INK,
    )
    finish()


def example():
    title(
        "02 / A worked merge",
        "Recycle only after the last read",
        "Use four records per page for this illustration. The real benchmarks use thousands.",
    )
    para(
        "Two already-sorted runs occupy physical slots 0 and 1. Letters distinguish equal keys: 4a and 4b came before 4c in the original input. The comparator sees only the numeric key.",
        M,
        146,
        CW,
        10.5,
        15,
    )
    label(
        "One possible execution order is shown below; actual workers can run concurrently.",
        M,
        193,
        8.2,
        "Italic",
        MUTED,
    )
    label("LEFT RUN · SLOT 0", M, 212, 8.5, "Bold", TEAL)
    label("RIGHT RUN · SLOT 1", M + 263, 212, 8.5, "Bold", BLUE)
    chips(["1", "4a", "4b", "8"], M, 233, 48, color=TEAL)
    chips(["2", "4c", "7", "9"], M + 263, 233, 48, color=BLUE)
    box(M, 286, CW, 59, PALE, 6)
    para(
        "<b>Partition at output rank 4:</b> take the first 3 records from the left run and the first 1 from the right. The remaining records form the second output page. Merge Path searches find such cuts without producing the entire merged array first. [1]",
        M + 13,
        298,
        CW - 26,
        9.7,
        14,
    )
    y = 370
    for number, head, body in [
        (
            "1",
            "Load the first output page",
            "Load [1, 4a, 4b] and [2] into shared memory. Both input pages still have one future reader, so neither is free. Write [1, 2, 4a, 4b] into spare slot 2.",
        ),
        (
            "2",
            "Load the second output page",
            "Load [8] and [4c, 7, 9]. Both reader counts now reach zero. Slots 0 and 1 can be recycled. Write [4c, 7, 8, 9] into slot 1, using the on-chip copy of its former contents.",
        ),
        (
            "3",
            "Put the pages in their final positions",
            "The output map is [2, 1]. Slot 0 is free, so copy output page 0 from slot 2 into slot 0. Output page 1 is already in place. The original array now contains the complete result.",
        ),
    ]:
        box(M, y, 25, 25, TEAL, 12)
        label(number, M + 12.5, y + 7, 10, "Bold", WHITE, "center")
        minor(head, M + 38, y)
        y = para(body, M + 38, y + 22, CW - 38, 10, 14.5) + 22
    label("FINAL STABLE ORDER", M, 667, 8.5, "Bold", MUTED)
    chips(["1", "2", "4a", "4b", "4c", "7", "8", "9"], M, 687, 56, h=33, color=TEAL)
    para(
        "<b>Why ties stay stable:</b> the initial sort is stable, the left run wins equal-key comparisons, and partition searches use the same rule. Moving whole pages preserves their logical sequence. The letters are explanatory labels, not extra indices required by the algorithm.",
        M,
        735,
        CW,
        9.5,
        13.5,
    )
    finish()


def memory():
    title(
        "03 / The workspace",
        "A small reserve keeps work moving",
        "The reserve is tied to the number of workers. Metadata is tied to the number of pages.",
    )
    para(
        "Each worker processes a contiguous interval of output pages in order. Consumed input can be stranded at the two run boundaries, but the progress argument bounds this by two pages' worth per worker. It is this scheduling rule that makes the reserve predictable.",
        M,
        146,
        CW,
        10.5,
        15.5,
    )
    gap = 15
    bw = (CW - gap) / 2
    box(M, 222, bw, 112, GREEN_PALE)
    box(M + bw + gap, 222, bw, 112, PALE)
    label("2P + 4", M + 15, 238, 27, "Bold", TEAL)
    para(
        "Spare pages when a complete output page fits in shared memory. Load first, release readers, then obtain the destination.",
        M + 15,
        278,
        bw - 30,
        9.7,
        14,
    )
    label("3P + 4", M + bw + gap + 15, 238, 27, "Bold")
    para(
        "Spare pages for larger pages, which need several tiles. Their inputs remain protected until all tiles are consumed.",
        M + bw + gap + 15,
        278,
        bw - 30,
        9.7,
        14,
    )
    para(
        "<b>P counts launched worker blocks.</b> The argument assumes eventual progress of CUDA work and atomic operations; it does not require every block to be resident at once. A kernel boundary separates successive merge levels.",
        M,
        350,
        CW,
        9.7,
        14,
    )
    minor("Accounted storage across the random-record benchmarks", M, 411)
    rows = []
    for r in RANDOM:
        inp = int(r["n"]) * int(r["record_bytes"])
        a = int(r["baseline_temp_bytes"])
        b = int(r["candidate_temp_bytes"])
        rows.append(
            [
                workload(r),
                f"{a / 1e6:,.2f}",
                f"{b / 1e6:.2f}",
                f"{(inp + a) / 1e9:.3f}",
                f"{(inp + b) / 1e9:.3f}",
            ]
        )
    y = table(
        [
            "Records × size",
            "Thrust scratch<br/>MB",
            "Paged scratch<br/>MB",
            "Thrust peak<br/>GB",
            "Paged peak<br/>GB",
        ],
        rows,
        [133, 96, 90, 90, CW - 409],
        437,
        rowh=28,
        hh=36,
        fs=8.8,
        highlight=3,
    )
    para(
        "MB = 10⁶ bytes; GB = 10⁹ bytes. Peak = input + scratch. All explicit device workspace is counted, including maps, partitions, queues and locks. CUDA runtime/context and benchmark-only allocations are excluded.",
        M,
        y + 12,
        CW,
        8.8,
        12.5,
        MUTED,
    )
    minor("Choosing a page size", M, 710)
    para(
        "With N records, B records per page and S bytes per record, scratch is <b>O(PBS + N/B)</b>. Larger pages trade more spare-record storage for less metadata. For fixed P and S, growing B with √N gives square-root growth; the prototype's 65,536-record cap eventually leaves a small linear metadata term.",
        M,
        732,
        CW,
        9.4,
        13.5,
    )
    finish()


def performance():
    title(
        "04 / Measured performance",
        "Close to the comparison baseline",
        "Random records, key-only comparison, and a complete stable result in the original iterator range.",
    )
    C.drawImage(
        str(BUILD / "throughput.png"),
        M - 3,
        H - 155 - 193,
        width=CW + 6,
        height=193,
        mask="auto",
    )
    minor("Complete sort times", M, 374)
    rows = [
        [
            workload(r),
            f"{float(r['baseline_ms']):.2f}",
            f"{float(r['candidate_ms']):.2f}",
            f"{100 * float(r['throughput_ratio']):.1f}%",
        ]
        for r in RANDOM
    ]
    y = table(
        ["Records × size", "Thrust ms", "Paged ms", "Throughput retained"],
        rows,
        [153, 100, 100, CW - 353],
        398,
        rowh=29,
        hh=34,
        fs=9.6,
        highlight=3,
    )
    para(
        "<b>Throughput retained = Thrust time / paged time.</b> At 90% retained throughput, runtime is about 11% longer. The six random-record cases retain 85.9–91.8%; the largest 8- and 16-byte cases retain 86.2% and 90.3%, respectively.",
        M,
        y + 17,
        CW,
        10,
        14.5,
    )
    box(M, 676, CW, 106, PALE)
    label("HOW TO READ THESE NUMBERS", M + 14, 692, 8.5, "Bold", MUTED)
    para(
        "One RTX PRO 6000 Blackwell, 188 SMs; CUDA 12.9; CCCL 3.4.2. Values are medians of nine trials after five warmups per algorithm. CUDA-event intervals include the entire call, including host launch gaps and final page relocation. Input-reset copies and allocation growth are outside the timed interval.",
        M + 14,
        715,
        CW - 28,
        9.5,
        14,
    )
    finish()


def refinement():
    title(
        "05 / Refinements and workload shape",
        "Where the gains come from",
        "The strongest refinement is less scratch. Removing fixed setup overhead also helps shorter sorts.",
    )
    para(
        "The revised implementation releases single-tile inputs earlier, shrinks the free-page queue to the worker reserve, and shares metadata between merge and relocation phases. It also replaces two full device-property queries with the needed SM-count query, removing about 1.6 ms of setup on this machine.",
        M,
        145,
        CW,
        10.3,
        15,
    )
    chosen = [RANDOM[i] for i in [0, 1, 3, 5]]
    rows = []
    for r in chosen:
        old = ORIGINAL[(r["n"], r["record_bytes"])]
        rows.append(
            [
                workload(r),
                f"{float(old['candidate_ms']):.2f} → {float(r['candidate_ms']):.2f}",
                f"{int(old['candidate_temp_bytes']) / 1e6:.2f} → {int(r['candidate_temp_bytes']) / 1e6:.2f}",
            ]
        )
    y = table(
        ["Records × size", "Initial → refined ms", "Initial → refined scratch MB"],
        rows,
        [151, 151, CW - 302],
        226,
        rowh=29,
        hh=34,
        fs=9,
    )
    para(
        "The initial and refined values come from separate nine-trial runs on the same GPU and CCCL version. Large-array runtime changes of about 0.1% are effectively unchanged here. Scratch falls by 33–39% across the six random-record cases.",
        M,
        y + 12,
        CW,
        9.4,
        13.5,
        MUTED,
    )
    minor("Ordered inputs can bypass record merging", M, 443)
    para(
        "Already ordered run pairs retain their maps. Strictly separated pairs in reverse order can exchange page-map segments when alignment permits. These shortcuts explain the large speedups below; they should not be extrapolated to random data.",
        M,
        465,
        CW,
        9.6,
        14,
    )
    rows = []
    for dist, name in [
        ("random", "Random"),
        ("duplicates", "17 distinct keys"),
        ("equal", "All equal"),
        ("sorted", "Already sorted"),
        ("reverse", "Reverse ordered"),
        ("runs", "Two separated runs"),
        ("alternating", "Two interleaving runs"),
    ]:
        vals = [
            next(
                r
                for r in ROWS
                if int(r["n"]) == 1 << 26
                and int(r["record_bytes"]) == sz
                and r["distribution"] == dist
            )
            for sz in [8, 16]
        ]
        rows.append([name, *[f"{float(r['throughput_ratio']):.2f}×" for r in vals]])
    y = table(
        ["67.1M records", "8-byte records", "16-byte records"],
        rows,
        [249, 125, CW - 374],
        522,
        rowh=21,
        hh=29,
        fs=8.8,
    )
    para("Speed relative to Thrust; above 1× is faster.", M, y + 7, CW, 8.4, 11, MUTED)
    box(M, 731, CW, 55, "#FFF1E4", 6)
    para(
        f"<b>Plain 32-bit keys remain a weaker case.</b> At 67.1M keys, paged sort takes {float(SCALAR['candidate_ms']):.2f} ms: {100 * float(SCALAR['vs_merge']):.0f}% of the comparison baseline's throughput and {100 * float(SCALAR['vs_radix']):.0f}% of Thrust's default radix throughput. It uses {int(SCALAR['candidate_temp_bytes']) / 1e6:.2f} MB of scratch.",
        M + 12,
        741,
        CW - 24,
        9.2,
        13.2,
    )
    finish()


def validation():
    title(
        "06 / Evidence and use",
        "A working prototype, with clear limits",
        "Stability and data preservation were checked against independent reference results.",
    )
    metrics = [
        ("3,658", "correctness & stress cases"),
        ("12", "sanitizer configurations"),
    ]
    for i, (v, t) in enumerate(metrics):
        x = M + i * (CW + 16) / 2
        bw = (CW - 16) / 2
        box(x, 146, bw, 88, GREEN_PALE if i == 0 else PALE)
        label(v, x + 15, 160, 29, "Bold", TEAL)
        label(t, x + 15, 205, 9.5, "Sans", MUTED)
    para(
        'The 8- and 16-byte suites compare complete records with CPU <font name="Mono">std::stable_sort</font>, including the original position and payload. Cases cover ties, both comparator directions, tails and all supported page sizes. Stress cases exercise guarded subranges, workspace reuse, non-default streams, signed zeros and worker counts from 1 to 1,024.',
        M,
        252,
        CW,
        10.1,
        15,
    )
    para(
        "Memcheck, Initcheck, Racecheck and Synccheck each passed three configurations covering early release and larger-page merging. Large benchmark outputs also match Thrust exactly. These checks cover finite inputs and executions, rather than proving every possible global-memory schedule.",
        M,
        332,
        CW,
        10.1,
        15,
    )
    minor("Use the existing Thrust call shape", M, 411)
    box(M, 438, CW, 110, INK, 7)
    code = [
        "#include <thrust/sort.h>",
        '#include "paged_stable_sort.cuh"',
        "",
        "paged_sort::workspace scratch;",
        "paged_sort::policy policy;",
        "policy.scratch = &scratch;",
        "thrust::stable_sort(policy.on(stream), first, last, comp);",
    ]
    for i, s in enumerate(code):
        label(s, M + 14, 449 + i * 12.8, 8.8, "Mono", "#E1F0F2")
    para(
        "The custom policy opts into the prototype. The call returns the complete result in [first, last) and synchronizes its selected stream. An optional reusable workspace avoids repeated allocation growth.",
        M,
        564,
        CW,
        9.7,
        14,
    )
    minor("Current scope", M, 623)
    limits = [
        (
            "Data types",
            "Contiguous device iterators; trivially copyable, default-constructible records of at most 16 bytes.",
        ),
        (
            "Integration",
            "Graph capture, asynchronous policies, allocator integration and concurrent workspace reuse still need production work.",
        ),
        (
            "Performance",
            "One architecture and deterministic datasets. Very small workloads can have an unfavorable reserve; integer radix sorting remains much faster.",
        ),
        (
            "Traffic",
            "Normal merge passes still read and write the records. The final page permutation adds O(N) record movement.",
        ),
    ]
    y = 648
    for head, body in limits:
        label(head, M, y, 9, "Bold", TEAL)
        h = para(body, M + 87, y, CW - 87, 9.1, 12.5)
        y = max(y + 31, h + 8)
    finish()


def appendix():
    title(
        "Appendix / Accounting and reproduction",
        "Enough detail to reproduce the result",
        "The main explanation is complete. This page records the exact bound and the sources behind the study.",
    )
    para(
        "<b>Symbols.</b> N records; S bytes per record; B records per page; P worker blocks; Q = ⌈N/B⌉ pages. Let δ be 1 for a partial final page, else 0. Let E be 1 for early release, else 0; R = (3 − E)P + 4. K is the next power of two at least 3P + 4.",
        M,
        145,
        CW,
        9.6,
        14,
    )
    box(M, 215, CW, 57, PALE, 6)
    label("Exact auxiliary allocation, in bytes", M + 12, 226, 9, "Bold", MUTED)
    label(
        "(R + δ)BS + 8Q + max(20Q + 4(Q mod 2), 8(Q + R)) + 12K + 20",
        M + 12,
        249,
        9,
        "Mono",
    )
    para(
        "The terms are spare records, two page maps, the larger of the two phase-specific metadata layouts, the free queue, and counters. The shared phase region holds either reader counts plus partitions, or inverse ownership plus locks. Their lifetimes are separated by a kernel boundary.",
        M,
        286,
        CW,
        9.6,
        14,
    )
    para(
        "<b>Progress sketch for early release.</b> If L tasks have loaded and released their inputs and D have acquired destinations, then D ≤ L. At least L − 2P input pages can be reclaimed. Free pages are therefore at least R + L − 2P − D ≥ R − 2P = 4. Worker caches may hold some of them. The full argument and the 3P + 4 case are in DESIGN.md.",
        M,
        348,
        CW,
        9.6,
        14,
    )
    minor("Measurement setup", M, 424)
    para(
        "CCCL 3.4.2, CUDA 12.9.86, sm_120, driver 580.173.02. Measurements: 13 September 2026, nine-trial medians, five warmups, seed 42. Sorting used one RTX PRO 6000 Blackwell GPU with 188 SMs.",
        M,
        446,
        CW,
        9.4,
        13.5,
    )
    para(
        'In <font name="Mono">examples/low_memory_stable_sort</font>, README.md gives build commands; <font name="Mono">run_validation.sh</font> and <font name="Mono">run_benchmarks.py --full</font> reproduce the checks. <font name="Mono">results/final.csv</font>, <font name="Mono">scalar-final.csv</font> and <font name="Mono">environment.json</font> contain the saved measurements and environment details. Set <font name="Mono">CCCL_ROOT</font> to a CCCL 3.4.2 checkout to use the same dependency.',
        M,
        505,
        CW,
        9.1,
        13,
    )
    minor("Related work and primary sources", M, 578)
    refs = [
        (
            "[1]",
            "O. Green, R. McColl and D. A. Bader. <i>GPU Merge Path: A GPU Merging Algorithm.</i> ICS 2012. The partitioning primitive used here.",
            "https://davidbader.net/publication/2012-gm-ba/2012-gm-ba.pdf",
        ),
        (
            "[2]",
            "NVIDIA CCCL 3.4.2, <i>dispatch_merge_sort.cuh</i>. Baseline allocation and dispatch implementation used for the measurements.",
            "https://github.com/NVIDIA/cccl/blob/v3.4.2/cub/cub/device/dispatch/dispatch_merge_sort.cuh",
        ),
        (
            "[3]",
            "F. Moltmann, T.-V. Nakajima and S. Wild. <i>Virtual-Memory Powersort.</i> arXiv:2605.27147v2, 2026. Related stable sorting with reusable pages.",
            "https://arxiv.org/abs/2605.27147v2",
        ),
    ]
    y = 601
    for mark, txt, url in refs:
        label(mark, M, y, 9, "Bold", TEAL)
        y = (
            para(
                f'<link href="{url}" color="{TEAL}">{txt}</link>',
                M + 25,
                y,
                CW - 25,
                8.8,
                12.5,
            )
            + 9
        )
    para(
        "Page reuse has prior art [3]. This study evaluates a GPU schedule, reclamation protocol and final parallel relocation; it makes no claim that the underlying page-recycling idea is new.",
        M,
        750,
        CW,
        9.2,
        13.2,
        MUTED,
    )
    finish()


def main():
    make_plot()
    cover()
    overview()
    example()
    memory()
    performance()
    refinement()
    validation()
    appendix()
    assert PAGE == 8
    C.save()
    print(PDF)
    print(
        f"{PAGE} pages; {PDF.stat().st_size:,} bytes; reportlab {__import__('reportlab').Version}; matplotlib {matplotlib.__version__}"
    )


if __name__ == "__main__":
    main()
