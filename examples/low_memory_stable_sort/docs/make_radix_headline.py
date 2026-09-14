#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 Rory Mitchell
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Render the saved radix headline as a vector PDF and a PNG; no GPU work is run."""

import csv
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results" / "radix"
FONT_DIR = Path("/usr/share/fonts/truetype/noto")
INK, TEAL, BLUE = "#152C40", "#087F75", "#315CB5"
MUTED, PALE, INPUT = "#526371", "#F1F5F6", "#CFD8E0"
WHITE = "#FFFFFF"
W, H = A4
M = 48
CW = W - 2 * M


def main():
    rows = list(csv.DictReader((RESULTS / "final.csv").open()))
    row = next(
        r for r in rows if int(r["n"]) == 1 << 30 and r["distribution"] == "random"
    )
    env = json.loads((RESULTS / "environment.json").read_text())
    measured = datetime.fromisoformat(env["timestamp_utc"])
    os.environ["SOURCE_DATE_EPOCH"] = str(int(measured.timestamp()))
    n = int(row["n"])
    input_bytes = n * 4
    baseline_scratch = int(row["baseline_temp_bytes"])
    paged_scratch = int(row["candidate_temp_bytes"])
    baseline_peak = int(row["baseline_peak_bytes"])
    paged_peak = int(row["candidate_peak_bytes"])
    baseline_ms, paged_ms = float(row["baseline_ms"]), float(row["candidate_ms"])
    retained = baseline_ms / paged_ms * 100
    saved = (1 - paged_scratch / baseline_scratch) * 100
    assert baseline_peak == input_bytes + baseline_scratch
    assert paged_peak == input_bytes + paged_scratch
    assert abs(retained - float(row["throughput_ratio"]) * 100) < 0.001
    assert row["thrust_version"] == "300600"

    for name, filename in [
        ("Sans", "NotoSans-Regular.ttf"),
        ("Bold", "NotoSans-Bold.ttf"),
        ("SerifBold", "NotoSerif-Bold.ttf"),
    ]:
        pdfmetrics.registerFont(TTFont(name, str(FONT_DIR / filename)))
    pdfmetrics.registerFontFamily("Sans", normal="Sans", bold="Bold")
    pdf = HERE / "radix_headline.pdf"
    c = canvas.Canvas(str(pdf), pagesize=A4, pageCompression=1, invariant=1)
    c.setTitle("Stable GPU radix sort with recycled pages — headline result")
    c.setAuthor("GPU sorting prototype study")
    c.setCreator("docs/make_radix_headline.py")
    c.setSubject(
        "Measured radix-sort time, scratch storage, and peak memory for 1.07 billion keys"
    )

    def rect(x, y, width, height, color, radius=0):
        c.setFillColor(colors.HexColor(color))
        c.roundRect(x, H - y - height, width, height, radius, fill=1, stroke=0)

    def label(text, x, y, size=10, font="Sans", color=INK, align="left"):
        c.setFillColor(colors.HexColor(color))
        c.setFont(font, size)
        width = pdfmetrics.stringWidth(text, font, size)
        if align == "right":
            x -= width
        elif align == "center":
            x -= width / 2
        assert x >= 0 and x + width <= W, text
        c.drawString(x, H - y - size * 0.86, text)

    def para(text, x, y, width=CW, size=10, leading=15, color=INK):
        style = ParagraphStyle(
            "body",
            fontName="Sans",
            fontSize=size,
            leading=leading,
            textColor=colors.HexColor(color),
        )
        p = Paragraph(text, style)
        _, height = p.wrap(width, H)
        assert y + height < H - 35, text
        p.drawOn(c, x, H - y - height)

    rect(0, 0, W, 363, INK)
    label("GPU ALGORITHMS  /  RADIX EXPERIMENT", M, 43, 9, "Bold", "#9FD7D0")
    label("Stable GPU radix sort", M, 99, 32, "SerifBold", WHITE)
    label("with recycled pages", M, 147, 32, "SerifBold", WHITE)
    para(
        f"<b>{saved:.1f}% less temporary storage</b>, retaining {retained:.1f}% of Thrust's "
        "radix-sort throughput. The stable result returns to the original array.",
        M,
        214,
        465,
        13,
        20,
        "#DCE9EE",
    )
    rect(M, 311, 143, 22, TEAL, 11)
    label("STABLE 32-BIT KEYS", M + 71.5, 316, 8, "Bold", WHITE, "center")
    label(
        measured.strftime("%d %B %Y").upper() + "  •  CCCL 3.6.0 CHECKOUT",
        M + 157,
        317,
        8,
        "Sans",
        "#C5D7E1",
    )
    facts = [
        (f"{paged_scratch / 1e6:.2f} MB", "temporary storage"),
        (f"{retained:.1f}%", "Thrust throughput retained"),
        (f"{paged_peak / 1e9:.2f} GB", "accounted peak memory"),
    ]
    for i, (value, description) in enumerate(facts):
        x = M + i * (CW + 14) / 3
        width = (CW - 28) / 3
        rect(x, 390, width, 94, PALE, 8)
        label(value, x + 13, 409, 24, "Bold", TEAL)
        para(description, x + 13, 449, width - 26, 9, 12, MUTED)

    para(
        f"<b>{n:,} random unsigned 32-bit keys</b> on one RTX PRO 6000 Blackwell. "
        f"Thrust radix: <b>{baseline_ms:.2f} ms</b>; paged radix: <b>{paged_ms:.2f} ms</b>. "
        "Stable ordering is checked separately using tagged records.",
        M,
        505,
        CW,
        10.5,
        15.5,
    )
    label("The memory difference", M, 568, 12, "Bold")
    x = M + 88
    available = CW - 88
    input_width = available * input_bytes / baseline_peak
    for y, name, scratch, color in [
        (601, "Thrust radix", baseline_scratch, BLUE),
        (654, "Paged radix", paged_scratch, TEAL),
    ]:
        scratch_width = available * scratch / baseline_peak
        label(name, M, y, 10, "Bold")
        rect(x, y - 2, input_width, 22, INPUT)
        rect(x + input_width, y - 2, scratch_width, 22, color)
        label(
            f"{(input_bytes + scratch) / 1e9:.2f} GB",
            x + input_width + scratch_width,
            y + 27,
            9,
            "Bold",
            color,
            "right",
        )
        if name == "Thrust radix":
            label(
                f"{input_bytes / 1e9:.2f} GB input",
                x + input_width / 2,
                y + 4,
                8.5,
                "Bold",
                INK,
                "center",
            )
            label(
                f"{scratch / 1e9:.2f} GB scratch",
                x + input_width + scratch_width / 2,
                y + 4,
                8.5,
                "Bold",
                WHITE,
                "center",
            )
        else:
            label(
                f"{scratch / 1e6:.2f} MB scratch",
                x + input_width + 12,
                y + 4,
                8.5,
                "Bold",
                TEAL,
            )

    for offset, color, text in [
        (0, INPUT, "input"),
        (63, TEAL, "scratch (paged)"),
        (175, BLUE, "scratch (Thrust)"),
    ]:
        rect(M + offset, 715, 8, 8, color)
        label(text, M + offset + 14, 713, 8, "Sans", MUTED)
    para(
        "Peak = input + measured scratch. CUDA runtime/context and benchmark copies are excluded. "
        "Bars share a linear scale; the paged scratch allocation is the thin teal sliver.",
        M,
        745,
        CW,
        8.2,
        11,
        MUTED,
    )
    label(
        f"CUDA 12.9  /  {env['timed_repeats']} TRIAL MEDIANS  /  {env['warmups_per_algorithm']} WARMUPS  /  REUSED ALLOCATIONS",
        M,
        790,
        7.3,
        "Bold",
        MUTED,
    )
    label(
        "Final page permutation included; input resets and allocation growth excluded.",
        M,
        807,
        7.5,
        "Sans",
        MUTED,
    )
    c.showPage()
    c.save()
    subprocess.run(
        [
            "pdftoppm",
            "-f",
            "1",
            "-singlefile",
            "-scale-to",
            "2000",
            "-png",
            str(pdf),
            str(HERE / "radix_headline"),
        ],
        check=True,
    )
    print(pdf)
    print(HERE / "radix_headline.png")


if __name__ == "__main__":
    main()
