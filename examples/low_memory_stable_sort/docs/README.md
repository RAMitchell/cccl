# Illustrated algorithm guide

[Read the eight-page PDF](paged_stable_sort_explained.pdf). It explains the page map,
walks through a stable merge, describes the bounded reserve, and presents timing,
scratch and peak-memory results. The appendix gives the exact allocation formula,
measurement setup, and references.

The [standalone throughput figure](throughput.pdf) is a vector PDF suitable for reuse.

## Rebuild

From `examples/low_memory_stable_sort`:

```bash
python -m pip install -r docs/requirements.txt
python docs/make_pdf.py
```

The generator renders the saved CCCL 3.4.2 study from the CSV files and environment
metadata in `results/`. It does not run GPU benchmarks. The document date comes from
the measurement metadata.

The layout uses Noto Sans, Noto Serif, Noto Sans Mono and DejaVu Sans TrueType fonts
under `/usr/share/fonts/truetype`. Adjust `FONTS` and the fallback-font paths on another
platform. Fonts are embedded in the PDF, including the mathematical symbols.

Intermediate chart images go to `results/development/pdf`. Diagrams and charts are
generated locally; cited papers are linked, not reproduced.
