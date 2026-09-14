# Illustrated algorithm guide

[Read the eight-page PDF](paged_stable_sort_explained.pdf). It explains the page map,
walks through a stable merge, describes the bounded reserve, and presents the refined
implementation's timing, scratch and peak-memory results. The appendix records the exact
allocation formula, implementation revision and primary references.

The [standalone throughput figure](throughput.pdf) is a vector PDF suitable for reuse.

## Rebuild

From `examples/low_memory_stable_sort`, in a Python environment with the dependencies:

```bash
python -m pip install -r docs/requirements.txt
python docs/make_pdf.py
```

The generator reads `results/final.csv`, `original.csv`, `scalar-final.csv`, and
`environment.json`. It checks the historical measured hashes and the imported implementation hashes against
`../import-provenance.json` before rendering. Measurements remain those of the original
CCCL 3.4.2 study; the source changes made during import are recorded separately. It does not run GPU benchmarks or change the sorting implementation.

The layout uses local Noto Sans, Noto Serif, Noto Sans Mono and DejaVu Sans TrueType fonts
under `/usr/share/fonts/truetype`; adjust `FONTS` and the fallback-font paths when building
on another platform. Fonts are embedded in the final PDF. Mathematics uses a checked
fallback font so arrows, inequalities and square-root symbols render correctly.

Intermediate chart images are written under the ignored `results/development/pdf`
directory. The document date comes from the measurement metadata. All diagrams and
charts are generated locally; cited papers are linked, not reproduced.
