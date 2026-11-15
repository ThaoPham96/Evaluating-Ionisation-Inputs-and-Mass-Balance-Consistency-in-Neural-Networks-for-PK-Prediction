# Graphical abstract

`graphical-abstract.pdf` is an unchanged copy of the manuscript's `figures/toc.pdf` at Overleaf source commit `d526a705c48760f7440fc1e210b1aa38cc45da6c`.

`graphical-abstract.png` is a 600-dpi rendering of its single 234 × 126-point page, producing a 1950 × 1050-pixel image for the repository README. No labels, geometry, colours or scientific content were edited.

To regenerate the PNG with Poppler:

```bash
pdftoppm -png -singlefile -r 600 assets/graphical-abstract.pdf assets/graphical-abstract
```

The adjacent `graphical-abstract.provenance.json` records source and render checksums. The diagram describes the categorical-input and soft-loss comparison in the working manuscript; it does not claim completed results for the expanded continuous-descriptor robustness analysis.
