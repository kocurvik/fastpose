# fastpose — project website

This branch holds only the published page, so it can be served by GitHub Pages
without carrying the library sources.

- `index.html` — the whole site: markup, styles, script and the benchmark
  results, in one self-contained file. No build step and no external assets
  beyond the Google Fonts stylesheet.
- `.nojekyll` — tells GitHub Pages to serve the files as they are.

The numbers come from the CSVs produced by
[fastposebench](https://github.com/kocurvik/fastposebench) — every 5th pair of
the standard benchmark on ETH3D, ScanNet++ and PhotoTourism, with RoMa v2 and
LoMa correspondences. They are embedded in `index.html` as JSON; regenerate the
page rather than editing those numbers by hand.

To publish: Settings → Pages → Deploy from a branch → `website` / root.
