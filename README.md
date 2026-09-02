# fastpose — project website

This branch holds only the published page, so it can be served by GitHub Pages
without carrying the library sources. It is an orphan branch: it shares no
history with `main` and should never be merged into it.

Published at <https://kocurvik.github.io/fastpose/> (Settings → Pages → Deploy
from a branch → `website` / root).

## Layout

| Path | What it is |
|---|---|
| `index.html` | The served page — markup, styles, script and results in one self-contained file. Generated; do not edit by hand. |
| `src/page.html` | The page template. `__DATA__` marks where the results are inlined. Edit this. |
| `src/bench.json` | The benchmark results the page plots. |
| `src/build.py` | `src/page.html` + `src/bench.json` → `index.html`. |
| `src/make_data.py` | fastposebench CSVs → `src/bench.json`. |
| `.nojekyll` | Serve the files as they are, without Jekyll. |

## Rebuilding

After editing the template:

```
python src/build.py
```

After a new benchmark run:

```
python src/make_data.py --csv_dir /path/to/fastposebench/csv_results
python src/build.py
```

`make_data.py` needs `pandas`; `build.py` needs only the standard library. Commit
the regenerated `index.html` along with whatever changed under `src/` — Pages
serves the built file, not the template.

## The numbers

From the CSVs produced by
[fastposebench](https://github.com/kocurvik/fastposebench): every 5th pair of the
standard benchmark on ETH3D, ScanNet++ and PhotoTourism, with RoMa v2 and LoMa
correspondences, comparing fastpose against PoseLib at matched iteration counts
and error thresholds.

One configuration is withheld: the varying-focal RePoseD solver returns
degenerate poses under the 4-CPU driver while its 1-CPU and CUDA runs are
normal, so `make_data.py` drops those rows rather than reporting them as a
speed-up. The rule lives in its `DROP` list.
