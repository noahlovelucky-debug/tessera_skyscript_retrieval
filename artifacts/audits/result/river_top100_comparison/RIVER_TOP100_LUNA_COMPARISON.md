# River Top-100 High-Resolution Retrieval Audit

Luna judged each candidate solely from pixels. Every method searched the same 39,344-image test pool for `river`; each Top-100 list was sent in ten 10-image batches.

| Method | Global/local rule | Relevant / 100 | P@10 | P@25 | P@50 | P@100 | Discounted relevance@100 |
|---|---|---:|---:|---:|---:|---:|---:|
| old_gate | text sigmoid gate | 73 | 0.9000 | 0.8800 | 0.7800 | 0.7300 | 0.7682 |
| anchored_gate | anchored text gate, local-first | 64 | 1.0000 | 0.7200 | 0.6800 | 0.6400 | 0.6837 |
| no_gate | fixed 0.35 global / 0.65 local | 58 | 0.9000 | 0.8000 | 0.6600 | 0.5800 | 0.6398 |

## Interpretation

The anchored gate improves the first ten river candidates, but its relevance falls below the old gate as the candidate budget grows. The old gate is therefore stronger for a large candidate pool; the anchored gate is stronger for a short coarse-search first page. The fixed no-gate high-resolution baseline trails both at Top-100.

Visuals: `relevance_strips.jpg` shows every judged rank; `precision_curve.jpg` shows P@k; the `top100_grids/` directory contains the complete 100-image grids.
