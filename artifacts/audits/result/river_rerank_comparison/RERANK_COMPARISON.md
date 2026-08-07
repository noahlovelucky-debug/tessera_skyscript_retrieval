# River Rerank Comparison

All methods reorder exactly the same coarse Top-100 candidates. Relevance labels are Luna pixel judgments collected before any reranking.

| Method | P@10 | nDCG@10 | P@25 | nDCG@25 | P@100 | nDCG@100 |
|---|---:|---:|---:|---:|---:|---:|
| coarse | 0.9000 | 0.9337 | 0.8800 | 0.9065 | 0.7300 | 0.9601 |
| local_rerank | 1.0000 | 1.0000 | 0.8000 | 0.8535 | 0.7300 | 0.9560 |
| luna_confidence_rerank | 0.8000 | 0.8630 | 0.6800 | 0.7409 | 0.7300 | 0.9234 |
| luna_label_oracle_upper_bound | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.7300 | 1.0000 |

`luna_label_oracle_upper_bound` is intentionally not a deployable result: it sorts by the same boolean labels used for evaluation. `luna_confidence_rerank` is the deployable-style proxy, but it is still self-evaluated by the same judge and needs a separate human or independent judge audit before any product claim.
