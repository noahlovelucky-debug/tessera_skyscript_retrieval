# Multi-Class Rerank Comparison

All methods reorder exactly the same coarse Top-100 candidates per query. Relevance labels are Luna pixel judgments collected before any reranking.

| Method | Mean P@10 | Mean pool-nDCG@10 | Mean P@25 | Mean pool-nDCG@25 | Mean P@100 | Mean pool-nDCG@100 |
|---|---:|---:|---:|---:|---:|---:|
| coarse | 0.9000 | 0.9337 | 0.8800 | 0.9065 | 0.7300 | 0.9601 |
| local_rerank | 1.0000 | 1.0000 | 0.8000 | 0.8535 | 0.7300 | 0.9560 |
| luna_confidence_rerank | 0.8000 | 0.8630 | 0.6800 | 0.7409 | 0.7300 | 0.9234 |
| luna_label_oracle_upper_bound | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.7300 | 1.0000 |

`pool-nDCG` is normalized by the judged positives already present in each fixed Top-100 candidate pool. It is not a full-corpus nDCG or recall metric. `luna_label_oracle_upper_bound` is intentionally not a deployable result: it sorts by the same boolean labels used for evaluation. `luna_confidence_rerank` is the deployable-style proxy, but it is still self-evaluated by the same judge and needs a separate human or independent judge audit before any product claim.
