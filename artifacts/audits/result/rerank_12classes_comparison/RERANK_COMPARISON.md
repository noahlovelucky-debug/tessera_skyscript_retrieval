# Multi-Class Rerank Comparison

All methods reorder exactly the same coarse Top-100 candidates per query. Relevance labels are Luna pixel judgments collected before any reranking.

| Method | Mean P@10 | Mean pool-nDCG@10 | Mean P@25 | Mean pool-nDCG@25 | Mean P@100 | Mean pool-nDCG@100 |
|---|---:|---:|---:|---:|---:|---:|
| coarse | 0.9000 | 0.9027 | 0.9033 | 0.9035 | 0.8800 | 0.9687 |
| local_rerank | 0.9083 | 0.9062 | 0.8833 | 0.8885 | 0.8800 | 0.9659 |
| luna_confidence_rerank | 0.9833 | 0.9874 | 0.9567 | 0.9656 | 0.8800 | 0.9836 |
| luna_label_oracle_upper_bound | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.8800 | 1.0000 |

## Per-Query Detail

| Query | Coarse P@10 | Local P@10 | Luna confidence P@10 | Coarse pool-nDCG@10 | Local pool-nDCG@10 | Luna confidence pool-nDCG@10 | Coarse P@100 | Local P@100 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| river | 0.9000 | 1.0000 | 0.9000 | 0.9337 | 1.0000 | 0.9216 | 0.6900 | 0.6900 |
| school | 0.9000 | 0.9000 | 1.0000 | 0.9337 | 0.9216 | 1.0000 | 0.9300 | 0.9300 |
| farmland | 0.4000 | 0.5000 | 1.0000 | 0.4284 | 0.5366 | 1.0000 | 0.6700 | 0.6700 |
| hospital | 1.0000 | 1.0000 | 0.9000 | 1.0000 | 1.0000 | 0.9266 | 0.7100 | 0.7100 |
| airport | 0.9000 | 0.9000 | 1.0000 | 0.7799 | 0.7799 | 1.0000 | 0.9600 | 0.9600 |
| bridge | 0.9000 | 0.8000 | 1.0000 | 0.9149 | 0.8048 | 1.0000 | 0.7900 | 0.7900 |
| railway | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9800 | 0.9800 |
| swimming_pool | 0.9000 | 0.9000 | 1.0000 | 0.9364 | 0.9266 | 1.0000 | 0.9400 | 0.9400 |
| golf_course | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9800 | 0.9800 |
| parking_lot | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9900 | 0.9900 |
| industrial_building | 0.9000 | 0.9000 | 1.0000 | 0.9052 | 0.9052 | 1.0000 | 0.9200 | 0.9200 |
| cemetery | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

`pool-nDCG` is normalized by the judged positives already present in each fixed Top-100 candidate pool. It is not a full-corpus nDCG or recall metric. `luna_label_oracle_upper_bound` is intentionally not a deployable result: it sorts by the same boolean labels used for evaluation. `luna_confidence_rerank` is the deployable-style proxy, but it is still self-evaluated by the same judge and needs a separate human or independent judge audit before any product claim.
