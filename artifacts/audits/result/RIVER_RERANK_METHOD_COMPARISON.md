# River Retrieval Method Comparison

## Scope

This table reuses completed `river` Top-100 audits. `pool-nDCG` is normalized only by positives judged inside the returned Top-100, not all corpus positives. It is therefore a fixed-candidate-pool ordering metric, not full-corpus nDCG or recall.

| System | Reranker | Visual judge | P@10 | P@25 | P@50 | P@100 | pool-nDCG@100 |
|---|---|---|---:|---:|---:|---:|---:|
| High-resolution gated coarse retrieval | none | Luna | 0.90 | 0.88 | 0.78 | 0.73 | 0.9601 |
| High-resolution no-gate retrieval | none | Luna | 0.90 | 0.80 | 0.66 | 0.58 | 0.9383 |
| High-resolution gated retrieval | local 0.35 global + 0.65 token MaxSim | Luna | 1.00 | 0.80 | 0.78 | 0.73 | 0.9560 |
| High-resolution gated retrieval | Luna confidence | Luna | 0.80 | 0.68 | 0.64 | 0.73 | 0.9234 |
| High-resolution gated retrieval | Sol two-stage Top-30 rerank | Terra | 1.00 | 1.00 | 0.88 | 0.72 | 0.9894 |

## Interpretation

- The gated coarse model improves candidate coverage over no-gate on this query: P@100 is 0.73 versus 0.58 under the corresponding Luna audits.
- Local token reranking raises the first-page result (P@10) but does not improve coverage and lowers P@25/pool-nDCG@100. It is a lightweight online reranker, not a candidate-recall fix.
- Luna-confidence reranking is worse than the original order in this run. Its confidence was collected separately per 10-image batch, so values are not well calibrated across batches. It is also evaluated by Luna itself.
- Sol gives the strongest reordering under Terra: P@10/P@25 both reach 1.00. Its P@100 remains the same candidate pool; 0.72 instead of 0.73 reflects Terra's independent labels, not a loss caused by reranking.

The first, third, and fourth rows use the identical old-gated candidate pool and Luna labels. The no-gate row has a different retrieved candidate pool. The Sol row has the identical old-gated candidate pool but a different visual judge, Terra. Do not compare Terra and Luna absolute scores as if they were a single human ground truth.

## Basketball Court Is Not Strict OVSS

`basketball court` is present in the training captions: 559 training titles contain `basketball`, and 459 contain the exact phrase `basketball court`. Its existing visualizations are useful as a seen-caption retrieval example, but they are not valid evidence for an open-vocabulary unseen-class claim. A strict OVSS visual should use a query absent from training titles and should be evaluated separately, ideally with human labels.

Existing seen-query visualization: `artifacts/audits/result/basketball_gate_comparison/basketball_court.jpg`.
