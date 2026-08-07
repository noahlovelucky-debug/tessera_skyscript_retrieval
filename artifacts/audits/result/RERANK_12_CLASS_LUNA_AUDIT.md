# 12-Class Luna Fixed-Pool Audit

## What Was Evaluated

For each of 12 English open-vocabulary queries, the old gated high-resolution retriever first produced its Top-100 images from the test split. Luna then judged the 1,200 returned images from pixels alone. The title and manifest label were not shown to the judge.

The methods below all reorder exactly the same per-query Top-100 candidate pool. Therefore P@100 cannot change. `pool-nDCG` is normalized against the positive images judged inside that fixed pool, so it measures their ordering only. It is not full-corpus nDCG and it says nothing about positives missing from the Top-100.

| Method | Mean P@10 | Mean pool-nDCG@10 | Mean P@25 | Mean pool-nDCG@25 | Mean P@100 | Mean pool-nDCG@100 |
|---|---:|---:|---:|---:|---:|---:|
| Coarse gated retrieval | 0.9000 | 0.9027 | 0.9033 | 0.9035 | 0.8800 | 0.9687 |
| Local token rerank | 0.9083 | 0.9062 | 0.8833 | 0.8885 | 0.8800 | 0.9659 |
| Luna-confidence rerank | 0.9833 | 0.9874 | 0.9567 | 0.9656 | 0.8800 | 0.9836 |

Luna-confidence reranking is not a deployable accuracy claim: Luna also generated the labels used for this evaluation. The local token rerank is a real local-model comparison; it produces a small average P@10 gain but lowers the average P@25 and does not improve candidate recall/P@100.

## Per Query

| Query | Coarse P@10 | Local P@10 | Coarse P@100 | Coarse pool-nDCG@100 |
|---|---:|---:|---:|---:|
| river | 0.90 | 1.00 | 0.69 | 0.9524 |
| school | 0.90 | 0.90 | 0.93 | 0.9847 |
| farmland | 0.40 | 0.50 | 0.67 | 0.8638 |
| hospital | 1.00 | 1.00 | 0.71 | 0.9589 |
| airport | 0.90 | 0.90 | 0.96 | 0.9497 |
| bridge | 0.90 | 0.80 | 0.79 | 0.9614 |
| railway | 1.00 | 1.00 | 0.98 | 0.9998 |
| swimming_pool | 0.90 | 0.90 | 0.94 | 0.9857 |
| golf_course | 1.00 | 1.00 | 0.98 | 0.9935 |
| parking_lot | 1.00 | 1.00 | 0.99 | 0.9995 |
| industrial_building | 0.90 | 0.90 | 0.92 | 0.9747 |
| cemetery | 1.00 | 1.00 | 1.00 | 1.0000 |

`farmland` is the weakest coarse query here, with P@10 = 0.40 and P@100 = 0.67. This is the clearest evidence that the system is not uniformly reliable across land-use categories, despite a strong aggregate Top-100 result.

## Why P@100 And pool-nDCG Can Both Be High

For the 12-query average, 88 of 100 returned images are visually relevant, producing mean P@100 = 0.88. The remaining negative images tend to occur later in the list. Since nDCG discounts late ranks and the ideal ordering contains the same judged positives, mean `pool-nDCG@100 = 0.9687` is possible and expected. It must not be read as 96.87% full-corpus recall or as all-result relevance.

## Judge Variance

The earlier river audit reported P@100 = 0.73. The present audit re-judged the exact same 100 candidate images: all sample IDs and ranks matched, but Luna changed 10 binary decisions, resulting in P@100 = 0.69. Treat single-pass Luna metrics as an automated audit estimate, not a ground truth. A publishable or product acceptance metric should use repeated independent judgments with adjudication, or a human-labeled holdout.

## Reproducible Outputs

- Pixel judgments: `artifacts/audits/luna_rerank_12classes_old_coarse_top100_v1/candidates_luna_judged.csv`
- Reranked candidates and machine-readable metrics: `artifacts/audits/result/rerank_12classes_comparison/`
- Full comparison table: `artifacts/audits/result/rerank_12classes_comparison/RERANK_COMPARISON.md`
