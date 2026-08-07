# Anchored High-Resolution Gate: Luna Audit

Luna judges retrieval relevance only from pixels. Each query is sent with its ranked Top-10 images in a single request; the response contains one boolean judgment per rank.

## 100-Class Result

- System: `gated_highres`
- Checkpoint: `artifacts/audits/luna_basketball_anchored_v1/summary.json`
- Mean P@10: `0.5000`
- Mean discounted relevance@10: `0.6083`
- Mean Hit@10: `1.0000`

## Per-Class Results

| Query | P@10 | Discounted relevance@10 | Relevant / 10 | Global gate |
|---|---:|---:|---:|---:|
| basketball court | 0.5000 | 0.6083 | 5 | 0.3658 |

## 12-Class Gate Comparison

Baseline `gated_highres`: P@10 `0.3000`, discounted relevance `0.2687`.

| Query | Baseline P@10 | Anchored P@10 | Baseline discounted | Anchored discounted |
|---|---:|---:|---:|---:|
| basketball court | 0.3000 | 0.5000 | 0.2687 | 0.6083 |
