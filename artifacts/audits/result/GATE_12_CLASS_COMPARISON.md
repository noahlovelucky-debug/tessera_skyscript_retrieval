# Anchored High-Resolution Gate: Luna Audit

Luna judges retrieval relevance only from pixels. Each query is sent with its ranked Top-10 images in a single request; the response contains one boolean judgment per rank.

## 100-Class Result

- System: `gated_highres`
- Checkpoint: `artifacts/audits/luna_anchored_gated_v4_extended_v1/summary.json`
- Mean P@10: `0.9417`
- Mean discounted relevance@10: `0.9461`
- Mean Hit@10: `1.0000`

## Per-Class Results

| Query | P@10 | Discounted relevance@10 | Relevant / 10 | Global gate |
|---|---:|---:|---:|---:|
| airport | 1.0000 | 1.0000 | 10 | 0.3713 |
| bridge | 1.0000 | 1.0000 | 10 | 0.3753 |
| cemetery | 1.0000 | 1.0000 | 10 | 0.3679 |
| farmland | 0.8000 | 0.8365 | 8 | 0.3662 |
| golf_course | 1.0000 | 1.0000 | 10 | 0.3631 |
| hospital | 0.9000 | 0.9149 | 9 | 0.3738 |
| industrial_building | 0.8000 | 0.8048 | 8 | 0.3606 |
| parking_lot | 1.0000 | 1.0000 | 10 | 0.3698 |
| railway | 1.0000 | 1.0000 | 10 | 0.3716 |
| river | 1.0000 | 1.0000 | 10 | 0.3719 |
| school | 0.8000 | 0.7975 | 8 | 0.3691 |
| swimming_pool | 1.0000 | 1.0000 | 10 | 0.3627 |

## 12-Class Gate Comparison

Baseline `gated_highres`: P@10 `0.8833`, discounted relevance `0.8660`.

| Query | Baseline P@10 | Anchored P@10 | Baseline discounted | Anchored discounted |
|---|---:|---:|---:|---:|
| airport | 0.9000 | 1.0000 | 0.7799 | 1.0000 |
| bridge | 0.9000 | 1.0000 | 0.9149 | 1.0000 |
| cemetery | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| farmland | 0.3000 | 0.8000 | 0.2083 | 0.8365 |
| golf_course | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| hospital | 1.0000 | 0.9000 | 1.0000 | 0.9149 |
| industrial_building | 0.9000 | 0.8000 | 0.9052 | 0.8048 |
| parking_lot | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| railway | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| river | 0.9000 | 1.0000 | 0.9337 | 1.0000 |
| school | 0.8000 | 0.8000 | 0.7137 | 0.7975 |
| swimming_pool | 0.9000 | 1.0000 | 0.9364 | 1.0000 |
