# Sol Rerank, Terra Audit, And Cost

## Evaluation Setup

The old gated high-resolution retriever supplied the same fixed Top-100 candidate images for each query. Sol received only the query and pixels, scored every image in batches of 10, then globally reordered its stage-1 Top-30. Terra then independently received only the query and the re-ranked image pixels. Terra did not receive titles, manifest labels, Luna labels, or Sol scores.

The coarse and Sol columns below use exactly the same Terra labels per query. P@100 remains unchanged because reranking does not add candidates. `pool-nDCG` is normalized against positives already present in that fixed Top-100; it is not full-corpus nDCG or recall.

| Query | Coarse P@10 | Sol P@10 | Coarse P@25 | Sol P@25 | Coarse P@100 | Sol P@100 | Coarse pool-nDCG@100 | Sol pool-nDCG@100 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| river | 0.90 | 1.00 | 0.76 | 1.00 | 0.72 | 0.72 | 0.9459 | 0.9894 |
| school | 0.80 | 1.00 | 0.76 | 1.00 | 0.64 | 0.64 | 0.9319 | 0.9892 |
| farmland | 0.50 | 1.00 | 0.64 | 1.00 | 0.64 | 0.64 | 0.8733 | 0.9924 |
| hospital | 0.90 | 1.00 | 0.88 | 1.00 | 0.67 | 0.67 | 0.9427 | 0.9922 |
| **Mean** | **0.775** | **1.000** | **0.760** | **1.000** | **0.668** | **0.668** | **0.9235** | **0.9908** |

This is stronger evidence than Luna self-evaluation because Terra is a separate model. It is still not a commercial acceptance metric: Sol and Terra are in the same model family, and all labels remain automated. Use a frozen human-labeled holdout for a product claim.

## Actual Tokens And Price Proxy Per Top-100

The token values are averages over the four completed queries. Token details returned by the API report prompt-cache hits. Dollar values use the official public GPT-5.6 input/cached-input/output rates, not the private gateway's billing policy. The private provider's invoice is authoritative.

| Stage | Calls | Input tokens | Cached input | Output tokens | Public-price proxy |
|---|---:|---:|---:|---:|---:|
| Sol rerank | 11 | 71.5k | 42.2k | 6.0k | $0.347 |
| Terra independent audit | 10 | 61.1k | 38.4k | 5.6k | $0.150 |
| **Sol + Terra** | **21** | **132.6k** | **80.6k** | **11.6k** | **$0.498** |
| Luna audit only, prior 12-class run | 10 | 58.1k | 38.4k | 5.6k | $0.057 |

The public list prices used here are Sol $5/$0.50/$30, Terra $2.50/$0.25/$15, and Luna $1/$0.10/$6 per million uncached-input/cached-input/output tokens. Source: [OpenAI GPT-5.6 model comparison](https://developers.openai.com/api/docs/models/compare).

Sol is more expensive chiefly because its token prices are higher, not because the image payload is dramatically larger. Terra evaluation adds about $0.15 per Top-100. For online retrieval, use Sol only on a small coarse candidate set and reserve Terra for offline audits or sampled quality monitoring.

## Latency

Observed serial wall time was approximately 5.3 minutes per Top-100 for the current Sol + Terra procedure. It makes 21 dependent serial vision requests: 10 Sol stage-1 batches, one Sol stage-2 Top-30 reorder, then 10 Terra batches. Prior Luna-only auditing used 10 batches of 10 images, with about 107 seconds of summed service time and roughly two minutes wall time per Top-100.

| Design | Sol + Terra estimate | Luna-only estimate | Status |
|---|---:|---:|---|
| Current batched, serial | ~5.3 min | ~2 min | Measured end-to-end workflow |
| Ten-image batches, concurrency 10 | ~50-90 s | ~15-30 s | Estimate; three Sol/Terra waves remain dependent |
| 100 one-image calls, serial | ~35-55 min | ~17-25 min | Estimate; high per-request overhead and no batch context |
| 100 one-image calls, concurrency 100 | ~45-100 s | ~10-30 s | Theoretical lower-range estimate only; provider quotas may queue requests |

The 100-single-request design is not recommended. It loses Sol's within-batch comparison context, causes much more prompt overhead, and is likely to cost more. It cannot eliminate the sequential Sol stage-2 dependency. The practical production configuration is 10-image batches with bounded concurrency (for example 5-10), subject to a measured RPM/TPM pressure test on the private gateway.

## Outputs

- River: `artifacts/audits/sol_rerank_terra_audit_river_v1/`
- School, farmland, hospital: `artifacts/audits/sol_rerank_terra_audit_3classes_v1/`
- Runner: `scripts/sol_rerank_and_terra_audit.py`
