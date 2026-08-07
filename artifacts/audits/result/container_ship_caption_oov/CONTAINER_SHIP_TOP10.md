# Container Ship Caption-OOV Retrieval

`container ship` occurs zero times in the training, validation, and test captions. This makes it a caption-OOV query, not a proven visual-OOV concept: related terms such as `ship`, `boat`, `container`, and `dock` occur in training captions, and the frozen SkyCLIP backbone has its own pretraining data.

The old high-resolution gated retriever searched the 39,344-image test split for 100 candidates. Sol reranked the same candidates in two stages, and Terra judged each image from pixels without titles or Sol scores.

| Ordering | Terra P@10 | Terra P@25 | Terra P@100 | pool-nDCG@100 |
|---|---:|---:|---:|---:|
| High-resolution gated coarse | 0.20 | 0.16 | 0.11 | 0.5080 |
| Sol rerank | 0.90 | 0.40 | 0.11 | 0.9588 |

The figure contains only the requested Top-10 candidates per ordering. Green/red borders are Terra pixel judgments. Sol changes ranking only: the same Top-100 pool contains 11 Terra-relevant images, so P@100 cannot increase.

Visual: `container_ship_coarse_vs_sol_top10.jpg`.
