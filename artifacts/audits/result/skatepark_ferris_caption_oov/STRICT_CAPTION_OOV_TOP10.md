# Strict Caption-OOV Top-10 Retrieval

Both exact query phrases occur zero times in the training captions. They are caption-OOV tests only, not proof that the frozen visual backbone has never seen the concept. Sol and Terra used strict object definitions so nearby recreational or utility infrastructure would not count as a match.

| Query | Strict visible target | Coarse P@10 | Sol P@10 | Relevant in Top-100 |
|---|---|---:|---:|---:|
| `skatepark` | Purpose-built bowls, half-pipes, ramps, rails, or continuous skate layout | 0.00 | 0.10 | 3 |
| `ferris wheel` | Large circular passenger wheel with hub and radial spokes | 0.00 | 0.10 | 3 |

The results are negative stress tests, not evidence of reliable open-vocabulary retrieval. Manual inspection of the Top-10 grids shows mostly recreational paths, plazas, utility structures, or visually similar circular/curved patterns. The one Terra-positive candidate per Sol Top-10 should be independently human-verified before counting it as a true detection.

Sol improves the placement of the small number of Terra-positive images but cannot add missing candidates: P@100 remains 0.03 for both queries. This indicates that the current high-resolution coarse encoder does not provide enough candidate coverage for these strict unseen objects.

## Visuals

- `skatepark_coarse_vs_sol_top10.jpg`
- `ferris_wheel_coarse_vs_sol_top10.jpg`

Green/red borders are Terra judgments under the strict definitions. Each image contains only the requested Top-10 candidates before and after Sol reranking.
