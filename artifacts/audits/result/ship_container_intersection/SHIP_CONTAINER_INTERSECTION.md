# Ship And Container Hard-Intersection Test

The high-resolution gated retriever independently searched the full test split for `ship` and `container`, retaining Top-100 per query. Their hard intersection contains 14 images. Results are ordered by the worse of the two ranks, favoring images that rank highly for both text queries.

Manual inspection of the Top-10 visualization finds parking areas, warehouses, container-related facilities, and roads; it does not show a visible ship hull carrying containers. Therefore hard image-level intersection does not solve `container ship` retrieval in the current embedding space.

The failure is expected: `ship` and `container` can each match proxy semantics in unrelated image regions. A reliable composition requires pixel-level co-location: a ship mask, a container mask within the ship hull, and water context. The hard intersection remains a useful low-cost candidate filter, but it cannot be used as the final ranking rule.

Visual: `ship_container_hard_intersection_top10.jpg`.
