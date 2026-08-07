# Anchored High-Resolution Gate: Luna Audit

Luna judges retrieval relevance only from pixels. Each query is sent with its ranked Top-10 images in a single request; the response contains one boolean judgment per rank.

## 100-Class Result

- System: `gated_highres`
- Checkpoint: `artifacts/audits/luna_anchored_gated_v4_100classes_v1/summary.json`
- Mean P@10: `0.8470`
- Mean discounted relevance@10: `0.8491`
- Mean Hit@10: `1.0000`

Frequency strata use the fixed title-query protocol in the linked CSV: 1 is high test support, 2 is medium, and 3 is lower support.

| Stratum | Mean P@10 | Mean discounted relevance@10 | Mean Hit@10 |
|---:|---:|---:|---:|
| 1 | 0.8909 | 0.8915 | 1.0000 |
| 2 | 0.8727 | 0.8756 | 1.0000 |
| 3 | 0.7794 | 0.7822 | 1.0000 |

## Per-Class Results

| Query | Test examples | Stratum | P@10 | Discounted relevance@10 | Relevant / 10 | Global gate |
|---|---:|---:|---:|---:|---:|---:|
| Abandoned railway | 83 | 1 | 0.4000 | 0.3227 | 4 | 0.3591 |
| Alley classified as service road | 186 | 1 | 0.3000 | 0.2579 | 3 | 0.3644 |
| Baseball pitch for leisure and sports activities | 143 | 1 | 1.0000 | 1.0000 | 10 | 0.3519 |
| Cemetery | 355 | 1 | 1.0000 | 1.0000 | 10 | 0.3679 |
| Commercial building | 391 | 1 | 1.0000 | 1.0000 | 10 | 0.3673 |
| Farm auxiliary building | 85 | 1 | 1.0000 | 1.0000 | 10 | 0.3564 |
| Farm building | 152 | 1 | 1.0000 | 1.0000 | 10 | 0.3667 |
| Golf cartpath that is also road path | 73 | 1 | 1.0000 | 1.0000 | 10 | 0.3591 |
| Greenhouse | 103 | 1 | 1.0000 | 1.0000 | 10 | 0.3659 |
| Healthcare facility identified as hospital | 51 | 1 | 1.0000 | 1.0000 | 10 | 0.3603 |
| Hospital building | 115 | 1 | 0.6000 | 0.6817 | 6 | 0.3690 |
| Islet | 116 | 1 | 1.0000 | 1.0000 | 10 | 0.3715 |
| Marked road crossing | 112 | 1 | 1.0000 | 1.0000 | 10 | 0.3755 |
| Parking space | 339 | 1 | 1.0000 | 1.0000 | 10 | 0.3786 |
| Path designated for golf carts | 80 | 1 | 1.0000 | 1.0000 | 10 | 0.3577 |
| Playground area for leisure activities on the land | 60 | 1 | 0.6000 | 0.6243 | 6 | 0.3538 |
| Playground area for leisure land | 66 | 1 | 0.7000 | 0.7315 | 7 | 0.3566 |
| Power pole | 213 | 1 | 1.0000 | 1.0000 | 10 | 0.3637 |
| Pylon for aerialway | 94 | 1 | 1.0000 | 1.0000 | 10 | 0.3561 |
| Residential road | 136 | 1 | 1.0000 | 1.0000 | 10 | 0.3778 |
| Road serving as parking aisle | 53 | 1 | 1.0000 | 1.0000 | 10 | 0.3710 |
| Road with traffic signals | 73 | 1 | 1.0000 | 1.0000 | 10 | 0.3764 |
| Rough area on grassy golf course | 61 | 1 | 1.0000 | 1.0000 | 10 | 0.3536 |
| Secondary road with asphalt surface | 99 | 1 | 0.1000 | 0.0734 | 1 | 0.3727 |
| Silo | 110 | 1 | 1.0000 | 1.0000 | 10 | 0.3654 |
| Soccer pitch designated for leisure and sports activities | 40 | 1 | 1.0000 | 1.0000 | 10 | 0.3510 |
| Soccer pitch for leisureland activities | 42 | 1 | 0.8000 | 0.8365 | 8 | 0.3515 |
| Storage tank with man-made structure | 84 | 1 | 1.0000 | 1.0000 | 10 | 0.3512 |
| Supermarket | 60 | 1 | 1.0000 | 1.0000 | 10 | 0.3687 |
| University building | 46 | 1 | 0.9000 | 0.8900 | 9 | 0.3677 |
| View of taxiway at the airport | 190 | 1 | 1.0000 | 1.0000 | 10 | 0.3632 |
| Warehouse | 82 | 1 | 1.0000 | 1.0000 | 10 | 0.3697 |
| Zebra crossing on the road | 140 | 1 | 1.0000 | 1.0000 | 10 | 0.3688 |
| Asphalt road classified as primary highway, featuring roundabout at junction | 21 | 2 | 1.0000 | 1.0000 | 10 | 0.3543 |
| Building with storage tank | 19 | 2 | 1.0000 | 1.0000 | 10 | 0.3562 |
| Building with tile roof | 23 | 2 | 0.9000 | 0.9216 | 9 | 0.3656 |
| Building with tiled roof | 28 | 2 | 0.9000 | 0.9149 | 9 | 0.3647 |
| Bunker on golf course | 19 | 2 | 1.0000 | 1.0000 | 10 | 0.3574 |
| Bus stop equipped with public transport stop position | 21 | 2 | 1.0000 | 1.0000 | 10 | 0.3648 |
| Catenary mast for power lines | 30 | 2 | 0.9000 | 0.8900 | 9 | 0.3463 |
| Crossing with traffic signals | 23 | 2 | 1.0000 | 1.0000 | 10 | 0.3695 |
| Cycleway | 20 | 2 | 0.5000 | 0.5072 | 5 | 0.3692 |
| Designated area for dogs to play, dog park | 26 | 2 | 0.2000 | 0.1546 | 2 | 0.3563 |
| Detached building with gabled roof | 36 | 2 | 1.0000 | 1.0000 | 10 | 0.3578 |
| Golf tee on grassy land | 20 | 2 | 0.5000 | 0.6372 | 5 | 0.3569 |
| Grass area designated as golf tee | 20 | 2 | 0.4000 | 0.4342 | 4 | 0.3585 |
| Lock gate along waterway | 20 | 2 | 0.8000 | 0.8701 | 8 | 0.3582 |
| Marsh wetland | 25 | 2 | 1.0000 | 1.0000 | 10 | 0.3604 |
| Miniature golf leisure land | 21 | 2 | 1.0000 | 1.0000 | 10 | 0.3588 |
| Mixed leaf woodland | 38 | 2 | 1.0000 | 1.0000 | 10 | 0.3598 |
| Poultry house | 29 | 2 | 0.9000 | 0.9052 | 9 | 0.3596 |
| Public building | 21 | 2 | 0.9000 | 0.9337 | 9 | 0.3686 |
| Residential area consisting of single-family homes | 24 | 2 | 1.0000 | 1.0000 | 10 | 0.3528 |
| Residential road with asphalt surface | 35 | 2 | 1.0000 | 1.0000 | 10 | 0.3697 |
| Rest area on the road | 28 | 2 | 0.6000 | 0.6307 | 6 | 0.3753 |
| Road crossing with traffic signals | 20 | 2 | 1.0000 | 1.0000 | 10 | 0.3707 |
| Roller coaster, exhilarating and thrilling attraction | 20 | 2 | 1.0000 | 1.0000 | 10 | 0.3616 |
| Secondary road around roundabout | 22 | 2 | 0.9000 | 0.9364 | 9 | 0.3647 |
| Solar power generator using photovoltaic panels | 21 | 2 | 1.0000 | 1.0000 | 10 | 0.3535 |
| Storage tank created as man-made structure | 26 | 2 | 1.0000 | 1.0000 | 10 | 0.3530 |
| Trunk link highway with asphalt surface | 31 | 2 | 0.9000 | 0.7799 | 9 | 0.3613 |
| Unmarked crossing on footway | 22 | 2 | 0.5000 | 0.3796 | 5 | 0.3634 |
| View of motorway | 30 | 2 | 1.0000 | 1.0000 | 10 | 0.3699 |
| View of wastewater plant | 26 | 2 | 1.0000 | 1.0000 | 10 | 0.3499 |
| Wind generator using wind turbines of horizontal axis | 23 | 2 | 1.0000 | 1.0000 | 10 | 0.3513 |
| Woodland with needle-leaved trees | 26 | 2 | 1.0000 | 1.0000 | 10 | 0.3574 |
| Beautiful meadow | 15 | 3 | 0.9000 | 0.9364 | 9 | 0.3653 |
| Bridge with electrified railway tracks featuring contact line | 13 | 3 | 0.5000 | 0.4482 | 5 | 0.3511 |
| Clinic building offering healthcare services | 13 | 3 | 0.4000 | 0.5353 | 4 | 0.3628 |
| Concrete bench | 13 | 3 | 0.8000 | 0.8572 | 8 | 0.3692 |
| Ditch in the surroundings | 15 | 3 | 0.9000 | 0.8900 | 9 | 0.3648 |
| Driving range for golf on grass | 12 | 3 | 0.2000 | 0.1331 | 2 | 0.3551 |
| Expressway with asphalt surface classified as trunk highway | 18 | 3 | 1.0000 | 1.0000 | 10 | 0.3582 |
| Fenced power substation at the transmission level | 12 | 3 | 1.0000 | 1.0000 | 10 | 0.3465 |
| Government office | 13 | 3 | 0.8000 | 0.8358 | 8 | 0.3673 |
| Grassy area designated as driving range for golf | 18 | 3 | 0.7000 | 0.7656 | 7 | 0.3544 |
| Grassy area used as driving range for golf | 14 | 3 | 0.6000 | 0.4896 | 6 | 0.3524 |
| Grassy golf fairway | 15 | 3 | 1.0000 | 1.0000 | 10 | 0.3597 |
| Historic wayside cross | 14 | 3 | 1.0000 | 1.0000 | 10 | 0.3623 |
| Hut | 13 | 3 | 1.0000 | 1.0000 | 10 | 0.3702 |
| Image of wastewater | 14 | 3 | 1.0000 | 1.0000 | 10 | 0.3591 |
| Junction on the railway | 15 | 3 | 0.2000 | 0.1546 | 2 | 0.3664 |
| Marked crossing with zebra markings on the road | 15 | 3 | 0.8000 | 0.8205 | 8 | 0.3549 |
| Miniature golf course | 14 | 3 | 0.8000 | 0.8572 | 8 | 0.3639 |
| Narrow-gauge railway track with 1 electrified contact line | 13 | 3 | 0.4000 | 0.2980 | 4 | 0.3481 |
| Path with excellent trail visibility | 18 | 3 | 0.9000 | 0.9337 | 9 | 0.3605 |
| Paved residential road | 12 | 3 | 1.0000 | 1.0000 | 10 | 0.3715 |
| Railway line on embankment | 12 | 3 | 1.0000 | 1.0000 | 10 | 0.3578 |
| Residential road with asphalt surface, without sidewalks | 14 | 3 | 1.0000 | 1.0000 | 10 | 0.3593 |
| Retail building housing supermarket | 17 | 3 | 1.0000 | 1.0000 | 10 | 0.3518 |
| Scrap yard located in industrial area | 14 | 3 | 1.0000 | 1.0000 | 10 | 0.3503 |
| Secondary road leading to roundabout | 13 | 3 | 1.0000 | 1.0000 | 10 | 0.3638 |
| Secondary road with frontage road alongside | 17 | 3 | 0.6000 | 0.6267 | 6 | 0.3611 |
| Secondary road with separate sidewalks | 15 | 3 | 0.8000 | 0.7163 | 8 | 0.3642 |
| Storage rental shop located in building | 12 | 3 | 0.4000 | 0.5077 | 4 | 0.3517 |
| Unclassified road with bridge | 16 | 3 | 0.4000 | 0.4632 | 4 | 0.3633 |
| Uncontrolled crossing with central island on the road | 16 | 3 | 0.4000 | 0.3251 | 4 | 0.3655 |
| Wastewater plant | 16 | 3 | 1.0000 | 1.0000 | 10 | 0.3533 |
| Wind generator powered by horizontal-axis wind turbine | 13 | 3 | 1.0000 | 1.0000 | 10 | 0.3487 |
| Wind power generator with wind turbines | 15 | 3 | 1.0000 | 1.0000 | 10 | 0.3518 |
