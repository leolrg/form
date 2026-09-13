# Faster matching pipeline

The user approved three extensions to the existing exact CUDA matcher: remove rematch host round trips, improve search layout/scheduling, and reduce snapshot preparation. Preserve FORM's selected features, 27-neighbor search, strict thresholds, first-visited ties, four-coordinate distance, objective, and LM controller. Enlarged workloads must have identical CPU controls.

Search will read compact coordinate arrays and probe neighboring hash buckets across warp lanes. Candidate reduction retains the existing deterministic distance/tie ordering. Choose launch dimensions by measurement.

Accepted query indices will be grouped on device. Only small group counts and QR roots return during optimization. Raw correspondence arrays and map-insertion matches will be materialized after the rematch loop, or on explicit raw access. Deferred data must remain valid across factor construction, cache invalidation, rematches, reset, and destruction; summary counts must remain accurate.

Snapshot preparation will avoid eagerly converting every world map target back to its local frame on CPU. Reuse storage, compute local summary coordinates on GPU, and defer host target conversion until raw matches are requested. Retain CPU voxel construction for deterministic candidate ordering unless an exact replacement demonstrably improves it.

Measure all three changes against the frozen c4d1064 matcher and fresh original/optimized CPU controls at current, denser-feature, and larger-window settings. Use two reversed repeats over the same first 250 stairs scans, then complete 1190-scan stairs validation. Report timing scope, counts, trajectory differences, limitations, and commits. Preserve baseline artifacts and run correctness/sanitizer checks before timed runs.
