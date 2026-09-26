# Project status

## Outcome
Best verified pure-DL complete grouped5fold OOF81.71323318479078 (previous81.5027197525581). Target82 remains unmet. Last posterior-router fifth80.25089377481923, best52/naturalstop70 of80, below strict80.4. All training exited; no new jobs or continuation authorized.

## Completed
- Added an imagegen-generated architecture diagram to README and METHOD.md, checked against the delivered calibrated-pair architecture. Corrected generated tensor/stage annotations before publication. Documentation/image update only; training remains stopped and delivered archives unchanged.
- User-reported A-leaderboard score83.6915 added for the delivered calibrated-pair five-fold ensemble; provenance explicitly user feedback, not independently fetched. OOF81.713233 and final fifth-fold80.250894 remain unchanged. Delivered ZIPs remain immutable.
- Fivefold best-model delivery in separate immutable submission_calibrated_pair; repeated freshZIP CUDA inference identical across95948testrows; old delivery preserved.
- Method/experiment docs, compactreports, source-only inference package and portable fifth-fold training launcher.
-945 Python AST checks, training/inference help checks;24 inference modules match verified delivery (one trailing-blank cleanup; AST unchanged). No new training for verification.
- Git branch deep_learning published with source/docs/compactresults only. Data,weights,rawpredictions,ZIPs stay local.
- README now explicitly documents fifth-fold-first screening, same-configuration remaining-fold training, OOF versus submission provenance, and the final posterior-router mechanism/result. Documentation update only; training remains stopped.

## Publication and remaining work
Publication branch: origin/deep_learning. Final push is checked against the remote Git reference; use git status/log for commit identity. No more development or training is scheduled. No scientific result is pending; goal82 was not met.

## Future, only if explicitly requested
Explore useful routing mechanisms under fixedfifth strict80.4/max80/18/max2GPU. New deliveries in newdirectories; never overwrite existingversions.

- Localized the architecture illustration to Simplified Chinese using imagegen; README and METHOD now reference calibrated_dual_backbone_zh.png. Original English image retained. Visual text/layout/formulas checked; no model, training or delivered-archive changes.
