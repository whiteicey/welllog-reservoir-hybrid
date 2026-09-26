# Pure-DL workspace rules

- Goal: maintain reproducible pure neural well-log models predicting POR/PERM/SW. Best complete verified OOF is81.71323318479078;82 target unmet.
- User stop directive: last posterior-router run is FINISHED; start NO new training/continuation unless explicitly requested. Repository cleanup/publication does not authorize more training.
- Keep PROJECTS.md current after each batch; describe completed work, next/pending items and achieved/unachieved objectives.
- Never use test labels, tree predictions, pretrained weights, pseudo-labels, rowwise model selection or OOF-fitted blending. Preserve original groupedfolds/populations/seed/scoring.
- Future trials, only if requested: fixedfifth strict>80.4 before remainingfolds, max80/patience18/min10, actualbestcheckpoint, max2GPU. Historical frozen experiments retain original settings.
- Do not modify frozen experiments while running or silently invalidate recorded hashes. Portable wrappers may redirect paths/admission only and must record it.
- Record each experiment focus, evidence, gain and abandoned directions. See docs/deep_learning and results/deep_learning; no redundant per-trial preflight/review.
- Source and compact summaries only in Git; exclude data, weights, raw predictions, ZIPs, logs and caches. Keep key/milestone models locally; delete redundant failed artifacts rather than move to C.
- Every delivery uses a NEW directory; existing delivered artifacts are immutable.
- Use normal CUDA PyTorch locally. Never stop training merely for disk cleanup.
