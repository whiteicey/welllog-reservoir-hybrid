# Defaults for future experiments

New trainers use this budget. Historical and active source files remain frozen.

```python
cfg["epochs"] = 80
cfg["early_stopping"].update(patience=18, minimum_epochs=10, maximum_epochs=80)
# Keep64 steps per epoch; derive cosine horizon from the actual config.
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=cfg["epochs"] * cfg["steps"], eta_min=1e-5
)
```

The training loop reads cfg["epochs"]. Restore the actual best validation checkpoint.
Other settings follow the experiment's direct control; record budget and source hashes in config/README.
No per-trial independent preflight/review. Direct fixed fifth screening, then necessary promotion checks and otherfolds only after strict>80.4. FullOOF>82 remains the final goal.
