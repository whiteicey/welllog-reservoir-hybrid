#!/usr/bin/env bash
# NN chain 2: after nn_base_c, train a second gate-oriented net with a wider kernel/longer crop for diversity.
set -u
cd <PROJECT_ROOT>
PY=.venv/bin/python
while pgrep -f "bash queue_nn.sh" > /dev/null; do sleep 60; done
echo "[nnq2] start nn_gate_d $(date)"
$PY nn_seq.py --name nn_gate_d --epochs 60 --steps-per-epoch 100 --batch 12 --crop 4096 --channels 160 --kernel 9 --bin-weight 0.3 --seeds 11,12,13 > logs/nn_gate_d.log 2>&1
echo "[nnq2] nn_gate_d done $(date)"
