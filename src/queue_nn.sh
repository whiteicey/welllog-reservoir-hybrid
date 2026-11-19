#!/usr/bin/env bash
# NN chain: after nn_gate_b finishes, train the variant with v2 base features as input channels.
set -u
cd <PROJECT_ROOT>
PY=.venv/bin/python
while [ ! -f outputs/nn_gate_b/report.json ]; do sleep 60; done
echo "[nnq] nn_gate_b done $(date)"
$PY nn_seq.py --name nn_base_c --use-base 1 --epochs 60 --steps-per-epoch 100 --batch 16 --crop 3072 --channels 160 --bin-weight 1.0 --seeds 1,2 > logs/nn_base_c.log 2>&1
echo "[nnq] nn_base_c done $(date)"
