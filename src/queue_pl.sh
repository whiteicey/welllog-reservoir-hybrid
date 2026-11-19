#!/usr/bin/env bash
# Transductive (pseudo-label) sequence-net CV + stacked level-2 state model.
set -u
cd <PROJECT_ROOT>
PY=.venv/bin/python
$PY stack_states.py --name stack_states > logs/stack_states.log 2>&1 &
$PY nn_seq.py --name nn_gate_b_pl --pseudo-oof outputs/assembly_D/final_oof.npy --pseudo-weight 0.5 --epochs 60 --steps-per-epoch 100 --batch 16 --crop 3072 --channels 160 --bin-weight 0.3 --seeds 1,2,3 > logs/nn_gate_b_pl.log 2>&1
wait
echo "[pl] stack + nn_gate_b_pl done $(date)"
$PY nn_seq.py --name nn_base_c_pl --use-base 1 --pseudo-oof outputs/assembly_D/final_oof.npy --pseudo-weight 0.5 --epochs 60 --steps-per-epoch 100 --batch 16 --crop 3072 --channels 160 --bin-weight 1.0 --seeds 1,2 > logs/nn_base_c_pl.log 2>&1
echo "[pl] nn_base_c_pl done $(date)"
