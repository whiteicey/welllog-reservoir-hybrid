#!/usr/bin/env bash
# After stack_states and the transductive nets: assemble G (D + stacked gate) and H (G with transductive nets).
set -u
cd <PROJECT_ROOT>
PY=.venv/bin/python
while [ ! -f outputs/stack_states/report.json ]; do sleep 60; done
$PY assemble.py --config assemble_cfg_G.json --out outputs/assembly_G > logs/assembly_G.log 2>&1
echo "[pl2] G done $(date) $(grep TOTAL logs/assembly_G.log)"
while [ ! -f outputs/nn_base_c_pl/report.json ]; do sleep 60; done
$PY assemble.py --config assemble_cfg_H.json --out outputs/assembly_H > logs/assembly_H.log 2>&1
echo "[pl2] H done $(date) $(grep TOTAL logs/assembly_H.log)"
