#!/usr/bin/env bash
# DTW alignment experiment chain: similarity + alignment features (CPU) -> type-well diagnostic (CPU, background)
# -> CatBoost variants with +aln (GPU) -> assembly I (D sources + aln sources).
set -u
cd /home/ubuntu/kunlun_welllog_v3_20260919
PY=.venv/bin/python
$PY dtw_align.py > logs/dtw_align.log 2>&1 || { echo "[aln] dtw_align failed"; exit 1; }
echo "[aln] alignment features done $(date)"
nohup $PY diag_typewell.py > logs/diag_typewell.log 2>&1 &
run() { name=$1; shift; echo "[aln] start $name $(date)"; $PY exp_variant.py --name "$name" "$@" > "logs/$name.log" 2>&1; echo "[aln] end $name $(date) $(grep -E 'joint full|Error|error' logs/$name.log | tail -2 | tr '\n' ' ')"; }
run por_bins80_base_extra_aln --target POR --model bins --bins 80 --features base+extra+aln
run sw_bins60_enh_extra_aln   --target SW  --model bins --bins 60 --features enh+extra+aln
$PY exp_states.py --name states_base_extra_aln --features base+extra+aln > logs/states_base_extra_aln.log 2>&1
echo "[aln] states done $(date) $(grep -v Warning logs/states_base_extra_aln.log | grep raw | tail -1)"
run gate_sw_enh_extra_aln     --target SW  --model gate --iterations 950 --features enh+extra+aln
run gate_por_base_extra_aln   --target POR --model gate --iterations 500 --features base+extra+aln
run perm_bins60_base_aln      --target PERM --model bins --bins 60 --features base+aln
$PY assemble.py --config assemble_cfg_I.json --out outputs/assembly_I > logs/assembly_I.log 2>&1
echo "[aln] assembly I $(date) $(grep TOTAL logs/assembly_I.log)"
