#!/usr/bin/env bash
# Queue 3: reprioritised CatBoost variants (GPU shared with the sequence-net run).
set -u
cd /home/ubuntu/kunlun_welllog_v3_20260919
PY=.venv/bin/python
while [ ! -f outputs/perm_bins60_base/report.json ]; do sleep 30; done
run() { name=$1; shift; echo "[queue] start $name $(date)"; $PY exp_variant.py --name "$name" "$@" > "logs/$name.log" 2>&1; echo "[queue] end $name $(date) $(grep -E 'joint full|Error|error' logs/$name.log | tail -2 | tr '\n' ' ')"; }
run por_bins80_base_extra      --target POR  --model bins --bins 80 --features base+extra
run gate_sw_enh_extra_stack    --target SW   --model gate --iterations 950 --features enh+extra+stack
run gate_por_base_extra        --target POR  --model gate --iterations 500 --features base+extra
run por_bins80_base_4seeds     --target POR  --model bins --bins 80 --features base --seeds 42,2026,7,99
run sw_bins60_enh_extra_stack  --target SW   --model bins --bins 60 --features enh+extra+stack
run gate_perm_base_extra_stack --target PERM --model gate --iterations 800 --features base+extra+stack
run perm_bins60_base_extra_stack --target PERM --model bins --bins 60 --features base+extra+stack
echo "[queue] all done $(date)"
