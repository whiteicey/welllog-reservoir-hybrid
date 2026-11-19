#!/usr/bin/env bash
# Queue 4 (trimmed): only the two variants with a plausible payoff.
set -u
cd <PROJECT_ROOT>
PY=.venv/bin/python
while pgrep -f "bash queue3.sh" > /dev/null; do sleep 60; done
run() { name=$1; shift; echo "[queue] start $name $(date)"; $PY exp_variant.py --name "$name" "$@" > "logs/$name.log" 2>&1; echo "[queue] end $name $(date) $(grep -E 'joint full|Error|error' logs/$name.log | tail -2 | tr '\n' ' ')"; }
run sw_bins60_enh_extra_stack_4seeds --target SW --model bins --bins 60 --features enh+extra+stack --seeds 42,2026,7,99
run perm_bins100_base_extra_stack --target PERM --model bins --bins 100 --features base+extra+stack
echo "[queue] all done $(date)"
