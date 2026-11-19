#!/usr/bin/env bash
# Post-run: when every remaining experiment is finished, assemble F (all sources) and refresh the summary tables.
set -u
cd <PROJECT_ROOT>
PY=.venv/bin/python
while pgrep -f "bash queue3.sh|bash queue4.sh|bash queue_nn2.sh" > /dev/null; do sleep 120; done
echo "[post] all experiment queues finished $(date)"
# drop sources whose outputs do not exist (failed runs) before assembling
$PY - <<PYEOF
import json, os
c=json.load(open("assemble_cfg_F.json"))
for t in c:
    for k in ("dist_sources","gate_sources"):
        c[t][k]=[s for s in c[t][k] if os.path.exists(s["oof"]) and os.path.exists(s["test"])]
json.dump(c,open("assemble_cfg_F_avail.json","w"),indent=1)
print({t:(len(c[t]["dist_sources"]),len(c[t]["gate_sources"])) for t in c})
PYEOF
$PY assemble.py --config assemble_cfg_F_avail.json --out outputs/assembly_F > logs/assembly_F.log 2>&1
grep -v "Warning\|^ " logs/assembly_F.log | tail -5
$PY collect_results.py > logs/collect_results.log 2>&1
echo "[post] done $(date)"
