#!/usr/bin/env bash
# After the final v3 prediction exists: build the submission matrix (slots 2-4) and package each.
set -u
cd <PROJECT_ROOT>
PY=.venv/bin/python
V2=<PROJECT_ROOT>
while [ ! -f outputs/final_v3/result.json ]; do sleep 60; done
while pgrep -f "package_submission.py --final-dir outputs/final_v3" > /dev/null; do sleep 30; done
mkdir -p outputs/ablations
# slot 2: v3 gates + v2 continuous
$PY make_ablations.py --gate-from outputs/final_v3/result.json --cont-from $V2 --output outputs/ablations/slot2_v3gate_v2cont.json --name v3gate_v2cont > logs/ablation_slot2.log 2>&1
# slot 3: v2 gates + v3 continuous
$PY make_ablations.py --gate-from $V2 --cont-from outputs/final_v3/result.json --output outputs/ablations/slot3_v2gate_v3cont.json --name v2gate_v3cont > logs/ablation_slot3.log 2>&1
# slot 4: v3 without sequence nets (assembly E decision) -> needs a metadata copy with E's decision block
$PY - <<PYEOF
import json
m=json.load(open("outputs/final_v3/model_meta.json")); d=json.load(open("outputs/assembly_E/decision.json"))
m["decision"]=d; m["nns"]={}
json.dump(m,open("outputs/final_v3/model_meta_noNN.json","w"),indent=1)
PYEOF
$PY predict.py --data_dir data --output outputs/ablations/slot4_v3_noNN.json --model_dir outputs/final_v3/models --metadata outputs/final_v3/model_meta_noNN.json > logs/ablation_slot4.log 2>&1
for f in slot2_v3gate_v2cont slot3_v2gate_v3cont slot4_v3_noNN; do
  $PY validate_result.py --data_dir data --result outputs/ablations/$f.json --reference outputs/final_v3/result.json >> logs/ablations_validate.log 2>&1
  $PY -c "import zipfile,sys; z=zipfile.ZipFile(\"outputs/ablations/$f.zip\",\"w\",zipfile.ZIP_DEFLATED); z.write(\"outputs/ablations/$f.json\",\"result.json\"); z.close()"
done
cat logs/ablations_validate.log
echo "[ablations] done $(date)"
