#!/usr/bin/env bash
# Transductive final pipeline: wait for assembly H; if it beats D by >= 0.05 nested points, train the
# pseudo-labelled nets (80 labelled + 10 pseudo-labelled test wells), (stack if needed), predict, validate, package.
set -u
cd /home/ubuntu/kunlun_welllog_v3_20260919
PY=.venv/bin/python
while [ ! -f outputs/assembly_H/report.json ]; do sleep 120; done
echo "[plfinal] assembly H available $(date)"
$PY make_pl_cfg.py --assembly outputs/assembly_H --reference outputs/assembly_D --min-gain 0.05 > logs/make_pl_cfg.log 2>&1
cat logs/make_pl_cfg.log
if ! grep -q PROCEED logs/make_pl_cfg.log; then echo "[plfinal] no transductive final (gain below threshold) $(date)"; exit 0; fi
if grep -q needs_stack logs/make_pl_cfg.log; then
  $PY train_final.py --config final_cfg_v3_pl.json --out-dir outputs/final_v3_pl --only-stack --skip-nn > logs/train_final_pl_stack.log 2>&1
  echo "[plfinal] stack trained $(date)"
fi
$PY train_final.py --config final_cfg_v3_pl.json --out-dir outputs/final_v3_pl --skip-cb > logs/train_final_pl_nn.log 2>&1
echo "[plfinal] nets trained $(date)"
$PY predict.py --data_dir data --output outputs/final_v3_pl/result.json --model_dir outputs/final_v3_pl/models --metadata outputs/final_v3_pl/model_meta.json > logs/predict_final_pl.log 2>&1
$PY validate_result.py --data_dir data --result outputs/final_v3_pl/result.json --reference outputs/final_v3/result.json > logs/validate_final_pl.log 2>&1
cat logs/validate_final_pl.log
$PY package_submission.py --final-dir outputs/final_v3_pl --out-dir outputs/submission_v3_pl > logs/package_pl.log 2>&1
echo "[plfinal] packaged $(date)"
