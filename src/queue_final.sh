#!/usr/bin/env bash
# Final pipeline: wait for CatBoost full training and the nn_gate_d CV run, then train the final sequence nets,
# refresh the decision block, predict, validate, package.
set -u
cd <PROJECT_ROOT>
PY=.venv/bin/python
while pgrep -f "train_final.py --config final_cfg_v3.json --out-dir outputs/final_v3 --skip-nn" > /dev/null; do sleep 60; done
echo "[final] start nn training $(date)"
$PY train_final.py --config final_cfg_v3.json --out-dir outputs/final_v3 --skip-cb > logs/train_final_nn.log 2>&1
echo "[final] nn done $(date)"
$PY predict.py --data_dir data --output outputs/final_v3/result.json --model_dir outputs/final_v3/models --metadata outputs/final_v3/model_meta.json > logs/predict_final_v3.log 2>&1
echo "[final] predict done $(date)"
$PY validate_result.py --data_dir data --result outputs/final_v3/result.json --reference <PROJECT_ROOT> > logs/validate_final_v3.log 2>&1
cat logs/validate_final_v3.log
$PY package_submission.py --final-dir outputs/final_v3 --out-dir outputs/submission_v3 > logs/package_v3.log 2>&1
echo "[final] packaged $(date)"
