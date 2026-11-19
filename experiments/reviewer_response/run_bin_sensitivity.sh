#!/usr/bin/env bash
set -euo pipefail
ROOT="${WELLLOG_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
WORK=${REVIEW_OUTPUT_ROOT:-$ROOT/outputs/reviewer_response_20261117}
mkdir -p "$WORK/logs"
cd "$ROOT"
for K in 30 120; do
  NAME="review_perm_bins${K}_base"
  OUT="$ROOT/outputs/$NAME"
  if [[ -f "$OUT/oof.npz" && -f "$OUT/report.json" ]]; then
    echo "SKIP_COMPLETE $NAME"
    continue
  fi
  if [[ -e "$OUT" ]]; then
    echo "REFUSE_INCOMPLETE_EXISTING $OUT" >&2
    exit 2
  fi
  echo "START $NAME $(date -u +%FT%TZ)"
  "$ROOT/.venv/bin/python" -u exp_variant.py \
    --name "$NAME" --target PERM --model bins --bins "$K" \
    --features base --iterations 1200 --seeds 42,2026 \
    2>&1 | tee "$WORK/logs/${NAME}.log"
  echo "DONE $NAME $(date -u +%FT%TZ)"
done
printf '{"status":"complete","bins":[30,60,120],"seeds":[42,2026],"folds":5}\n' > "$WORK/bin_sensitivity_status.json"
