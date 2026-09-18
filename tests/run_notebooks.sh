#!/usr/bin/env bash
# Execute the notebooks. Run from inside an Anyscale workspace at the repo root.
#   bash tests/run_notebooks.sh full        # every notebook, in place, keeping outputs
#   bash tests/run_notebooks.sh smoke-cpu   # 01-03 on CPU with a data subset, into /tmp
set -euo pipefail
cd "$(dirname "$0")/.."
MODE="${1:-full}"
ALL=(00_before_you_start 01_pytorch_as_is 02_ray_train_distributed 03_ray_tune_with_ray_train
     04_run_as_anyscale_jobs)
SMOKE=(01_pytorch_as_is 02_ray_train_distributed 03_ray_tune_with_ray_train)

execute() {  # $1 stem, $2 output dir ("" = in place)
  local args=(--to notebook --execute --ExecutePreprocessor.timeout=3600
              --ExecutePreprocessor.kernel_name=python3)
  if [ -z "$2" ]; then args+=(--inplace); else args+=(--output-dir "$2"); fi
  echo "=== $(date -u +%H:%M:%S) executing notebooks/$1.ipynb"
  jupyter nbconvert "${args[@]}" "notebooks/$1.ipynb"
}

case "$MODE" in
  full)      for nb in "${ALL[@]}"; do execute "$nb" ""; done ;;
  smoke-cpu) export USE_GPU=0 SMOKE_TEST=1 NUM_WORKERS=2 NUM_SAMPLES=2 MAX_CONCURRENT_TRIALS=1
             mkdir -p /tmp/nb-smoke
             for nb in "${SMOKE[@]}"; do execute "$nb" /tmp/nb-smoke; done ;;
  *)         echo "usage: $0 {full|smoke-cpu}" >&2; exit 2 ;;
esac
echo "=== $(date -u +%H:%M:%S) $MODE finished OK"
