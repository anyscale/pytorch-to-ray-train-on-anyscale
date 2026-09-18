#!/usr/bin/env bash
# Submit the curriculum scripts as Anyscale Jobs IN PARALLEL and wait for all.
# Each job gets its own cluster, so they run concurrently rather than queueing.
#   bash tests/run_jobs.sh              # all three at once
#   bash tests/run_jobs.sh 02 03        # only the configs whose filename contains these
set -uo pipefail
cd "$(dirname "$0")/.."
CONFIGS=(jobs/job_01_pytorch.yaml jobs/job_02_ray_train.yaml jobs/job_03_tune.yaml)
NAMES=(pytorch-01 ray-train-02 tune-03)

selected=()
for i in "${!CONFIGS[@]}"; do
  if [ $# -gt 0 ]; then
    match=0
    for want in "$@"; do [[ "${CONFIGS[$i]}" == *"$want"* ]] && match=1; done
    [ $match -eq 1 ] || continue
  fi
  selected+=("$i")
done

# Submit without --wait: each call returns as soon as the job is accepted. Collect every
# failure instead of stopping at the first, same as the wait loop below.
fail=0
for i in "${selected[@]}"; do
  echo "=== $(date -u +%H:%M:%S) submitting ${CONFIGS[$i]}"
  anyscale job submit -f "${CONFIGS[$i]}" --working-dir . || { echo "SUBMIT FAILED ${CONFIGS[$i]}"; fail=1; }
done

if [ $fail -ne 0 ]; then
  echo "At least one submission failed. Not waiting on jobs that never started."
  exit 1
fi

# Then wait on each. They have been running concurrently the whole time.
fail=0
for i in "${selected[@]}"; do
  name=${NAMES[$i]}
  if anyscale job wait -n "$name" --state SUCCEEDED --timeout-s 5400; then
    echo "OK       $name"
  else
    echo "FAILED   $name"
    fail=1
  fi
done

if [ $fail -ne 0 ]; then
  echo "At least one job failed. Diagnose with the inspect skill, or:"
  echo "  anyscale job logs -n <name> --tail --max-lines 200"
  exit 1
fi
echo "=== $(date -u +%H:%M:%S) all requested jobs SUCCEEDED"
