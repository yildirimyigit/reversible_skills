#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   scripts/run_triage_all.sh /workspace/data/demos /workspace/results/triage
#
# Notes:
# - Assumes your evaluator is scripts/snapshot_restore.py
# - Runs headless by default (remove --no-headless if you want GUI)
# - Uses conservative settings; adjust as needed

DEMOS_DIR="${1:-/workspace/data/demos}"
OUT_DIR="${2:-/workspace/results/triage}"

EVAL="python scripts/snapshot_restore.py"

mkdir -p "${OUT_DIR}"

SUMMARY_TXT="${OUT_DIR}/summary.txt"
SUMMARY_CSV="${OUT_DIR}/summary.csv"
: > "${SUMMARY_TXT}"
echo "demo_npz,task,verdict,split_kf,split_t" > "${SUMMARY_CSV}"

# Find all demo NPZs for the 6 tasks (adjust patterns if your filenames differ)
mapfile -t DEMOS < <(find "${DEMOS_DIR}" -maxdepth 1 -type f -name "*.npz" | sort)

if [[ ${#DEMOS[@]} -eq 0 ]]; then
  echo "No .npz files found in ${DEMOS_DIR}"
  exit 1
fi

for demo in "${DEMOS[@]}"; do
  base="$(basename "${demo}" .npz)"
  out_json="${OUT_DIR}/triage_${base}.json"
  out_log="${OUT_DIR}/triage_${base}.log"

  echo "=== ${base} ===" | tee -a "${SUMMARY_TXT}"
  echo "demo: ${demo}" | tee -a "${SUMMARY_TXT}"

  # Run triage
  # - n_trials 10, tau 0.8: robust but not too slow
  # - z_success_eps 0.1: matches your current CLI default in the posted code
  # - adjust kp/vmax if needed per task later
  set +e
  ${EVAL} \
    --demo_npz "${demo}" \
    --rollback_eval \
    --n_trials 10 \
    --tau_success 0.8 \
    --z_success_eps 0.1 \
    --kp 6.0 \
    --vmax 1.0 \
    --q_tol_inf 0.01 \
    --max_inner_steps 4 \
    --max_total_steps 2000 \
    --out_json "${out_json}" \
    > "${out_log}" 2>&1
  rc=$?
  set -e

  # Parse verdict from log (your script prints one of these)
  # - reversible: "demo is directly reversible under this controller"
  # - irreversible: "FIRST FAIL: seg ..." and "SPLIT: kf_row=... timestep=..."
  if grep -q "demo is directly reversible" "${out_log}"; then
    verdict="reversible"
    split_kf=""
    split_t=""
  else
    verdict="needs_split"
    split_kf="$(grep -oP 'SPLIT: kf_row=\K[0-9]+' "${out_log}" | tail -n 1 || true)"
    split_t="$(grep -oP 'timestep=\K[0-9]+' "${out_log}" | tail -n 1 || true)"
  fi

  # Try to extract task name from log (it prints "[demo] task=...")
  task="$(grep -oP '^\[demo\] task=\K[^ ]+' "${out_log}" | head -n 1 || true)"

  echo "verdict: ${verdict}  split_kf=${split_kf}  split_t=${split_t}" | tee -a "${SUMMARY_TXT}"
  echo "" | tee -a "${SUMMARY_TXT}"

  # CSV line
  echo "\"${demo}\",\"${task}\",\"${verdict}\",\"${split_kf}\",\"${split_t}\"" >> "${SUMMARY_CSV}"

  # If the evaluator exited nonzero (e.g., SystemExit 4), we still keep logs/json.
  # You can enforce strict failure by uncommenting below:
  # [[ $rc -eq 0 ]] || exit $rc
done

echo "Done."
echo "Summary: ${SUMMARY_TXT}"
echo "CSV: ${SUMMARY_CSV}"
