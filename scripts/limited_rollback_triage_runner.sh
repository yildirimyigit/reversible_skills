set -euo pipefail

DEMOS=/workspace/data/demos
OUT=/workspace/results/triage_smoke
mkdir -p "$OUT"

tasks_reversible=("BlockPyramid" "InsertUsbInComputer" "PhoneOnBase")
tasks_irreversible=("CloseDrawer" "PutUmbrellaInUmbrellaStand" "ScoopWithSpatula")

run_one () {
  local task=$1
  local demo=$2
  local npz="${DEMOS}/${task}_var00_demo$(printf "%04d" "$demo").npz"
  local outdir="${OUT}/${task}"
  mkdir -p "$outdir"
  local outjson="${outdir}/${task}_var00_demo$(printf "%04d" "$demo").json"

  python3 /workspace/scripts/rollback_triage.py \
    --demo_npz "$npz" --task "$task" --variation 0 \
    --out_json "$outjson" \
    --n_rollouts 10 --settle_steps 2 --success_thresh 0.02 \
    --kp 6.0 --vmax 1.0 --min_kf_gap 5 \
    --headless
}

for t in "${tasks_reversible[@]}" "${tasks_irreversible[@]}"; do
  for d in 0 1; do
    run_one "$t" "$d"
  done
done
