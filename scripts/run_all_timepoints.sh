#!/bin/bash
#==============================================================================
# run_all_timepoints.sh — 处理 baseline + visit 所有数据
#
# 用法:
#   bash scripts/run_all_timepoints.sh            # 跑全量
#   bash scripts/run_all_timepoints.sh baseline    # 只跑 baseline
#   bash scripts/run_all_timepoints.sh visit       # 只跑 visit
#==============================================================================

set -euo pipefail

TIMEPOINT="${1:-all}"
BASE=/mnt/d/project2

run_batch() {
    local tp=$1
    echo ""
    echo "========================================"
    echo "处理 $tp 数据 $(date)"
    echo "========================================"

    mkdir -p "$BASE/output/${tp}_ASL" "$BASE/output/${tp}_T1" \
             "$BASE/output/${tp}_fMRI" "$BASE/output/${tp}_DWI"

    TIMEPOINT="$tp" bash "$BASE/scripts/run_batch.sh" 2>&1 | tee "$BASE/output/${tp}_batch.log"

    echo "$tp 完成"
}

case "$TIMEPOINT" in
    baseline) run_batch baseline ;;
    visit)    run_batch visit ;;
    all)
        run_batch baseline
        run_batch visit
        ;;
    *)
        echo "用法: $0 [baseline|visit|all]"
        exit 1
        ;;
esac

echo ""
echo "=== 全部 timepoint 完成 ==="
echo "Baseline: $(ls $BASE/output/baseline_fMRI/*/coupling_lh.npy 2>/dev/null | wc -l) 被试"
echo "Visit:    $(ls $BASE/output/visit_fMRI/*/coupling_lh.npy 2>/dev/null | wc -l) 被试"
