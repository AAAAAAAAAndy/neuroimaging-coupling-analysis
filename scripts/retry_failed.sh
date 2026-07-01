#!/bin/bash
#==============================================================================
# retry_failed.sh — 清理失败被试的中间产物并标记重试
#
# 用法:
#   bash scripts/retry_failed.sh              # 清理所有失败被试
#   bash scripts/retry_failed.sh --dry-run    # 仅显示会清理的内容
#   bash scripts/retry_failed.sh --subject B1_0024  # 清理指定被试
#
# 清理范围: 删除不完整的中间产物（已成功完成的步骤产物保留）
#==============================================================================

set -euo pipefail

BASE=/mnt/d/project2
OUT=$BASE/output
LOG_DIR=$OUT/batch_logs
FS_DIR=$OUT/freesurfer

DRY_RUN=false
TARGET_SUBJ=""

# 解析参数
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)   DRY_RUN=true; shift ;;
        --subject)   TARGET_SUBJ="$2"; shift 2 ;;
        *)           echo "未知参数: $1"; exit 1 ;;
    esac
done

get_failed_subjects() {
    if [ -n "$TARGET_SUBJ" ]; then
        echo "$TARGET_SUBJ"
        return
    fi
    for logfile in "$LOG_DIR"/*.log; do
        [ -f "$logfile" ] || continue
        local subj
        subj=$(basename "$logfile" .log)
        if grep -q "\[FAIL\]" "$logfile" 2>/dev/null && ! grep -q "\[DONE\]" "$logfile" 2>/dev/null; then
            echo "$subj"
        fi
    done
}

clean_subject() {
    local subj=$1
    local cleaned=0

    echo "--- $subj ---"

    # 确定失败发生在哪一步
    local last_step=""
    if [ -f "$LOG_DIR/${subj}.log" ]; then
        last_step=$(grep -oP '\[step_\S+\]' "$LOG_DIR/${subj}.log" 2>/dev/null | tail -1 | tr -d '[]')
    fi
    echo "  最后步骤: ${last_step:-未知}"

    # 按步骤清理（只清理失败步骤及之后的产物）
    # Step 11: 耦合
    if [ "$last_step" = "step_11" ] || [ -z "$last_step" ]; then
        for f in "$OUT/baseline_fMRI/$subj/coupling_lh.npy" \
                 "$OUT/baseline_fMRI/$subj/coupling_rh.npy"; do
            if [ -f "$f" ]; then
                $DRY_RUN || rm -f "$f"
                echo "  清理: $f"
                cleaned=$((cleaned + 1))
            fi
        done
    fi

    # Step 10: 表面投射
    if [ "$last_step" = "step_10" ] || [ "$last_step" = "step_11" ] || [ -z "$last_step" ]; then
        for f in "$OUT/baseline_T1/$subj"/surface/fsaverage5_*.mgh; do
            if [ -f "$f" ]; then
                $DRY_RUN || rm -f "$f"
                echo "  清理: $(basename "$f")"
                cleaned=$((cleaned + 1))
            fi
        done
    fi

    # Step 5-9: ALFF + 中间产物
    if [ "$last_step" = "step_5_9" ] || [ "$last_step" = "step_10" ] || \
       [ "$last_step" = "step_11" ] || [ -z "$last_step" ]; then
        for f in "$OUT/baseline_fMRI/$subj/${subj}_ALFF.nii.gz" \
                 "$OUT/baseline_fMRI/$subj/${subj}_BOLD_preproc.nii.gz" \
                 "$OUT/baseline_fMRI/$subj/${subj}_BOLD_mc.nii.gz" \
                 "$OUT/baseline_fMRI/$subj/${subj}_BOLD_mc_despike.nii.gz" \
                 "$OUT/baseline_fMRI/$subj/${subj}_BOLD_brain.nii.gz" \
                 "$OUT/baseline_fMRI/$subj/_motion_params.npy" \
                 "$OUT/baseline_fMRI/$subj/_brainmask_bold.nii.gz" \
                 "$OUT/baseline_fMRI/$subj/_aseg_bold.nii.gz" \
                 "$OUT/baseline_fMRI/$subj/_wm_mask.nii.gz" \
                 "$OUT/baseline_fMRI/$subj/_csf_mask.nii.gz" \
                 "$OUT/baseline_fMRI/$subj/_mc_ref.nii.gz" \
                 "$OUT/baseline_fMRI/$subj/_mc_vol_"*.nii.gz \
                 "$OUT/baseline_fMRI/$subj/_mc_vol_"*.mat; do
            if [ -f "$f" ]; then
                $DRY_RUN || rm -f "$f"
                echo "  清理: $(basename "$f")"
                cleaned=$((cleaned + 1))
            fi
        done
    fi

    # Step 4: BOLD NIfTI
    if [ "$last_step" = "step_4" ] || [ "$last_step" = "step_5_9" ] || \
       [ "$last_step" = "step_10" ] || [ "$last_step" = "step_11" ] || [ -z "$last_step" ]; then
        local bold="$OUT/baseline_fMRI/$subj/${subj}_BOLD.nii.gz"
        if [ -f "$bold" ]; then
            $DRY_RUN || rm -f "$bold"
            echo "  清理: $(basename "$bold")"
            cleaned=$((cleaned + 1))
        fi
    fi

    # Step 3: 不完整的 FreeSurfer（只在失败时清理）
    if [ -d "$FS_DIR/$subj" ] && [ ! -f "$FS_DIR/$subj/surf/lh.sphere.reg" ]; then
        $DRY_RUN || rm -rf "$FS_DIR/$subj"
        echo "  清理: freesurfer/$subj/ (不完整)"
        cleaned=$((cleaned + 1))
    fi

    # Step 1: CBF（通常不会失败，但以防万一）
    # Step 2: T1（通常不会失败）
    # 保留这些，因为它们很少失败

    # 清理日志
    if [ -f "$LOG_DIR/${subj}.log" ]; then
        $DRY_RUN || rm -f "$LOG_DIR/${subj}.log"
        echo "  清理: 日志"
        cleaned=$((cleaned + 1))
    fi

    echo "  清理了 $cleaned 个文件"
}

# ---- 主流程 ----
echo "=== 重试失败任务 ==="
echo "模式: $([ "$DRY_RUN" = true ] && echo '预览 (dry-run)' || echo '实际清理')"
echo ""

total=0
while read -r subj; do
    [ -z "$subj" ] && continue
    total=$((total + 1))
    clean_subject "$subj"
done < <(get_failed_subjects)

echo ""
echo "共处理 $total 个失败被试"
if [ "$DRY_RUN" = true ]; then
    echo "以上为预览，实际清理请去掉 --dry-run"
else
    echo "清理完成，请运行: bash scripts/run_batch.sh"
fi
