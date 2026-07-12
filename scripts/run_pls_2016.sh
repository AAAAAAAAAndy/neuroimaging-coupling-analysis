#!/usr/bin/env bash
#==============================================================================
# run_pls_2016.sh — 论文 2016 SC-FC PLS 批量处理
# 1. 跑 recon-all 构建皮层表面
# 2. 跑 PLS 管道 (DWI + 连接组)
#==============================================================================

set -euo pipefail

BASE=/mnt/d/project2
DATA=$BASE/data
FS_DIR=$BASE/output/freesurfer
MIND=/home/sad/miniconda3/envs/mind/bin/python
LOG_DIR=$BASE/output/batch_logs

export FREESURFER_HOME=/usr/local/freesurfer
export FS_LICENSE=$FREESURFER_HOME/license.txt
export SUBJECTS_DIR=$FS_DIR
export PATH=$FREESURFER_HOME/bin:/usr/local/fsl/bin:$HOME/abin:$PATH
export FSFAST_HOME=$FREESURFER_HOME/fsfast
export FSF_OUTPUT_FORMAT=nii.gz
export FSLDIR=/usr/local/fsl

get_dwi_subjects() {
    comm -12 \
        <(ls "$DATA/baseline_DWI" | sort) \
        <(comm -12 \
            <(ls "$DATA/baseline_fMRI" | sort) \
            <(ls "$DATA/baseline_T1" | sort))
}

echo "========================================"
echo "论文 2016 SC-FC PLS 批量处理 $(date)"
echo "========================================"

subjects=$(get_dwi_subjects)
total=$(echo "$subjects" | wc -l)
echo "DWI+BOLD+T1 重叠: $total 被试"

# Phase 1: Recon-all for DWI subjects without FS surfaces
echo ""
echo "=== Phase 1: FreeSurfer recon-all ==="
count=0
for subj in $subjects; do
    count=$((count + 1))
    sphere="$FS_DIR/$subj/surf/lh.sphere.reg"

    if [ -f "$sphere" ]; then
        echo "[$count/$total] SKIP $subj (FS done)"
        continue
    fi

    # Clean incomplete FS dir
    if [ -d "$FS_DIR/$subj" ] && [ ! -f "$sphere" ]; then
        rm -rf "$FS_DIR/$subj"
    fi

    # Wait for recon slot
    while true; do
        running=$(pgrep -c "recon-all" 2>/dev/null || echo 0)
        [ "$running" -lt 4 ] && break
        sleep 15
    done

    t1="$BASE/output/baseline_T1/$subj/${subj}_T1.nii.gz"
    if [ ! -f "$t1" ]; then
        # Convert T1 first
        mkdir -p "$BASE/output/baseline_T1/$subj"
        echo "  Converting T1 for $subj"
        dcm2niix -z y -f "${subj}_T1" -o "$BASE/output/baseline_T1/$subj" \
            -p n -v 0 "$DATA/baseline_T1/$subj" 2>/dev/null || true
    fi

    if [ -f "$t1" ]; then
        echo "[$count/$total] RECON $subj"
        nohup recon-all -subjid "$subj" -i "$t1" -sd "$FS_DIR" \
            -all -openmp 4 > "$FS_DIR/${subj}_recon.log" 2>&1 &
    else
        echo "[$count/$total] SKIP $subj (no T1)"
    fi
done

echo "等待 recon-all 完成..."
wait
while pgrep -f "recon-all" >/dev/null 2>&1; do sleep 30; done
echo "Recon-all 完成"

# Phase 2: PLS pipeline for each subject
echo ""
echo "=== Phase 2: PLS pipeline ==="
count=0
for subj in $subjects; do
    count=$((count + 1))

    if [ -f "$BASE/output/baseline_DWI/$subj/SC_matrix.npy" ]; then
        echo "[$count/$total] SKIP $subj (SC done)"
        continue
    fi

    if [ ! -f "$FS_DIR/$subj/surf/lh.sphere.reg" ]; then
        echo "[$count/$total] SKIP $subj (no FS)"
        continue
    fi

    while true; do
        running=$(pgrep -cf "process_one.py" 2>/dev/null || echo 0)
        [ "$running" -lt 6 ] && break
        sleep 10
    done

    echo "[$count/$total] PLS $subj"
    "$MIND" "$BASE/scripts/process_one.py" --subject "$subj" --paper 2016 \
        >> "$LOG_DIR/${subj}.log" 2>&1 &
done

echo "等待 PLS 管道完成..."
wait
echo "=== 全部完成 ==="

sc_done=$(ls output/baseline_DWI/*/SC_matrix.npy 2>/dev/null | wc -l)
echo "SC 矩阵完成: $sc_done / $total"
