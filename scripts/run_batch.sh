#!/bin/bash
#==============================================================================
# run_batch.sh — 批量预处理（断点续传 + 信号处理 + 步骤编号）
#
# 用法:
#   screen -dmS batch bash scripts/run_batch.sh          # 后台启动
#   screen -r batch                                       # 查看实时日志
#   bash scripts/run_batch.sh --status                    # 另一终端查进度
#   bash scripts/run_batch.sh --step-status B1_0024       # 查单个被试步骤状态
#   Ctrl+C / kill                                         → 自动清理子进程
#
# 断点续传: 已完成的步骤自动跳过（检查输出文件是否存在）
# 信号处理: trap SIGINT/SIGTERM → kill 子进程 → 退出
#==============================================================================

set -euo pipefail

BASE=/mnt/d/project2
MIND=/home/sad/miniconda3/envs/mind/bin/python
MAX_PARALLEL=6
MAX_RECON=4

# ---- Timepoint: baseline or visit ----
TIMEPOINT="${TIMEPOINT:-baseline}"
DATA=$BASE/data
FS_DIR=$BASE/output/freesurfer
LOG_DIR=$BASE/output/batch_logs

# Modality-specific data/data prefixes
ASL_PREFIX=${TIMEPOINT}_ASL
BOLD_PREFIX=${TIMEPOINT}_fMRI
T1_PREFIX=${TIMEPOINT}_T1
DWI_PREFIX=${TIMEPOINT}_DWI

OUT_ASL=$BASE/output/${TIMEPOINT}_ASL
OUT_T1=$BASE/output/${TIMEPOINT}_T1
OUT_FMRI=$BASE/output/${TIMEPOINT}_fMRI
OUT_DWI=$BASE/output/${TIMEPOINT}_DWI

export FREESURFER_HOME=/usr/local/freesurfer
export FS_LICENSE=$FREESURFER_HOME/license.txt
export SUBJECTS_DIR=$FS_DIR
export PATH=$FREESURFER_HOME/bin:/usr/local/fsl/bin:$HOME/abin:$PATH
export FSFAST_HOME=$FREESURFER_HOME/fsfast
export FSF_OUTPUT_FORMAT=nii.gz
export FSLDIR=/usr/local/fsl

mkdir -p "$LOG_DIR" "$OUT_ASL" "$OUT_T1" "$OUT_FMRI"

# ---- 子进程跟踪（用于信号清理）----
CHILD_PIDS=()

cleanup() {
    echo ""
    echo "[$(date '+%H:%M:%S')] 收到终止信号，清理子进程..."
    for pid in "${CHILD_PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null
    echo "[$(date '+%H:%M:%S')] 已清理，退出。"
    exit 0
}
trap cleanup SIGINT SIGTERM

# ---- 受试者列表（所有模态并集）----
get_subjects() {
    # Collect all unique subject IDs from all modality directories
    # Handles: baseline_ASL/, baseline_ASL_special/*/, visit_*/
    {
        # Standard baseline modalities
        ls "$DATA/${TIMEPOINT}_ASL" 2>/dev/null
        ls "$DATA/${TIMEPOINT}_fMRI" 2>/dev/null
        ls "$DATA/${TIMEPOINT}_T1" 2>/dev/null
        ls "$DATA/${TIMEPOINT}_DWI" 2>/dev/null

        # ASL_special: deeper nesting (e.g., ASL_3D_tra_M0/A1_0460/)
        if [ -d "$DATA/${TIMEPOINT}_ASL_special" ]; then
            for subdir in "$DATA/${TIMEPOINT}_ASL_special"/*/; do
                ls "$subdir" 2>/dev/null
            done
        fi
    } | grep -E "^(B1_|A1_|sub)" | sort -u
}

# ---- 步骤状态检查 ----
step_done() {
    local subj=$1 step=$2
    case "$step" in
        step_1)   [ -f "$OUT_ASL/$subj/${subj}_CBF.nii.gz" ] ;;
        step_2)   [ -f "$OUT_T1/$subj/${subj}_T1.nii.gz" ] ;;
        step_3)   [ -f "$FS_DIR/$subj/surf/lh.sphere.reg" ] ;;
        step_4)   [ -f "$OUT_FMRI/$subj/${subj}_BOLD.nii.gz" ] ;;
        step_5_9) [ -f "$OUT_FMRI/$subj/${subj}_ALFF.nii.gz" ] ;;
        step_10)  [ -f "$OUT_T1/$subj/surface/fsaverage5_${subj}_cbf_lh.mgh" ] ;;
        step_11)  [ -f "$OUT_FMRI/$subj/coupling_lh.npy" ] ;;
        *)        return 1 ;;
    esac
}

is_complete() {
    local subj=$1
    step_done "$subj" step_1 && \
    step_done "$subj" step_2 && \
    step_done "$subj" step_3 && \
    step_done "$subj" step_4 && \
    step_done "$subj" step_5_9 && \
    step_done "$subj" step_10 && \
    step_done "$subj" step_11
}

is_recon_complete() {
    [ -f "$FS_DIR/$1/surf/lh.sphere.reg" ]
}

# ---- 确保 T1 NIfTI 存在 ----
ensure_t1_nifti() {
    local subj=$1
    local t1="$OUT_T1/$subj/${subj}_T1.nii.gz"
    [ -f "$t1" ] && return 0

    mkdir -p "$OUT_T1/$subj"
    local src="$DATA/$T1_PREFIX/$subj"
    [ ! -d "$src" ] && return 1

    dcm2niix -z y -f "${subj}_T1" -o "$OUT_T1/$subj" \
        -p n -v 0 "$src" 2>/dev/null || true
    [ -f "$t1" ]
}

# ---- 单个被试处理 ----
process_subject() {
    local subj=$1
    local logfile="$LOG_DIR/${subj}.log"

    echo "[$(date '+%H:%M:%S')] [START] $subj" >> "$logfile"

    # 确保 T1 NIfTI 存在
    ensure_t1_nifti "$subj" || { echo "NO T1" >> "$logfile"; return 1; }

    # 等待 recon-all 完成（需要 FS 表面做 surface projection）
    echo "[$(date '+%H:%M:%S')] [wait] 等待 recon-all..." >> "$logfile"
    while ! is_recon_complete "$subj"; do
        sleep 30
    done

    # 运行主管道
    "$MIND" "$BASE/scripts/process_one.py" --subject "$subj" >> "$logfile" 2>&1
    local rc=$?

    if [ $rc -eq 0 ]; then
        echo "[$(date '+%H:%M:%S')] [DONE] $subj" >> "$logfile"
    else
        echo "[$(date '+%H:%M:%S')] [FAIL] rc=$rc" >> "$logfile"
    fi
    return $rc
}

# ---- 启动 recon-all ----
start_recon() {
    local subj=$1
    is_recon_complete "$subj" && return 0
    local t1="$OUT_T1/$subj/${subj}_T1.nii.gz"
    [ ! -f "$t1" ] && return 0

    # 清理不完整的 freesurfer 目录
    [ -d "$FS_DIR/$subj" ] && [ ! -f "$FS_DIR/$subj/surf/lh.sphere.reg" ] && \
        rm -rf "$FS_DIR/$subj"

    nohup recon-all -subjid "$subj" -i "$t1" -sd "$FS_DIR" \
        -all -openmp 4 > "$FS_DIR/${subj}_recon.log" 2>&1 &
    echo "[$(date '+%H:%M:%S')] [RECON] $subj (PID $!)"
}

# ---- 单被试步骤状态 ----
show_step_status() {
    local subj=$1
    echo "===== $subj ====="
    for step in step_1 step_2 step_3 step_4 step_5_9 step_10 step_11; do
        local label
        case "$step" in
            step_1)   label="ASL→CBF" ;;
            step_2)   label="T1→NIfTI" ;;
            step_3)   label="recon-all" ;;
            step_4)   label="BOLD→NIfTI" ;;
            step_5_9) label="BOLD→ALFF" ;;
            step_10)  label="Surface投影" ;;
            step_11)  label="耦合计算" ;;
        esac
        if step_done "$subj" "$step"; then
            printf "  %-12s ✅ %s\n" "$step" "$label"
        else
            printf "  %-12s ⏳ %s\n" "$step" "$label"
        fi
    done
}

# ---- 全局状态报告 ----
show_status() {
    local total=0 complete=0 recon_done=0 recon_running=0 failed=0

    while read -r subj; do
        [ -z "$subj" ] && continue
        total=$((total + 1))
        if is_complete "$subj"; then
            complete=$((complete + 1))
        else
            [ -f "$LOG_DIR/${subj}.log" ] && grep -q "\[FAIL\]" "$LOG_DIR/${subj}.log" 2>/dev/null && \
                failed=$((failed + 1))
        fi
        is_recon_complete "$subj" && recon_done=$((recon_done + 1))
    done < <(get_subjects)

    recon_running=$(pgrep -f "recon-all" 2>/dev/null | wc -l)
    local running=$(pgrep -f "process_one.py" 2>/dev/null | wc -l)

    echo "=============================="
    echo "进度报告 $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=============================="
    echo "受试者总数:     $total"
    echo "预处理完成:     $complete ($(( complete * 100 / (total + 1) ))%)"
    echo "Recon-all 完成: $recon_done"
    echo "Recon-all 运行: $recon_running"
    echo "预处理失败:     $failed"
    echo "预处理运行:     $running"
    echo "可用内存:       $(free -g | awk '/Mem:/{print $7}')GB"
    echo "=============================="

    echo ""
    echo "前 10 个受试者:"
    local i=0
    while read -r subj; do
        [ -z "$subj" ] && continue
        i=$((i + 1))
        [ $i -gt 10 ] && break

        local p2022="⏳"
        local fs_status="⏳"
        if is_complete "$subj"; then
            p2022="✅"
        elif [ -f "$LOG_DIR/${subj}.log" ] && grep -q "\[FAIL\]" "$LOG_DIR/${subj}.log" 2>/dev/null; then
            p2022="❌"
        elif pgrep -f "process_one.py.*$subj" >/dev/null 2>&1; then
            p2022="🔄"
        fi
        is_recon_complete "$subj" && fs_status="✅" || {
            pgrep -f "recon-all.*$subj" >/dev/null 2>&1 && fs_status="🔄"
        }
        printf "  %-12s 预处理:%-2s  FreeSurfer:%-2s\n" "$subj" "$p2022" "$fs_status"
    done < <(get_subjects)
}

# ---- 主循环（交错模式：recon-all 与处理并行） ----
main() {
    echo "========================================"
    echo "批量预处理启动 $(date)"
    echo "========================================"

    local subjects
    subjects=$(get_subjects)
    local total=$(echo "$subjects" | wc -l)
    echo "共 $total 个受试者"

    # Build queue: subjects that need processing
    local queue=()
    for subj in $subjects; do
        is_complete "$subj" && continue
        queue+=("$subj")
    done
    local qtotal=${#queue[@]}
    echo "待处理: $qtotal 个被试"

    local idx=0
    local done=0
    # ProcessId → Subject mapping for process_one jobs
    declare -A PO_SUBJ

    while [ $done -lt $qtotal ]; do
        # 1. Launch recon-all jobs (up to MAX_RECON) for subjects that need it
        local recon_running=$(pgrep -c "recon-all" 2>/dev/null || echo 0)
        local i=$idx
        while [ $i -lt $qtotal ] && [ $recon_running -lt $MAX_RECON ]; do
            local subj="${queue[$i]}"
            is_recon_complete "$subj" && { i=$((i+1)); continue; }

            # Ensure T1 NIfTI exists
            ensure_t1_nifti "$subj" || { echo "[$i/$qtotal] NO_T1 $subj"; i=$((i+1)); continue; }

            local t1="$OUT_T1/$subj/${subj}_T1.nii.gz"
            [ ! -f "$t1" ] && { i=$((i+1)); continue; }

            # Clean incomplete directories
            [ -d "$FS_DIR/$subj" ] && [ ! -f "$FS_DIR/$subj/surf/lh.sphere.reg" ] && \
                rm -rf "$FS_DIR/$subj"

            nohup recon-all -subjid "$subj" -i "$t1" -sd "$FS_DIR" \
                -all -openmp 4 > "$FS_DIR/${subj}_recon.log" 2>&1 &
            echo "[$(date '+%H:%M:%S')] [RECON] $subj"
            recon_running=$((recon_running+1))
            i=$((i+1))
        done

        # 2. Launch process_one jobs (up to MAX_PARALLEL) for subjects with recon-all done
        local po_running=$(pgrep -f "process_one.py" 2>/dev/null | wc -l)
        for subj in "${queue[@]}"; do
            [ $po_running -ge $MAX_PARALLEL ] && break
            is_complete "$subj" && continue
            is_recon_complete "$subj" || continue
            pgrep -f "process_one.py.*$subj" >/dev/null 2>&1 && continue

            echo "[$(date '+%H:%M:%S')] [PROC] $subj"
            process_subject "$subj" >> "$LOG_DIR/${subj}.log" 2>&1 &
            CHILD_PIDS+=($!)
            PO_SUBJ[$!]="$subj"
            po_running=$((po_running+1))
        done

        # 3. Wait briefly and check progress
        sleep 60

        # Count completed
        done=0
        for subj in "${queue[@]}"; do
            is_complete "$subj" && done=$((done+1))
        done

        local recon_r=$(pgrep -c "recon-all" 2>/dev/null || echo 0)
        local proc_r=$(pgrep -f "process_one.py" 2>/dev/null | wc -l)
        echo "[$(date '+%H:%M:%S')] 进度: $done/$qtotal done | recon-all: $recon_r | proc: $proc_r"
    done

    echo "=== 预处理完成 ==="
    show_status
}

# ---- 入口 ----
case "${1:-}" in
    --status|-s)       show_status ;;
    --step-status|-ss) show_step_status "${2:?需要被试ID}" ;;
    --help|-h)         echo "用法: $0 [--status | --step-status <subj>]" ;;
    *)                 main ;;
esac
