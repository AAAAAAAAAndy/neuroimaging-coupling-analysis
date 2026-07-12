#!/bin/bash
#==============================================================================
# data_status.sh — 实时监控各数据子目录处理进度
#==============================================================================
BASE=/mnt/d/project2
DATA=$BASE/data
OUT=$BASE/output

echo "========================================"
echo "  数据目录实时监控 $(date '+%H:%M:%S')"
echo "========================================"
echo ""

printf "%-30s %8s %8s %8s\n" "目录" "原始" "已处理" "完成%"
printf "%-30s %8s %8s %8s\n" "----" "----" "------" "-----"

# 遍历所有数据子目录
for data_dir in "$DATA"/*/; do
    dir_name=$(basename "$data_dir")

    # 计算原始被试数（处理 ASL_special 的嵌套结构）
    if [ "$dir_name" = "baseline_ASL_special" ] || [ "$dir_name" = "visit_ASL_special" ]; then
        n_raw=0
        for subdir in "$data_dir"/*/; do
            count=$(ls "$subdir" 2>/dev/null | grep -E "^(B1_|A1_|sub)" | wc -l)
            n_raw=$((n_raw + count))
        done
        # 输出也在 ASL_special 下
        out_dir="$OUT/$dir_name"
        n_done=0
        if [ -d "$out_dir" ]; then
            for subdir in "$out_dir"/*/; do
                count=$(find "$subdir" -maxdepth 1 -name "*_CBF.nii.gz" 2>/dev/null | wc -l)
                n_done=$((n_done + count))
            done
        fi
    else
        n_raw=$(ls "$data_dir" 2>/dev/null | grep -E "^(B1_|A1_|sub)" | wc -l)
        out_dir="$OUT/$dir_name"
        # 根据模态确定输出文件模式
        case "$dir_name" in
            *_ASL)     pattern="*_CBF.nii.gz" ;;
            *_fMRI)    pattern="*_ALFF.nii.gz" ;;
            *_T1)      pattern="*_T1.nii.gz" ;;
            *_DWI)     pattern="dwi.mif" ;;
            *)         pattern="*.nii.gz" ;;
        esac
        n_done=$(find "$out_dir" -maxdepth 2 -name "$pattern" 2>/dev/null | wc -l)
    fi

    # 计算完成率
    if [ "$n_raw" -gt 0 ]; then
        pct=$((n_done * 100 / n_raw))
    else
        pct=0
    fi

    printf "%-30s %8s %8s %7s%%\n" "$dir_name" "$n_raw" "$n_done" "$pct"
done

echo ""
echo "========================================"
echo "  合计"
echo "========================================"

# 统计总唯一被试
total_raw=$(ls -d "$DATA"/*/ 2>/dev/null | while read d; do
    if [ "$(basename $d)" = "baseline_ASL_special" ]; then
        for sub in "$d"/*/; do ls "$sub" 2>/dev/null; done
    else
        ls "$d" 2>/dev/null
    fi
done | grep -E "^(B1_|A1_|sub)" | sort -u | wc -l)

total_done=$(find "$OUT" -maxdepth 3 -name "*_CBF.nii.gz" -o -name "*_ALFF.nii.gz" -o -name "*_T1.nii.gz" 2>/dev/null | xargs -I {} dirname {} 2>/dev/null | xargs -I {} basename {} 2>/dev/null | sort -u | wc -l)

echo "  总唯一被试: $total_raw"
echo "  已有产物被试: $total_done"
echo ""

echo "========================================"
echo "  运行中的进程"
echo "========================================"
echo "  recon-all: $(pgrep -c 'recon-all' 2>/dev/null || echo 0)"
echo "  process_one.py: $(pgrep -cf 'process_one.py' 2>/dev/null || echo 0)"
echo ""
echo "  Screen sessions:"
screen -ls 2>/dev/null | grep -E "batch|pls" | awk '{print "    " $0}'
