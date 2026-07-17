#!/bin/bash
# monitor_realtime.sh — 实时监控 + 论文复现度评估
BASE=/mnt/d/project2

while true; do
    clear
    echo "========================================"
    echo "  实时监控 $(date '+%H:%M:%S')"
    echo "========================================"
    echo ""

    # 进程状态
    echo "--- 进程 ---"
    echo "  recon-all:     $(pgrep -c 'recon-all' 2>/dev/null || echo 0)"
    echo "  process_one:   $(pgrep -cf 'process_one.py' 2>/dev/null || echo 0)"
    echo "  dcm2niix:      $(pgrep -c 'dcm2niix' 2>/dev/null || echo 0)"
    echo "  free mem:      $(free -g | awk '/Mem:/{print $7}')GB"
    echo ""

    # 各目录进度
    echo "--- 数据目录处理进度 ---"
    printf "%-30s %7s %7s %6s\n" "目录" "原始" "产物" "%"
    printf "%-30s %7s %7s %6s\n" "----" "----" "----" "---"

    for dir in baseline_ASL baseline_ASL_special baseline_DWI baseline_T1 baseline_fMRI visit_ASL visit_DWI visit_T1 visit_fMRI; do
        raw=$(ls "$BASE/data/$dir/" 2>/dev/null | grep -E "^(B1_|A1_|sub)" | wc -l)
        case "$dir" in
            *_ASL|*_ASL_special) pat="*_CBF.nii.gz";;
            *_DWI) pat="dwi.mif";;
            *_T1) pat="*_T1.nii.gz";;
            *_fMRI) pat="*_ALFF.nii.gz";;
        esac
        done=$(find "$BASE/output/$dir" -maxdepth 3 -name "$pat" 2>/dev/null | wc -l)
        [ "$raw" -gt 0 ] && pct=$((done*100/raw)) || pct=0
        printf "%-30s %7s %7s %5s%%\n" "$dir" "$raw" "$done" "$pct"
    done

    echo ""
    echo "========================================"
    echo "  论文复现度评估"
    echo "========================================"
    echo ""
    echo "论文 2022 (CBF-ALFF 耦合):"
    echo "  [x] ASL->CBF (3 种格式)"
    echo "  [x] T1->NIfTI + recon-all"
    echo "  [x] BOLD->NIfTI + mcflirt 运动校正"
    echo "  [x] 3dDespike 异常值去除"
    echo "  [x] 脑掩膜 + WM/CSF 分割 (Python affine resampling)"
    echo "  [x] 36 参数混杂回归"
    echo "  [x] 0.01-0.08 Hz 带通滤波"
    echo "  [x] ALFF 计算"
    echo "  [x] 皮层表面投射 (bbregister + vol2surf)"
    echo "  [x] 15mm FWHM 局部加权回归耦合"
    echo "  [ ] GAM 组分析 + Spin Test (产后分析)"
    echo ""
    echo "论文 2016 (SC-FC PLS 耦合):"
    echo "  [x] DWI->MIF (mrconvert)"
    echo "  [x] 响应函数 + FOD (dwi2response/dwi2fod)"
    echo "  [x] 确定性纤维追踪 (tckgen SD_STREAM)"
    echo "  [x] 114 区 SC 矩阵 (tck2connectome)"
    echo "  [ ] PLS SVD + 置换检验 + Bootstrap (产后分析)"
    echo ""
    echo "按 Ctrl+C 退出监控"

    sleep 30
done
