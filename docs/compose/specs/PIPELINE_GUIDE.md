# 批量预处理操作指南 / Batch Processing Guide

## 快速开始

```bash
# 1. 激活 mind 环境
conda activate mind

# 2. 查看当前进度
bash scripts/run_batch.sh --status

# 3. 批量预处理（断点续传，随时可中断重启）
bash scripts/run_batch.sh

# 4. 中断后恢复 → 直接重新运行
bash scripts/run_batch.sh
```

## 输出结构

```
derivatives/
├── paper2022/
│   ├── cbf_native/<subj>/<subj>_CBF.nii.gz    # ASL 衍生 CBF
│   ├── nifti/<subj>/<subj>_T1.nii.gz          # T1 结构像
│   ├── nifti/<subj>/<subj>_BOLD.nii.gz        # BOLD 4D
│   ├── alff/<subj>/<subj>_ALFF.nii.gz          # ALFF 图
│   ├── surface/<subj>/fsaverage5_*_cbf_*.mgh   # 表面投射 CBF
│   ├── surface/<subj>/fsaverage5_*_alff_*.mgh  # 表面投射 ALFF
│   ├── coupling/<subj>/coupling_lh.npy        # 耦合图 LH
│   └── coupling/<subj>/coupling_rh.npy        # 耦合图 RH
├── freesurfer/<subj>/                          # FreeSurfer 皮层表面
├── group_mean_coupling_lh/rh.npy              # 组均值
└── batch_logs/<subj>.log                      # 个体日志
```

## 容错机制

- **断点续传**: 已完成的受试者自动跳过
- **独立处理**: 单受试者失败不影响其他
- **内存控制**: 监控可用内存，低于 4GB 时等待
- **并行控制**: 4 并行预处理 + 3 并行 recon-all
- **日志追溯**: 每个受试者有独立日志

## 错误恢复

```bash
# 查看失败的受试者
grep -l "FAIL" derivatives/batch_logs/*.log

# 清理并重启（保留已完成数据）
bash scripts/retry_failed.sh
bash scripts/run_batch.sh
```

## 分析阶段

```bash
# Phase 1: 全部预处理（上面脚本）
# Phase 2: 表面投射 + 耦合 + 组分析
/home/sad/miniconda3/envs/mind/bin/python scripts/phase3_surface_coupling.py

# Phase 3: PLS SC-FC 分析（论文 2016）
/home/sad/miniconda3/envs/mind/bin/python scripts/analysis/paper2016_pls.py
```

## 时间评估

- 预处理: ~215 × 120s / 4 并行 ≈ 1.8 小时
- recon-all: ~215 × 15min / 3 并行 ≈ 18 小时
- 总时间: ~20 小时
