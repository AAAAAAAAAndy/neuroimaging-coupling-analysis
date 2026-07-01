# 神经影像耦合分析 / Neuroimaging Coupling Analysis

复现两篇神经影像学论文的方法学：

- **论文 2016** (Misic et al., Cerebral Cortex)：结构-功能耦合 PLS 分析
- **论文 2022** (Baller et al., Cell Reports)：CBF-ALFF 神经血管耦合分析

---

## 项目结构

```
project2/
├── data/                              # 原始 DICOM 数据（只读）
│   ├── baseline_ASL/                  #   基线 ASL（3D pCASL, DERIVED CBF）
│   ├── baseline_fMRI/                 #   基线 BOLD 静息态
│   ├── baseline_T1/                   #   基线 T1 MPRAGE
│   ├── baseline_DWI/                  #   基线 DWI（部分被试）
│   ├── visit_ASL|fMRI|T1|DWI/        #   随访数据
│   └── baseline_ASL_special/          #   特殊 ASL 序列
├── doc/                               # 复现的目标论文
│   ├── 2016结构-功能耦合方法.pdf
│   └── 2022-神经血管耦合*.pdf
├── materials/                         # 被试元数据
│   ├── 人员及序列表（含脑计划和仑卡）.xlsx
│   └── 两篇分析结果-2026年6月23日233132.docx
├── scripts/                           # 所有脚本
│   ├── process_one.py                 # ★ 单被试端到端管道（步骤 1-12）
│   ├── run_batch.sh                   # ★ 批量脚本（断点续传 + 信号处理）
│   ├── retry_failed.sh                # ★ 失败任务清理重试
│   ├── preprocess/                    #   预处理模块
│   │   ├── __init__.py                #     共享路径/工具函数
│   │   ├── asl_to_cbf.py             #     Step 1:  ASL → CBF
│   │   ├── t1_preprocess.py          #     Step 2-3: T1 NIfTI + recon-all
│   │   ├── bold_preprocess.py        #     Step 4-9: BOLD → ALFF
│   │   └── dwi_tractography.py       #     Step 12:  DWI → SC 矩阵
│   ├── surface/                       #   表面分析模块
│   │   └── projection_coupling.py    #     Step 10-11: 表面投射 + 耦合
│   ├── analysis/                      #   组分析脚本
│   │   ├── paper2016_pipeline.py     #     PLS + Rich Club + Louvain
│   │   ├── paper2016_pls.py          #     PLS 核心算法
│   │   ├── paper2022_pipeline.py     #     GAM + Spin Test
│   │   ├── analysis_module.py        #     共享分析函数
│   │   ├── cbf_alff_pipeline.py      #     CBF-ALFF 耦合子模块
│   │   └── pls_sc_fc.py              #     PLS 子模块
│   └── configs/
│       └── pipeline_config.yaml      #   全局参数配置
├── output/                            # 所有处理产物（与 data/ 镜像）
│   ├── baseline_ASL/<subj>/           #   CBF
│   ├── baseline_T1/<subj>/            #   T1 NIfTI + 表面投射
│   ├── baseline_fMRI/<subj>/          #   BOLD + ALFF + 耦合
│   ├── baseline_DWI/<subj>/           #   SC 矩阵
│   ├── freesurfer/<subj>/             #   皮层表面重建
│   └── batch_logs/                    #   个体处理日志
└── README.md
```

---

## 环境

| 工具 | 版本 | 路径 |
|------|------|------|
| FreeSurfer | 7.4.1 | `/usr/local/freesurfer/` |
| FSL | 6.0.5.1 | `/usr/local/fsl/` |
| AFNI | 26.1.06 | `~/abin/` |
| MRtrix3 | 3.0.3 | `/usr/bin/` |
| Python | 3.9 (mind env) | `/home/sad/miniconda3/envs/mind/` |

```bash
conda activate mind
export FREESURFER_HOME=/usr/local/freesurfer
export FS_LICENSE=$FREESURFER_HOME/license.txt
export SUBJECTS_DIR=/mnt/d/project2/output/freesurfer
export FSLDIR=/usr/local/fsl
export PATH=$FREESURFER_HOME/bin:$FSLDIR/bin:$HOME/abin:$PATH
```

---

## 管道步骤

### 论文 2022（CBF-ALFF 耦合）

| 步骤 | 函数 | 工具 | 说明 |
|------|------|------|------|
| **step_1** | `asl_to_cbf()` | pydicom | ASL DERIVED → CBF NIfTI；无 DERIVED 时 pCASL 降级 |
| **step_2** | `t1_to_nifti()` | dcm2niix | T1 DICOM → NIfTI |
| **step_3** | `recon_all()` | FreeSurfer | `recon-all -all` 皮层重建 |
| **step_4** | `bold_to_nifti()` | dcm2niix | BOLD DICOM → 4D NIfTI (`-4 y -w 0`) |
| **step_5** | `motion_correct()` | FSL mcflirt | 6-DOF 刚体运动校正 |
| **step_5b** | `despike()` | AFNI 3dDespike | 时间序列异常值去除 |
| **step_6** | `brain_mask_and_seg()` | FreeSurfer | 脑掩膜 + WM/CSF 分割 |
| **step_7** | `confound_regression()` | Python | 36 参数混杂回归 |
| **step_8** | `bandpass()` | scipy | Butterworth 带通滤波 (0.01–0.08 Hz) |
| **step_9** | `compute_alff()` | numpy FFT | ALFF 图 |
| **step_10** | `project_to_surface()` | FreeSurfer | bbregister + mri_vol2surf → fsaverage5 |
| **step_11** | `compute_coupling()` | Python | 15 mm FWHM 邻域加权回归 ALFF~CBF |

### 论文 2016（SC-FC PLS）

| 步骤 | 函数 | 工具 | 说明 |
|------|------|------|------|
| **step_12a** | `dwi_dicom_to_mif()` | MRtrix3 mrconvert | DWI DICOM → .mif |
| **step_12b** | `dwi_response_and_fod()` | dwi2response + dwi2fod | 响应函数 + CSD FOD |
| **step_12c** | `dwi_tractography()` | tckgen SD_STREAM | 确定性纤维追踪 |
| **step_12d** | `dwi_connectome()` | tck2connectome | 114 区 SC 矩阵 |
| — | `paper2016_pipeline.py` | Python | PLS + 置换 + Bootstrap + Rich Club |

---

## 使用方法

### 批量处理（screen 后台）

```bash
cd /mnt/d/project2

# 启动
screen -dmS batch bash scripts/run_batch.sh

# 实时日志
screen -r batch

# 另一终端查进度
bash scripts/run_batch.sh --status

# 查单个被试步骤状态
bash scripts/run_batch.sh --step-status B1_0024

# 重试失败任务
bash scripts/retry_failed.sh              # 清理中间产物
bash scripts/run_batch.sh                 # 重跑
```

### 单被试处理

```bash
# 完整管道 (论文 2022 + 2016)
/home/sad/miniconda3/envs/mind/bin/python scripts/process_one.py --subject B1_0024

# 仅论文 2022
python scripts/process_one.py --subject B1_0024 --paper 2022

# 仅论文 2016
python scripts/process_one.py --subject B1_0024 --paper 2016
```

### 组分析

```bash
# 论文 2022: GAM + Spin Test
python scripts/analysis/paper2022_pipeline.py

# 论文 2016: PLS + Rich Club
python scripts/analysis/paper2016_pipeline.py
```

---

## 断点续传机制

每个步骤在执行前检查输出文件是否存在：

```
step_1  → output/baseline_ASL/<subj>/<subj>_CBF.nii.gz
step_2  → output/baseline_T1/<subj>/<subj>_T1.nii.gz
step_3  → output/freesurfer/<subj>/surf/lh.sphere.reg
step_4  → output/baseline_fMRI/<subj>/<subj>_BOLD.nii.gz
step_5_9→ output/baseline_fMRI/<subj>/<subj>_ALFF.nii.gz
step_10 → output/baseline_T1/<subj>/surface/fsaverage5_*_cbf_lh.mgh
step_11 → output/baseline_fMRI/<subj>/coupling_lh.npy
```

中断后重新运行，已完成的步骤自动跳过。

---

## 信号处理

`run_batch.sh` 捕获 `SIGINT`/`SIGTERM`，自动清理所有子进程后退出：

```bash
trap cleanup SIGINT SIGTERM
cleanup() {
    for pid in "${CHILD_PIDS[@]}"; do kill "$pid" 2>/dev/null; done
    wait 2>/dev/null
    exit 0
}
```

---

## 故障排查

| 问题 | 解决 |
|------|------|
| OOM (137) | 减小 `MAX_PARALLEL` (改为 2-3) |
| FreeSurfer license | `export FS_LICENSE=/usr/local/freesurfer/license.txt` |
| 3dDespike 找不到 | `export PATH=$HOME/abin:$PATH` |
| recon-all 中途失败 | `retry_failed.sh` 清理不完整目录后重跑 |
| BOLD dcm2niix 超时 | 重跑即可（已有输出会跳过） |

---

## 处理时间

| 阶段 | 时间/被试 | 215 被试 (6 并行) |
|------|----------|-------------------|
| ASL → CBF | 2s | ~1 min |
| T1 → NIfTI | 5s | ~3 min |
| recon-all | 15-20 min | ~9 h (4 并行) |
| BOLD → NIfTI | 75-120s | ~1 h |
| BOLD → ALFF | 15s | ~9 min |
| 表面投射 + 耦合 | 35s | ~20 min |
