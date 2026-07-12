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
│   ├── baseline_ASL_special/          #   特殊 ASL 序列（mosaic 格式，双层嵌套）
│   │   ├── ASL_3D_tra_M0/             #     30 被试，2 DICOM/被试（M0 + ΔM）
│   │   └── ASL_3D_tra_iso/            #     30 被试，6 DICOM/被试（多 PLD）
│   ├── baseline_DWI/                  #   基线 DWI（部分被试）
│   ├── baseline_fMRI/                 #   基线 BOLD 静息态
│   ├── baseline_T1/                   #   基线 T1 MPRAGE
│   ├── visit_ASL/                     #   随访 ASL（子目录含 DERIVED CBF）
│   ├── visit_DWI/                     #   随访 DWI
│   ├── visit_fMRI/                    #   随访 BOLD
│   └── visit_T1/                      #   随访 T1
├── docs/                              # 复现的目标论文
│   ├── 2016结构-功能耦合方法.pdf
│   └── 2022-神经血管耦合Developmental coupling of cerebral blood flow and (1).pdf
├── materials/                         # 被试元数据
├── scripts/                           # 所有脚本
│   ├── process_one.py                 # ★ 单被试端到端管道（步骤 1-12）
│   ├── run_batch.sh                   # ★ 批量脚本（断点续传 + 信号处理）
│   ├── run_all_timepoints.sh          # ★ baseline + visit 全量
│   ├── data_status.sh                 # ★ 实时进度监控
│   ├── retry_failed.sh                # ★ 失败任务清理重试
│   ├── preprocess/                    #   预处理模块
│   │   ├── __init__.py                #     共享路径/工具函数（timepoint-agnostic）
│   │   ├── asl_to_cbf.py             #     Step 1:  ASL → CBF（3 种格式）
│   │   ├── asl_special.py           #      Siemens mosaic 解码器
│   │   ├── t1_preprocess.py          #     Step 2-3: T1 NIfTI + recon-all
│   │   ├── bold_preprocess.py        #     Step 4-9: BOLD → ALFF
│   │   └── dwi_tractography.py       #     Step 12:  DWI → SC 矩阵
│   ├── surface/                       #   表面分析模块
│   │   └── projection_coupling.py    #     Step 10-11: 表面投射 + 耦合
│   └── analysis/                      #   组分析脚本
│       ├── paper2016_pipeline.py     #     PLS + Rich Club + Louvain
│       ├── paper2016_pls.py          #     PLS 核心算法
│       ├── paper2022_pipeline.py     #     GAM + Spin Test
│       └── analysis_module.py        #     共享分析函数
├── output/                            # 所有处理产物（与 data/ 镜像）
│   ├── baseline_ASL/<subj>/           #   CBF
│   ├── baseline_T1/<subj>/            #   T1 NIfTI + 表面投射
│   ├── baseline_fMRI/<subj>/          #   BOLD + ALFF + 耦合
│   ├── baseline_DWI/<subj>/           #   SC 矩阵
│   ├── baseline_ASL_special/<sub>/<subj>/  #   mosaic 格式 CBF
│   ├── visit_<modality>/<subj>/       #   随访数据产物
│   ├── freesurfer/<subj>/             #   皮层表面重建
│   └── batch_logs/                    #   个体处理日志
└── README.md
```

---

## 数据概况

| 目录 | 被试数 | 说明 |
|------|--------|------|
| baseline_ASL | 234 | 3D pCASL，含 DERIVED CBF |
| baseline_ASL_special/ASL_3D_tra_M0 | 30 | Siemens mosaic，2 DICOM/被试 |
| baseline_ASL_special/ASL_3D_tra_iso | 30 | Siemens mosaic，6 DICOM/被试 |
| baseline_fMRI | 332 | 静息态 BOLD |
| baseline_T1 | 369 | MPRAGE |
| baseline_DWI | 94 | 部分被试 |
| visit_ASL | 70 | 子目录含 DERIVED CBF |
| visit_fMRI | 83 | |
| visit_T1 | 83 | |
| visit_DWI | 75 | |
| **总计独立被试** | **455** | |

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

### 论文 2022（CBF-ALFF 耦合）— `process_one.py`

| 步骤 | 函数 | 工具 | 说明 |
|------|------|------|------|
| **step_1** | `asl_to_cbf()` | pydicom | ASL→CBF（3 种格式：标准 DERIVED、mosaic、visit DERIVED） |
| **step_2** | `t1_to_nifti()` | dcm2niix | T1 DICOM → NIfTI |
| **step_3** | `recon_all()` | FreeSurfer | `recon-all -all` 皮层重建 |
| **step_4** | `bold_to_nifti()` | dcm2niix | BOLD DICOM → 4D NIfTI (`-4 y -w 0`) |
| **step_5** | `motion_correct()` | FSL mcflirt | 6-DOF 刚体运动校正 |
| **step_5b** | `despike()` | AFNI 3dDespike | 时间序列异常值去除 |
| **step_6** | `brain_mask_and_seg()` | FreeSurfer | 脑掩膜 + WM/CSF 分割（Python affine resampling） |
| **step_7** | `confound_regression()` | Python | 36 参数混杂回归 |
| **step_8** | `bandpass()` | scipy | Butterworth 带通滤波 (0.01–0.08 Hz) |
| **step_9** | `compute_alff()` | numpy FFT | ALFF 图 |
| **step_10** | `project_to_surface()` | FreeSurfer | bbregister + mri_vol2surf → fsaverage5 |
| **step_11** | `compute_coupling()` | Python | 15 mm FWHM 邻域加权回归 ALFF~CBF |

### 论文 2016（SC-FC PLS）— `analysis/paper2016_pipeline.py`

| 步骤 | 函数 | 工具 | 说明 |
|------|------|------|------|
| **step_12a** | `dwi_dicom_to_mif()` | MRtrix3 mrconvert | DWI DICOM → .mif |
| **step_12b** | `dwi_response_and_fod()` | dwi2response + dwi2fod | 响应函数 + CSD FOD |
| **step_12c** | `dwi_tractography()` | tckgen SD_STREAM | 确定性纤维追踪 |
| **step_12d** | `dwi_connectome()` | tck2connectome | 114 区 SC 矩阵 |
| — | `paper2016_pipeline.py` | Python | PLS + 置换 + Bootstrap + Rich Club |

---

## 使用方法

### 批量处理（screen 后台，全量 455 被试）

```bash
cd /mnt/d/project2

# 启动 baseline batch（screen 名为 batch）
screen -dmS batch bash -c 'cd /mnt/d/project2 && TIMEPOINT=baseline bash scripts/run_batch.sh 2>&1 | tee output/baseline_batch.log'

# 启动 visit batch（screen 名为 visit_batch）
screen -dmS visit_batch bash -c 'cd /mnt/d/project2 && TIMEPOINT=visit bash scripts/run_batch.sh 2>&1 | tee output/visit_batch.log'

# 实时日志
screen -r batch
screen -r visit_batch

# 另一终端查进度
bash scripts/run_batch.sh --status
bash scripts/data_status.sh   # 各目录详细进度

# 重试失败任务
bash scripts/retry_failed.sh
bash scripts/run_batch.sh
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

## ASL 数据的三种格式

| 格式 | 位置 | 特征 | 处理方式 |
|------|------|------|----------|
| 标准 DERIVED | `baseline_ASL/<subj>/` | 36 ORIGINAL + 36 DERIVED DICOM | 直接堆叠 DERIVED 图像 |
| Mosaic | `baseline_ASL_special/ASL_3D_tra_M0/` | 2 DICOM/被试，448×448 Siemens mosaic | mosaic 解码 + pCASL 公式 |
| Mosaic 多PLD | `baseline_ASL_special/ASL_3D_tra_iso/` | 6 DICOM/被试，InstanceNumbers 1,41,81,... | mosaic 解码，取 M0 + 平均灌注 |
| Visit DERIVED | `visit_ASL/<subj>/CBF_Sep.../` | 子目录含 DERIVED CBF | 直接复制为输出 |

---

## 输出镜像策略

`output/` 目录结构与 `data/` 完全镜像：

```
data/baseline_ASL_special/ASL_3D_tra_M0/A1_0460/
  → output/baseline_ASL_special/ASL_3D_tra_M0/A1_0460/A1_0460_CBF.nii.gz

data/visit_ASL/sub005_V1/CBF_Sep 21 2025 14-05-50 CST_Series0950/
  → output/visit_ASL/sub005_V1/CBF_Sep 21 2025 14-05-50 CST_Series0950/sub005_V1_CBF.nii.gz
```

实现：`asl_to_cbf()` 中 `rel_path = asl_dir.relative_to(DATA)` → `out_dir = BASE / 'output' / rel_path`

---

## 断点续传

每个步骤执行前检查输出文件是否存在，已完成步骤自动跳过。中断后重新运行即续传。

## 信号处理

`run_batch.sh` 捕获 `SIGINT`/`SIGTERM`，自动清理所有子进程后退出。

---

## 故障排查

| 问题 | 解决 |
|------|------|
| OOM (137) | 减小 `MAX_PARALLEL` |
| FreeSurfer license | `export FS_LICENSE=/usr/local/freesurfer/license.txt` |
| 3dDespike 找不到 | `export PATH=$HOME/abin:$PATH` |
| mosaic 解码失败 | 使用 `pydicom force=True` |
| recon-all 中途失败 | `retry_failed.sh` 清理后重跑 |

---

## 处理时间

| 阶段 | 时间/被试 | 455 被试 (4 并行 recon) |
|------|----------|------------------------|
| ASL → CBF | 2s | ~1 min |
| T1 → NIfTI | 5s | ~3 min |
| recon-all | ~15-20 min | ~12 h |
| BOLD → NIfTI | 75-120s | ~1 h |
| BOLD → ALFF | 15s | ~9 min |
| 表面投射 + 耦合 | 35s | ~20 min |
