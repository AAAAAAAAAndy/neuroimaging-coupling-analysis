"""
Shared utilities for preprocessing pipeline.
Centralizes path configuration, logging setup, and common operations.
"""
import os
import sys
import time
import shutil
import logging
import subprocess
from pathlib import Path

# ---- Path configuration ----
import os

BASE = Path('/mnt/d/project2')
DATA = BASE / 'data'
FS_DIR = BASE / 'output' / 'freesurfer'

def get_timepoint(subj):
    """Detect timepoint from subject ID prefix."""
    if subj.startswith(('B1_', 'A1_')):
        return 'baseline'
    elif subj.startswith('sub'):
        return 'visit'
    return os.environ.get('TIMEPOINT', 'baseline')

def set_subject(subj):
    """Update module-level paths for the given subject (detects timepoint).

    Updates globals in preprocess AND its submodules (bold_preprocess,
    t1_preprocess, asl_to_cbf, dwi_tractography, surface.projection_coupling)
    so path lookups reflect the current subject's timepoint.
    """
    global _current_timepoint, OUT_ASL, OUT_T1, OUT_FMRI, OUT_DWI
    global DATA_ASL, DATA_BOLD, DATA_T1, DATA_DWI
    _current_timepoint = get_timepoint(subj)
    OUT_ASL = BASE / 'output' / f'{_current_timepoint}_ASL'
    OUT_T1 = BASE / 'output' / f'{_current_timepoint}_T1'
    OUT_FMRI = BASE / 'output' / f'{_current_timepoint}_fMRI'
    OUT_DWI = BASE / 'output' / f'{_current_timepoint}_DWI'
    DATA_ASL = DATA / f'{_current_timepoint}_ASL'
    DATA_BOLD = DATA / f'{_current_timepoint}_fMRI'
    DATA_T1 = DATA / f'{_current_timepoint}_T1'
    DATA_DWI = DATA / f'{_current_timepoint}_DWI'

    # Propagate to already-imported submodules so their `from preprocess
    # import OUT_*` bindings stay in sync.
    import sys
    updates = {
        'OUT_ASL': OUT_ASL, 'OUT_T1': OUT_T1,
        'OUT_FMRI': OUT_FMRI, 'OUT_DWI': OUT_DWI,
        'DATA_ASL': DATA_ASL, 'DATA_BOLD': DATA_BOLD,
        'DATA_T1': DATA_T1, 'DATA_DWI': DATA_DWI,
        'TIMEPOINT': _current_timepoint,
    }
    for modname in list(sys.modules):
        if modname.startswith('preprocess') or modname.startswith('surface'):
            mod = sys.modules[modname]
            for k, v in updates.items():
                setattr(mod, k, v)

_current_timepoint = os.environ.get('TIMEPOINT', 'baseline')
OUT_ASL = BASE / 'output' / f'{_current_timepoint}_ASL'
OUT_T1 = BASE / 'output' / f'{_current_timepoint}_T1'
OUT_FMRI = BASE / 'output' / f'{_current_timepoint}_fMRI'
OUT_DWI = BASE / 'output' / f'{_current_timepoint}_DWI'
DATA_ASL = DATA / f'{_current_timepoint}_ASL'
DATA_BOLD = DATA / f'{_current_timepoint}_fMRI'
DATA_T1 = DATA / f'{_current_timepoint}_T1'
DATA_DWI = DATA / f'{_current_timepoint}_DWI'

# Backward compat
TIMEPOINT = _current_timepoint

# ---- FreeSurfer environment ----
def setup_freesurfer_env():
    """Set up FreeSurfer environment variables for subprocess calls."""
    env = os.environ.copy()
    env['FREESURFER_HOME'] = '/usr/local/freesurfer'
    env['FSFAST_HOME'] = '/usr/local/freesurfer/fsfast'
    env['SUBJECTS_DIR'] = str(FS_DIR)
    env['FSF_OUTPUT_FORMAT'] = 'nii.gz'
    env['PATH'] = '/usr/local/freesurfer/bin:' + env.get('PATH', '')
    env['FSLDIR'] = '/usr/local/fsl'
    env['PATH'] = '/usr/local/fsl/bin:' + env['PATH']
    env['PATH'] = str(Path.home() / 'abin') + ':' + env['PATH']
    return env


# ---- Logging ----
def get_logger(name, log_file=None):
    """Get a logger with consistent formatting."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        fmt = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        logger.addHandler(sh)
        if log_file:
            fh = logging.FileHandler(log_file)
            fh.setFormatter(fmt)
            logger.addHandler(fh)
    logger.setLevel(logging.INFO)
    return logger


# ---- Command execution ----
def run_cmd(cmd, timeout=600, log_stderr=True):
    """Run a command with timeout and optional error logging."""
    logger = logging.getLogger('util')
    logger.info(f'RUN: {" ".join(str(c) for c in cmd)}')
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0 and log_stderr:
        logger.warning(f'  STDERR: {r.stderr[:500]}')
    return r


def run_cmd_env(cmd, env=None, timeout=36000):
    """Run a command with custom environment (e.g., FreeSurfer)."""
    if env is None:
        env = setup_freesurfer_env()
    logger = logging.getLogger('util')
    logger.info(f'RUN: {" ".join(str(c) for c in cmd)}')
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)


# ---- Intermediate file management ----
def atomic_write_nifti(data, affine, output_path):
    """Write NIfTI atomically (write to tmp then rename)."""
    import nibabel as nib
    tmp_path = str(output_path) + '.tmp'
    img = nib.Nifti1Image(data, affine)
    nib.save(img, tmp_path)
    os.replace(tmp_path, str(output_path))


def clean_intermediates(subj_dir, patterns):
    """Clean up intermediate files matching patterns."""
    for pattern in patterns:
        for f in Path(subj_dir).glob(pattern):
            f.unlink(missing_ok=True)


def is_step_done(output_path):
    """Check if a step's output already exists (resume capability)."""
    return Path(output_path).exists()


def ensure_dir(path):
    """Create directory if it doesn't exist."""
    Path(path).mkdir(parents=True, exist_ok=True)
