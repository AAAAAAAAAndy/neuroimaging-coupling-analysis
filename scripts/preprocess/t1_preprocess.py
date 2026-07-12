"""
T1 preprocessing (Steps 2-3).
Step 2: DICOM → NIfTI via dcm2niix.
Step 3: FreeSurfer recon-all for cortical surface reconstruction.
"""
import logging
import shutil
import subprocess
import numpy as np
import nibabel as nib
from pathlib import Path
from preprocess import (
    OUT_T1, DATA_T1, FS_DIR, is_step_done, ensure_dir,
    setup_freesurfer_env, run_cmd
)

logger = logging.getLogger('preprocess.t1')


def t1_to_nifti(subject_id):
    """T1 DICOM → NIfTI via dcm2niix."""
    out = OUT_T1 / subject_id / f'{subject_id}_T1.nii.gz'
    if is_step_done(out):
        return str(out)
    ensure_dir(out.parent)

    src = DATA_T1 / subject_id
    if not src.exists():
        return None

    run_cmd(['dcm2niix', '-z', 'y', '-f', f'{subject_id}_T1',
             '-o', str(out.parent), '-p', 'n', '-v', '0', str(src)],
            timeout=300)

    if out.exists():
        logger.info(f'T1: {out.stat().st_size / 1e6:.1f}MB')
        return str(out)
    return None


def recon_all(subject_id, t1_path, timeout=36000):
    """
    FreeSurfer recon-all with full environment setup.
    Cleans incomplete directories before starting.
    """
    sphere = FS_DIR / subject_id / 'surf' / 'lh.sphere.reg'
    subj_dir = FS_DIR / subject_id

    if sphere.exists():
        return str(subj_dir)

    # Clean incomplete recon-all (must fully remove, FreeSurfer re-creates)
    if subj_dir.exists() and not sphere.exists():
        shutil.rmtree(str(subj_dir), ignore_errors=True)
    # Do NOT create subj_dir here — FreeSurfer creates it via -i flag

    env = setup_freesurfer_env()
    cmd = ['recon-all', '-subjid', subject_id, '-i', str(t1_path),
           '-sd', str(FS_DIR), '-all', '-openmp', '4']

    logger.info(f'recon-all start for {subject_id}...')

    # Write debug log outside subject dir to avoid FreeSurfer conflicts
    dbg_log = FS_DIR / f'{subject_id}_recon-debug.log'

    import time
    t0 = time.time()
    with open(dbg_log, 'w') as dbg:
        r = subprocess.run(cmd, stdout=dbg, stderr=subprocess.STDOUT,
                           timeout=timeout, env=env)
    elapsed = time.time() - t0

    if sphere.exists():
        logger.info(f'recon-all done: {elapsed:.0f}s')
        dbg_log.unlink(missing_ok=True)
        return str(subj_dir)

    # Log last lines for debugging
    if dbg_log.exists():
        lines = dbg_log.read_text().strip().split('\n')
        logger.warning(f'recon-all failed. Last lines:\n' + '\n'.join(lines[-5:]))
    return None
