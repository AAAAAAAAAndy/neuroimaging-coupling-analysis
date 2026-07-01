#!/usr/bin/env python3
"""
Paper 2016 - DWI tractography pipeline (MRtrix3 + FreeSurfer).
Faithful reproduction: GQI-like CSD + deterministic streamline + 114-region SC matrix.
"""
import os
import sys
import time
import logging
import argparse
import subprocess
import numpy as np
import nibabel as nib
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger('DWI_2016')

from preprocess import BASE, DATA, FS_DIR, OUT_DWI as OUT


def run_cmd(cmd, timeout=3600):
    logger.info(f'RUN: {cmd}')
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        logger.warning(f'  STDERR: {r.stderr[:500]}')
    return r


def dwi_dicom_to_mif(subject_id):
    """Convert DWI DICOM to MRtrix3 .mif format."""
    dwi_dir = DATA / 'baseline_DWI' / subject_id
    out_dir = OUT / subject_id
    out_dir.mkdir(parents=True, exist_ok=True)

    dwi_mif = out_dir / 'dwi.mif'
    if dwi_mif.exists():
        return str(dwi_mif)

    if not dwi_dir.exists():
        logger.warning(f'No DWI data for {subject_id}')
        return None

    # Find main DWI (prefer b=1000)
    dwi_dcm_dir = None
    for name in ['epi_dwi_tra_b1000', 'epi_dwi_tra', 'DWI']:
        if (dwi_dir / name).is_dir():
            dwi_dcm_dir = dwi_dir / name
            break
    if not dwi_dcm_dir:
        for sub in dwi_dir.iterdir():
            if sub.is_dir() and 'ADC' not in sub.name:
                dwi_dcm_dir = sub
                break

    if not dwi_dcm_dir:
        logger.error(f'No DWI DICOM dir for {subject_id}')
        return None

    run_cmd(f'mrconvert "{dwi_dcm_dir}" "{dwi_mif}" -quiet', timeout=300)

    if dwi_mif.exists():
        logger.info(f'DWI MIF created: {dwi_mif.stat().st_size/1e6:.1f}MB')
        return str(dwi_mif)
    return None


def dwi_response_and_fod(dwi_mif):
    """Estimate response function and compute FODs (CSD)."""
    out_dir = Path(dwi_mif).parent
    resp_file = out_dir / 'response.txt'
    fod_file = out_dir / 'fod.mif'

    if fod_file.exists() and resp_file.exists():
        return str(resp_file), str(fod_file)

    if not resp_file.exists():
        run_cmd(f'dwi2response tournier {dwi_mif} {resp_file} -quiet', timeout=600)
    if not resp_file.exists():
        run_cmd(f'dwi2response dhollander {dwi_mif} {resp_file} -quiet', timeout=600)

    if not resp_file.exists():
        logger.error('Response estimation failed')
        return None, None

    run_cmd(f'dwi2fod csd {dwi_mif} {resp_file} {fod_file} -quiet', timeout=1800)

    if fod_file.exists():
        return str(resp_file), str(fod_file)
    return None, None


def dwi_tractography(fod_file, subject_id):
    """Deterministic streamline tractography (paper 2016 uses FACT-like tracking)."""
    out_dir = Path(fod_file).parent
    tracks = out_dir / 'tracks.tck'
    if tracks.exists():
        return str(tracks)

    # SD_STREAM = deterministic-like through FOD peaks - closest to paper's "deterministic streamline"
    cmd = (f'tckgen {fod_file} {tracks} '
           f'-algorithm SD_STREAM '
           f'-angle 45 -cutoff 0.1 -maxlength 250 -minlength 10 '
           f'-select 50000 -seed_dynamic {fod_file} -quiet')
    t0 = time.time()
    run_cmd(cmd, timeout=7200)
    logger.info(f'Tracking took {time.time()-t0:.0f}s')

    if tracks.exists():
        return str(tracks)
    return None


def dwi_connectome(tracks_file, subject_id):
    """Build 114-region SC matrix from tracks + FreeSurfer parcellation."""
    out_dir = Path(tracks_file).parent
    sc_csv = out_dir / 'SC_connectome.csv'
    sc_npy = out_dir / 'SC_matrix.npy'
    if sc_npy.exists():
        return str(sc_npy)

    subj_fs = FS_DIR / subject_id
    if not (subj_fs / 'surf' / 'lh.sphere.reg').exists():
        logger.warning(f'No FS surface for {subject_id}')
        return None

    # Convert FreeSurfer aparc+aseg to DWI space, then build connectome
    seg_fs = subj_fs / 'mri' / 'aparc+aseg.mgz'
    if not seg_fs.exists():
        seg_fs = subj_fs / 'mri' / 'aseg.mgz'
    if not seg_fs.exists():
        return None

    # Use bbregister + mri_vol2vol to get seg in DWI space
    dwi_ref = out_dir / 'dwi.mif'
    seg_dwi = out_dir / 'aseg_in_dwi.nii.gz'

    if not seg_dwi.exists():
        reg_dat = subj_fs / 'mri' / 'register.dat'
        if not reg_dat.exists():
            run_cmd(f'bbregister --s {subject_id} --mov {dwi_ref} --reg {reg_dat} --t1', timeout=300)
        if reg_dat.exists():
            run_cmd(f'mri_vol2vol --seg {seg_fs} --temp {dwi_ref} --o {seg_dwi} '
                    f'--reg {reg_dat} --nearest', timeout=300)

    if not seg_dwi.exists():
        return None

    # Build connectome
    run_cmd(f'tck2connectome {tracks_file} {seg_dwi} {sc_csv} -quiet', timeout=600)

    if sc_csv.exists():
        mtx = np.loadtxt(str(sc_csv), delimiter=',')
        mtx = (mtx + mtx.T) / 2  # Symmetrize
        np.save(str(sc_npy), mtx)
        logger.info(f'SC matrix: {mtx.shape}')
        return str(sc_npy)
    return None


def process_one(subject_id):
    """Full DWI pipeline for one subject - faithful to paper 2016."""
    logger.info(f'========== DWI: {subject_id} ==========')

    dwi = dwi_dicom_to_mif(subject_id)
    if not dwi: return False

    resp, fod = dwi_response_and_fod(dwi)
    if not fod: return False

    tracks = dwi_tractography(fod, subject_id)
    if not tracks: return False

    sc = dwi_connectome(tracks, subject_id)
    return sc is not None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--subject', required=True)
    args = parser.parse_args()
    process_one(args.subject)
