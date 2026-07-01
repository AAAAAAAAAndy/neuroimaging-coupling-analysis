#!/usr/bin/env python3
"""
Pipeline orchestrator: single-subject end-to-end processing.
Imports from modular preprocess/surface packages.

Usage:
    python scripts/process_one.py --subject B1_0024
"""
import sys
import logging
import argparse
import time
from pathlib import Path

# Add scripts/ to path so packages are importable
_scripts_dir = str(Path(__file__).parent)
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

from preprocess.asl_to_cbf import asl_to_cbf
from preprocess.t1_preprocess import t1_to_nifti, recon_all
from preprocess.bold_preprocess import bold_to_nifti, bold_to_alff
from surface.projection_coupling import project_to_surface, compute_coupling
from preprocess.dwi_tractography import (
    dwi_dicom_to_mif, dwi_response_and_fod,
    dwi_tractography as dwi_track, dwi_connectome
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger('pipeline')


def run_paper2022(subject_id):
    """
    Paper 2022 pipeline: CBF-ALFF coupling.
    Returns True if all steps succeed.

    Step numbering:
        1   ASL → CBF (pCASL quantification)
        2   T1 DICOM → NIfTI
        3   FreeSurfer recon-all (cortical surface)
        4   BOLD DICOM → 4D NIfTI
        5   BOLD motion correction (mcflirt)
        5b  3dDespike (AFNI outlier removal)
        6   Brain mask + WM/CSF segmentation
        7   36-parameter confound regression
        8   Bandpass filter (0.01–0.08 Hz)
        9   ALFF computation
        10  Surface projection → fsaverage5
        11  CBF-ALFF coupling (local weighted regression)
    """
    logger.info(f'===== Paper 2022: {subject_id} =====')
    t0 = time.time()

    # Step 1: ASL → CBF
    logger.info('[step_1] ASL → CBF')
    cbf = asl_to_cbf(subject_id)
    if not cbf:
        logger.error('FAIL: step_1 CBF')
        return False

    # Step 2: T1 → NIfTI
    logger.info('[step_2] T1 DICOM → NIfTI')
    t1 = t1_to_nifti(subject_id)
    if not t1:
        logger.error('FAIL: step_2 T1')
        return False

    # Step 3: FreeSurfer recon-all
    logger.info('[step_3] FreeSurfer recon-all')
    fs = recon_all(subject_id, t1)
    if not fs:
        logger.error('FAIL: step_3 recon-all')
        return False

    # Step 4: BOLD → NIfTI
    logger.info('[step_4] BOLD DICOM → NIfTI')
    bold = bold_to_nifti(subject_id)
    if not bold:
        logger.error('FAIL: step_4 BOLD')
        return False

    # Steps 5-9: BOLD preprocessing → ALFF
    logger.info('[step_5→9] BOLD preprocessing → ALFF')
    alff = bold_to_alff(subject_id, bold)
    if not alff:
        logger.error('FAIL: step_5-9 ALFF')
        return False

    # Step 10: Surface projection
    logger.info('[step_10] Surface projection → fsaverage5')
    for hemi in ['lh', 'rh']:
        s1 = project_to_surface(subject_id, cbf, hemi, 'cbf')
        s2 = project_to_surface(subject_id, alff, hemi, 'alff')
        if not s1 or not s2:
            logger.error(f'FAIL: step_10 surface ({hemi})')
            return False

    # Step 11: Coupling
    logger.info('[step_11] CBF-ALFF coupling')
    result = compute_coupling(subject_id)
    if not result:
        logger.error('FAIL: step_11 coupling')
        return False

    elapsed = time.time() - t0
    logger.info(f'Paper 2022 done ({elapsed:.0f}s): mean_abs={result["mean_abs"]:.6f}')
    return True


def run_paper2016(subject_id):
    """
    Paper 2016 pipeline: DWI tractography → SC matrix.
    Returns True if all steps succeed.

    Step numbering:
        12a  DWI DICOM → MRtrix3 .mif
        12b  Response function estimation (dwi2response)
        12c  FOD estimation (dwi2fod CSD)
        12d  Deterministic streamline tractography (tckgen SD_STREAM)
        12e  SC connectome matrix (tck2connectome)
    """
    logger.info(f'===== Paper 2016: {subject_id} =====')
    t0 = time.time()

    # Step 12a: DWI DICOM → .mif
    logger.info('[step_12a] DWI DICOM → .mif')
    dwi = dwi_dicom_to_mif(subject_id)
    if not dwi:
        logger.warning('DWI not available, skipping paper 2016')
        return False

    # Step 12b-c: Response + FOD
    logger.info('[step_12b-c] Response + FOD estimation')
    resp, fod = dwi_response_and_fod(dwi)
    if not fod:
        logger.error('FAIL: step_12b-c FOD')
        return False

    # Step 12d: Tractography
    logger.info('[step_12d] Deterministic streamline tractography')
    tracks = dwi_track(fod, subject_id)
    if not tracks:
        logger.error('FAIL: step_12d tractography')
        return False

    # Step 12e: Connectome
    logger.info('[step_12e] SC connectome')
    sc = dwi_connectome(tracks, subject_id)
    if not sc:
        logger.error('FAIL: step_12e connectome')
        return False

    elapsed = time.time() - t0
    logger.info(f'Paper 2016 done ({elapsed:.0f}s): SC={sc}')
    return True


def main():
    parser = argparse.ArgumentParser(description='Single-subject pipeline')
    parser.add_argument('--subject', required=True, help='Subject ID')
    parser.add_argument('--paper', choices=['2022', '2016', 'both'], default='both',
                        help='Which paper pipeline to run')
    args = parser.parse_args()

    subject_id = args.subject
    t0 = time.time()

    ok_2022 = True
    ok_2016 = True

    if args.paper in ('2022', 'both'):
        ok_2022 = run_paper2022(subject_id)
    if args.paper in ('2016', 'both'):
        ok_2016 = run_paper2016(subject_id)

    elapsed = time.time() - t0
    status = 'DONE' if (ok_2022 and ok_2016) else 'FAIL'
    logger.info(f'{status} {subject_id} (total {elapsed:.0f}s)')

    return 0 if (ok_2022 and ok_2016) else 1


if __name__ == '__main__':
    sys.exit(main())
