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
from preprocess.bold_preprocess import bold_to_nifti, bold_to_alff, bold_to_fc
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
    Handles missing modalities gracefully - process whatever data is available.

    Step numbering:
        1   ASL → CBF (pCASL quantification) [optional]
        2   T1 DICOM → NIfTI [if T1 available]
        3   FreeSurfer recon-all [if T1 available]
        4   BOLD DICOM → 4D NIfTI [if BOLD available]
        5   BOLD motion correction (mcflirt)
        5b  3dDespike (AFNI outlier removal)
        6   Brain mask + WM/CSF segmentation
        7   36-parameter confound regression
        8   Bandpass filter (0.01–0.08 Hz)
        9   ALFF computation
        10  Surface projection → fsaverage5 [if FS available]
        11  CBF-ALFF coupling (local weighted regression) [if both CBF+ALFF available]
    """
    logger.info(f'===== Paper 2022: {subject_id} =====')
    t0 = time.time()

    # Step 1: ASL → CBF (optional)
    logger.info('[step_1] ASL → CBF')
    cbf = asl_to_cbf(subject_id) if _has_data(subject_id, 'asl') else None
    if not cbf:
        logger.info('  No ASL data, skipping CBF')

    # Step 2: T1 → NIfTI
    logger.info('[step_2] T1 DICOM → NIfTI')
    t1 = t1_to_nifti(subject_id) if _has_data(subject_id, 't1') else None
    if not t1:
        logger.info('  No T1 data, skipping surface reconstruction')
        return False  # T1 is essential for surface projection

    # Step 3: FreeSurfer recon-all
    logger.info('[step_3] FreeSurfer recon-all')
    fs = recon_all(subject_id, t1)
    if not fs:
        logger.error('FAIL: step_3 recon-all')
        return False

    # Step 4: BOLD → NIfTI
    logger.info('[step_4] BOLD DICOM → NIfTI')
    bold = bold_to_nifti(subject_id) if _has_data(subject_id, 'bold') else None
    if not bold:
        logger.info('  No BOLD data, skipping ALFF and coupling')
        return True  # T1 processing done, no BOLD = OK

    # Steps 5-9: BOLD preprocessing → ALFF
    logger.info('[step_5→9] BOLD preprocessing → ALFF')
    alff = bold_to_alff(subject_id, bold)
    if not alff:
        logger.error('FAIL: step_5-9 ALFF')
        return False

    # Step 10: Surface projection
    logger.info('[step_10] Surface projection → fsaverage5')
    for hemi in ['lh', 'rh']:
        if cbf:
            project_to_surface(subject_id, cbf, hemi, 'cbf')
        project_to_surface(subject_id, alff, hemi, 'alff')

    # Step 11: Coupling (only if both CBF and ALFF available)
    if cbf:
        logger.info('[step_11] CBF-ALFF coupling')
        result = compute_coupling(subject_id)
        if not result:
            logger.warning('step_11 coupling returned no result')
    else:
        logger.info('[step_11] Skipped (no CBF)')

    elapsed = time.time() - t0
    logger.info(f'Paper 2022 done ({elapsed:.0f}s)')
    return True


def _has_data(subject_id, modality):
    """Check if subject has raw DICOM data for given modality."""
    from preprocess import DATA, TIMEPOINT
    paths = {
        'asl': [
            DATA / f'{TIMEPOINT}_ASL' / subject_id,
        ] + (
            [
                DATA / f'{TIMEPOINT}_ASL_special' / sub / subject_id
                for sub in (DATA / f'{TIMEPOINT}_ASL_special').iterdir()
                if sub.is_dir()
            ]
            if (DATA / f'{TIMEPOINT}_ASL_special').exists()
            else []
        ),
        'bold': DATA / f'{TIMEPOINT}_fMRI' / subject_id,
        't1': DATA / f'{TIMEPOINT}_T1' / subject_id,
        'dwi': DATA / f'{TIMEPOINT}_DWI' / subject_id,
    }
    check_paths = paths.get(modality, [])
    if not isinstance(check_paths, list):
        check_paths = [check_paths]
    return any(p.exists() for p in check_paths)


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
        12f  BOLD → FC matrix (for PLS)
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

    # Step 12f: BOLD → FC matrix
    logger.info('[step_12f] BOLD → FC matrix')
    bold_path = OUT_FMRI / subject_id / f'{subject_id}_BOLD.nii.gz'
    if not bold_path.exists():
        # Try to convert BOLD from DICOM
        bold = bold_to_nifti(subject_id)
        if bold:
            bold_path = Path(bold)

    if bold_path.exists():
        fc = bold_to_fc(subject_id, str(bold_path))
        if fc is not None:
            out_dir = OUT_DWI / subject_id
            out_dir.mkdir(parents=True, exist_ok=True)
            np.save(str(out_dir / 'FC_matrix.npy'), fc)
            logger.info(f'FC matrix: {fc.shape}')

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
