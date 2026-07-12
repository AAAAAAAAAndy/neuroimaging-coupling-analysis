"""
Surface projection + CBF-ALFF coupling (Steps 10-11).
Step 10: Volume → fsaverage5 surface (bbregister + mri_vol2surf + mri_surf2surf)
Step 11: Local weighted regression coupling on surface
"""
import sys
import logging
import time
import numpy as np
import nibabel as nib
from pathlib import Path
from scipy.spatial import cKDTree

# Ensure scripts/ is on path for cross-package imports
_scripts_dir = str(Path(__file__).resolve().parent.parent)
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

from preprocess import (
    OUT_T1, OUT_FMRI, OUT_DWI, FS_DIR, is_step_done, ensure_dir,
    setup_freesurfer_env, run_cmd
)

logger = logging.getLogger('surface.coupling')


def project_to_surface(subject_id, vol_path, hemi, map_name):
    """Project volume to fsaverage5 surface via bbregister → mri_vol2surf → mri_surf2surf."""
    subj_fs = FS_DIR / subject_id
    surf_dir = OUT_T1 / subject_id / 'surface'
    out_path = surf_dir / f'fsaverage5_{subject_id}_{map_name}_{hemi}.mgh'

    if is_step_done(out_path):
        return str(out_path)
    ensure_dir(surf_dir)

    # BBReg if needed
    reg_file = subj_fs / 'mri' / 'register.dat'
    if not reg_file.exists():
        run_cmd(['bbregister', '--s', subject_id, '--mov', str(vol_path),
                 '--reg', str(reg_file), '--t1'])

    # Volume → subject surface
    tmp_surf = surf_dir / f'_tmp_{map_name}_{hemi}.mgh'
    run_cmd(['mri_vol2surf', '--mov', str(vol_path), '--reg', str(reg_file),
             '--hemi', hemi, '--projfrac', '0.5', '--o', str(tmp_surf)])

    if not tmp_surf.exists():
        return None

    # Subject surface → fsaverage5
    run_cmd(['mri_surf2surf', '--srcsubject', subject_id, '--trgsubject', 'fsaverage5',
             '--hemi', hemi, '--sval', str(tmp_surf), '--tval', str(out_path),
             '--noreshape'])
    tmp_surf.unlink(missing_ok=True)

    return str(out_path) if out_path.exists() else None


def compute_coupling(subject_id, neighborhood_fwhm=15.0, min_snr=50.0):
    """
    CBF-ALFF coupling via local weighted regression on fsaverage5 surface.
    For each vertex: fit ALFF ~ CBF using Gaussian-weighted neighbors (15mm FWHM).
    Slope = coupling value.
    """
    out_dir = OUT_FMRI / subject_id
    coupling_lh = out_dir / 'coupling_lh.npy'
    coupling_rh = out_dir / 'coupling_rh.npy'

    if is_step_done(coupling_lh) and is_step_done(coupling_rh):
        return {
            'n_valid': 0,
            'mean_abs': float(np.mean(np.abs(np.concatenate([
                np.load(str(coupling_lh)), np.load(str(coupling_rh))
            ]))))
        }

    surf_dir = OUT_T1 / subject_id / 'surface'
    files = {
        'cbf_lh': surf_dir / f'fsaverage5_{subject_id}_cbf_lh.mgh',
        'cbf_rh': surf_dir / f'fsaverage5_{subject_id}_cbf_rh.mgh',
        'alff_lh': surf_dir / f'fsaverage5_{subject_id}_alff_lh.mgh',
        'alff_rh': surf_dir / f'fsaverage5_{subject_id}_alff_rh.mgh',
    }

    if not all(f.exists() for f in files.values()):
        return None

    lh_cbf = nib.freesurfer.read_morph_data(str(files['cbf_lh']))
    rh_cbf = nib.freesurfer.read_morph_data(str(files['cbf_rh']))
    lh_alff = nib.freesurfer.read_morph_data(str(files['alff_lh']))
    rh_alff = nib.freesurfer.read_morph_data(str(files['alff_rh']))

    cbf = np.concatenate([lh_cbf, rh_cbf])
    alff = np.concatenate([lh_alff, rh_alff])
    n_lh = len(lh_cbf)

    # SNR filter
    snr = np.abs(cbf) / (np.std(cbf) + 1e-10)
    valid = snr >= min_snr

    # Surface sphere coordinates
    cl, _ = nib.freesurfer.read_geometry(str(FS_DIR / 'fsaverage5' / 'surf' / 'lh.sphere'))
    cr, _ = nib.freesurfer.read_geometry(str(FS_DIR / 'fsaverage5' / 'surf' / 'rh.sphere'))
    coords = np.vstack([cl, cr])

    sigma = neighborhood_fwhm / (2 * np.sqrt(2 * np.log(2)))
    tree = cKDTree(coords)

    n = len(cbf)
    coupling = np.zeros(n, dtype=np.float32)

    t0 = time.time()
    for i in np.where(valid)[0]:
        nbrs = tree.query_ball_point(coords[i], r=3 * sigma)
        if len(nbrs) < 3:
            continue
        dist = np.sqrt(np.sum((coords[nbrs] - coords[i]) ** 2, axis=1))
        w = np.exp(-0.5 * (dist / sigma) ** 2)
        w /= (np.sum(w) + 1e-10)
        X = np.column_stack([np.ones(len(nbrs)), cbf[nbrs]])
        W = np.diag(w)
        try:
            beta = np.linalg.lstsq(W @ X, W @ alff[nbrs], rcond=None)[0]
            coupling[i] = beta[1]
        except Exception:
            pass

    elapsed = time.time() - t0

    ensure_dir(out_dir)
    np.save(str(coupling_lh), coupling[:n_lh])
    np.save(str(coupling_rh), coupling[n_lh:])

    valid_coupling = coupling[valid]
    mean_abs = float(np.mean(np.abs(valid_coupling))) if len(valid_coupling) > 0 else 0.0
    logger.info(f'Coupling done ({elapsed:.0f}s): mean_abs={mean_abs:.6f}, '
                f'n_valid={int(np.sum(valid))}')

    return {'n_valid': int(np.sum(valid)), 'mean_abs': mean_abs}
