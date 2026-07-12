"""
BOLD preprocessing (Steps 4-9).
Step 4: DICOM → NIfTI
Step 5: Motion correction (mcflirt)
Step 5b: 3dDespike
Step 6: Brain mask + WM/CSF segmentation
Step 7: 36-parameter confound regression
Step 8: Bandpass filter (0.01-0.08 Hz)
Step 9: ALFF computation
"""
import os
import logging
import time
import subprocess
import numpy as np
import nibabel as nib
from scipy import signal
from scipy.ndimage import binary_erosion
from pathlib import Path
from preprocess import (
    OUT_FMRI, DATA, FS_DIR, is_step_done, ensure_dir,
    setup_freesurfer_env, run_cmd
)

logger = logging.getLogger('preprocess.bold')


def bold_to_nifti(subject_id):
    """BOLD DICOM → 4D NIfTI via dcm2niix."""
    out = OUT_FMRI / subject_id / f'{subject_id}_BOLD.nii.gz'
    if is_step_done(out):
        return str(out)
    ensure_dir(out.parent)

    src = DATA / 'baseline_fMRI' / subject_id
    if not src.exists():
        return None

    run_cmd(['dcm2niix', '-z', 'y', '-f', f'{subject_id}_BOLD',
             '-4', 'y', '-w', '0', '-o', str(out.parent),
             '-p', 'n', '-v', '0', str(src)],
            timeout=600)

    if out.exists():
        logger.info(f'BOLD: {out.stat().st_size / 1e6:.0f}MB')
        return str(out)
    return None


def motion_correct(subject_id, bold_path):
    """Motion correction using FSL mcflirt (6-DOF rigid body)."""
    mc_path = OUT_FMRI / subject_id / f'{subject_id}_BOLD_mc.nii.gz'
    if is_step_done(mc_path):
        return str(mc_path)

    ensure_dir(mc_path.parent)
    import time as _time
    t0 = _time.time()

    run_cmd(['mcflirt', '-in', bold_path, '-out', str(mc_path),
             '-refvol', 'middle', '-plots', '-report'],
            timeout=3600)

    if mc_path.exists():
        elapsed = _time.time() - t0
        logger.info(f'MC done: {elapsed:.0f}s')
        return str(mc_path)

    # Fallback: use flirt -dof 6
    logger.warning('mcflirt failed, falling back to flirt -dof 6')
    return _flirt_motion_correct(subject_id, bold_path)


def _flirt_motion_correct(subject_id, bold_path):
    """Fallback motion correction using flirt (volume-by-volume)."""
    mc_path = OUT_FMRI / subject_id / f'{subject_id}_BOLD_mc.nii.gz'
    if is_step_done(mc_path):
        return str(mc_path)

    img = nib.load(bold_path)
    data = img.get_fdata().astype(np.float32)
    affine = img.affine
    nt = data.shape[3]
    if nt < 2:
        return bold_path

    ref_idx = nt // 2
    ref_path = mc_path.parent / '_mc_ref.nii.gz'
    nib.save(nib.Nifti1Image(data[..., ref_idx], affine), str(ref_path))

    corrected = np.zeros_like(data, dtype=np.float32)
    corrected[..., ref_idx] = data[..., ref_idx]
    mats = []

    for t in range(nt):
        if t == ref_idx:
            mats.append(np.eye(4))
            continue
        vol_t = mc_path.parent / f'_mc_vol_{t:04d}.nii.gz'
        out_t = mc_path.parent / f'_mc_vol_{t:04d}_r.nii.gz'
        mat_t = mc_path.parent / f'_mc_vol_{t:04d}.mat'
        nib.save(nib.Nifti1Image(data[..., t], affine), str(vol_t))

        run_cmd(['flirt', '-in', vol_t, '-ref', ref_path, '-out', out_t,
                 '-dof', '6', '-interp', 'trilinear', '-omat', mat_t])

        if out_t.exists():
            corrected[..., t] = nib.load(str(out_t)).get_fdata().astype(np.float32)
        else:
            corrected[..., t] = data[..., t]

        if mat_t.exists():
            mats.append(np.loadtxt(str(mat_t)))
        else:
            mats.append(np.eye(4))

        for f in [vol_t, out_t, mat_t]:
            f.unlink(missing_ok=True)

    ref_path.unlink(missing_ok=True)
    nib.save(nib.Nifti1Image(corrected, affine), str(mc_path))

    # Save motion parameters
    mats_arr = np.stack(mats)
    motion_params = mats_arr[:, :3, :].reshape(nt, 12)
    np.save(str(mc_path.parent / '_motion_params.npy'), motion_params)

    logger.info(f'MC (flirt fallback): {nt} volumes')
    return str(mc_path)


def despike(mc_path):
    """Remove intensity outliers with AFNI 3dDespike."""
    mc_path = Path(mc_path)
    despike_path = mc_path.parent / f'{mc_path.stem}_despike.nii.gz'
    if is_step_done(despike_path):
        return str(despike_path)

    afni_bin = Path.home() / 'abin' / '3dDespike'
    if not afni_bin.exists():
        logger.info('3dDespike not found, skipping')
        return str(mc_path)

    env = os.environ.copy()
    env['PATH'] = str(Path.home() / 'abin') + ':' + env.get('PATH', '')

    r = subprocess.run(
        f'3dDespike -prefix {despike_path} {mc_path}',
        shell=True, capture_output=True, text=True, env=env, timeout=300
    )

    if despike_path.exists():
        logger.info('3dDespike done')
        return str(despike_path)
    return str(mc_path)


def _resample_mask_to_bold(mask_img, bold_img):
    """Resample a mask from T1 space to BOLD space using coordinate mapping."""
    from scipy.ndimage import map_coordinates

    mask_data = mask_img.get_fdata()
    bold_data = bold_img.get_fdata()
    bold_shape = bold_data.shape[:3] if bold_data.ndim == 4 else bold_data.shape

    # Create BOLD voxel coordinates
    coords = np.meshgrid(
        np.arange(bold_shape[0]),
        np.arange(bold_shape[1]),
        np.arange(bold_shape[2]),
        indexing='ij'
    )
    bold_voxel = np.stack([c.ravel() for c in coords])  # (3, N)

    # BOLD voxel → BOLD world → T1 world → T1 voxel
    bold_affine = bold_img.affine
    mask_affine = mask_img.affine
    bold_world = bold_affine[:3, :3] @ bold_voxel + bold_affine[:3, 3:4]
    mask_voxel = np.linalg.inv(mask_affine[:3, :3]) @ (bold_world - mask_affine[:3, 3:4])

    # Resample using interpolation
    resampled = map_coordinates(mask_data, mask_voxel, order=0, mode='constant', cval=0)
    return resampled.reshape(bold_shape)


def brain_mask_and_seg(subject_id, mc_path):
    """Generate brain mask and WM/CSF segmentation from FreeSurfer."""
    mc_path = Path(mc_path)
    subj_fs = FS_DIR / subject_id
    reg_file = subj_fs / 'mri' / 'register.dat'
    out_dir = mc_path.parent

    # BBReg if needed
    if not reg_file.exists():
        run_cmd(['bbregister', '--s', subject_id, '--mov', str(mc_path),
                 '--reg', str(reg_file), '--t1'])

    brainmask_mgz = subj_fs / 'mri' / 'brainmask.mgz'
    aseg_mgz = subj_fs / 'mri' / 'aseg.mgz'
    if not brainmask_mgz.exists() or not aseg_mgz.exists():
        return None

    masks = {}
    bold_img = nib.load(str(mc_path))
    bold_affine = bold_img.affine
    bold_data = bold_img.get_fdata()
    is_4d = bold_data.ndim == 4
    bold_shape = bold_data.shape[:3] if is_4d else bold_data.shape

    # Brain mask: read from T1 space and resample to BOLD space
    brainmask_img = nib.load(str(brainmask_mgz))
    brain_resampled = _resample_mask_to_bold(brainmask_img, bold_img)

    if is_4d:
        masked = bold_data * (brain_resampled > 0.5).astype(np.float32)[..., np.newaxis]
    else:
        masked = bold_data * (brain_resampled > 0.5).astype(np.float32)
    brain_path = out_dir / f'{subject_id}_BOLD_brain.nii.gz'
    nib.save(nib.Nifti1Image(masked, bold_affine), str(brain_path))
    masks['brain'] = str(brain_path)

    # Aseg: read from T1 space and resample to BOLD space
    aseg_img = nib.load(str(aseg_mgz))
    aseg_resampled = _resample_mask_to_bold(aseg_img, bold_img).astype(int)

    wm_labels = [2, 41, 7, 46, 251, 252, 253, 254, 255]
    csf_labels = [4, 5, 14, 15, 24, 31, 43, 44, 63]
    wm_mask = binary_erosion(np.isin(aseg_resampled, wm_labels), iterations=1).astype(np.float32)
    csf_mask = binary_erosion(np.isin(aseg_resampled, csf_labels), iterations=1).astype(np.float32)
    wm_path = out_dir / '_wm_mask.nii.gz'
    csf_path = out_dir / '_csf_mask.nii.gz'
    nib.save(nib.Nifti1Image(wm_mask, bold_affine), str(wm_path))
    nib.save(nib.Nifti1Image(csf_mask, bold_affine), str(csf_path))
    masks['wm'] = str(wm_path)
    masks['csf'] = str(csf_path)

    return masks


def confound_regression(data, masks, motion_file=None):
    """36-parameter confound regression (XCP-style)."""
    nx, ny, nz, nt = data.shape
    d2 = data.reshape(-1, nt)
    confounds = []

    # WM + CSF mean signals
    for tissue in ['wm', 'csf']:
        if tissue in masks:
            mask = nib.load(masks[tissue]).get_fdata().ravel()
            if np.sum(mask > 0) > 5:
                confounds.append(np.mean(d2[mask > 0], axis=0))

    # Motion parameters
    if motion_file and os.path.exists(motion_file):
        mp = np.load(motion_file)
        if mp.shape[0] == nt:
            for i in range(min(mp.shape[1], 6)):
                confounds.append(mp[:, i])

    if not confounds:
        return data

    # Build design matrix: intercept + confounds + derivatives + quadratics
    parts = [np.ones(nt)]
    for c in confounds:
        parts.append(c - np.mean(c))
    for c in confounds:
        d = np.zeros(nt)
        d[1:] = np.diff(c)
        parts.append(d)
    for c in confounds:
        parts.append((c - np.mean(c)) ** 2)
    for c in confounds:
        d = np.zeros(nt)
        d[1:] = np.diff(c)
        parts.append(d ** 2)

    X = np.column_stack(parts)

    cleaned = np.zeros_like(d2)
    for i in range(d2.shape[0]):
        if np.std(d2[i]) < 1e-6:
            continue
        beta = np.linalg.lstsq(X, d2[i], rcond=None)[0]
        cleaned[i] = d2[i] - X @ beta

    return cleaned.reshape(nx, ny, nz, nt)


def bandpass(data, tr=2.0, low=0.01, high=0.08):
    """Butterworth bandpass filter."""
    nt = data.shape[3]
    if nt < 10:
        return data
    fs = 1.0 / tr
    sos = signal.butter(4, [low, high], btype='bandpass', fs=fs, output='sos')
    pad = max(int(3 * fs / low), 128)
    d2 = data.reshape(-1, nt)
    padded = np.pad(d2, ((0, 0), (pad, pad)), mode='constant')
    filtered = signal.sosfiltfilt(sos, padded, axis=1)[:, pad:pad + nt]
    return filtered.reshape(data.shape)


def compute_alff(data, tr=2.0):
    """ALFF = sqrt(sum of power in 0.01-0.08 Hz band)."""
    nfft = int(2 ** np.ceil(np.log2(data.shape[3])))
    psd = np.abs(np.fft.rfft(data, n=nfft, axis=3)) ** 2
    freqs = np.fft.rfftfreq(nfft, d=tr)
    lf = (freqs >= 0.01) & (freqs <= 0.08)
    return np.sqrt(np.sum(psd[..., lf], axis=-1))


def bold_to_alff(subject_id, bold_path):
    """Full BOLD preprocessing pipeline → ALFF."""
    alff_path = OUT_FMRI / subject_id / f'{subject_id}_ALFF.nii.gz'
    if is_step_done(alff_path):
    return str(alff_path)


def bold_to_fc(subject_id, bold_path, n_regions=114):
    """
    Compute FC matrix from BOLD time series + FreeSurfer parcellation.
    Used for Paper 2016 PLS analysis.
    """
    import nibabel as nib
    from scipy import stats

    img = nib.load(bold_path)
    data = img.getfdata().astype(np.float32)
    if data.ndim != 4:
        return None

    nx, ny, nz, nt = data.shape

    # Get FreeSurfer parcellation
    subj_fs = FS_DIR / subject_id
    if not (subj_fs / 'surf' / 'lh.sphere.reg').exists():
        return None

    parc_file = subj_fs / 'mri' / 'aparc+aseg.mgz'
    if not parc_file.exists():
        parc_file = subj_fs / 'mri' / 'aseg.mgz'
    if not parc_file.exists():
        return None

    parc_img = nib.load(str(parc_file))
    parc_data = parc_img.getfdata().astype(int)

    # Resample parcellation to BOLD space
    bold_affine = img.affine
    bold_shape = data.shape[:3]
    coords = np.meshgrid(
        np.arange(bold_shape[0]), np.arange(bold_shape[1]),
        np.arange(bold_shape[2]), indexing='ij'
    )
    bold_voxel = np.stack([c.ravel() for c in coords])
    bold_world = bold_affine[:3, :3] @ bold_voxel + bold_affine[:3, 3:4]
    parc_voxel = np.linalg.inv(parc_img.affine[:3, :3]) @ (
        bold_world - parc_img.affine[:3, 3:4]
    )
    from scipy.ndimage import map_coordinates
    parc_bold = map_coordinates(parc_data, parc_voxel, order=0,
                                mode='constant', cval=0).reshape(bold_shape)

    # Extract regional time series
    unique_rois = np.unique(parc_bold)
    unique_rois = unique_rois[(unique_rois != 0) & (unique_rois < 1000)]

    regional_ts = np.zeros((len(unique_rois), nt), dtype=np.float32)
    for i, roi in enumerate(unique_rois):
        mask = parc_bold == roi
        if np.sum(mask) > 0:
            regional_ts[i] = np.mean(data[mask], axis=0)

    # Detrend
    t = np.arange(nt, dtype=np.float32)
    t_n = (t - t.mean()) / (t.std() + 1e-10)
    conf = np.column_stack([np.ones(nt), t_n])
    beta = np.linalg.lstsq(conf, regional_ts.T, rcond=None)[0]
    detrended = regional_ts - (conf @ beta).T

    # FC matrix
    fc_matrix = np.corrcoef(detrended)
    np.fill_diagonal(fc_matrix, 0)

    # Save
    out_dir = OUT_DWI / subject_id
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(str(out_dir / 'FC_matrix.npy'), fc_matrix)
    logger.info(f'FC matrix: {fc_matrix.shape}')
    return fc_matrix

    # Motion correction
    mc_path = motion_correct(subject_id, bold_path)

    # Despike
    mc_path = despike(mc_path)

    # Brain mask + segmentation
    masks = brain_mask_and_seg(subject_id, mc_path)

    # Load data
    if masks and 'brain' in masks:
        data = nib.load(masks['brain']).get_fdata().astype(np.float32)
    else:
        data = nib.load(mc_path).get_fdata().astype(np.float32)

    # Confound regression
    motion_file = str(Path(mc_path).parent / '_motion_params.npy')
    data_clean = confound_regression(data, masks, motion_file)

    # Bandpass filter
    data_filt = bandpass(data_clean)

    # Save preprocessed BOLD
    affine = nib.load(mc_path).affine
    preproc_path = OUT_FMRI / subject_id / f'{subject_id}_BOLD_preproc.nii.gz'
    nib.save(nib.Nifti1Image(data_filt, affine), str(preproc_path))

    # Compute ALFF
    alff = compute_alff(data_filt)
    nib.save(nib.Nifti1Image(alff, affine), str(alff_path))
    logger.info(f'ALFF: mean={alff.mean():.2f}')
    return str(alff_path)
