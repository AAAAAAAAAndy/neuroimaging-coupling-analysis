"""
ASL → CBF preprocessing (Step 1).
Paper 2022: pCASL quantification using single-compartment model.
"""
import logging
import numpy as np
import nibabel as nib
import pydicom
from pathlib import Path
from preprocess import BASE, OUT_ASL, DATA_ASL, DATA, TIMEPOINT, is_step_done, ensure_dir

logger = logging.getLogger('preprocess.asl')


def _find_asl_dir(subject_id):
    """Find ASL DICOM directory across standard, special, and visit locations.
    Returns (path, mode) where mode is:
        'standard' - flat DICOM (baseline_ASL)
        'special'  - mosaic in baseline_ASL_special/*/
        'visit_derived' - DERIVED CBF in visit_ASL/*/CBF_*/
        'visit_raw' - raw ASL in visit_ASL/*/3D ASL*/
    """
    # Standard location
    standard = DATA / f'{TIMEPOINT}_ASL' / subject_id
    if standard.exists():
        if standard.is_dir():
            # Check contents: DICOM files (baseline) or subdirectories (visit)
            has_dicom = False
            for f in standard.iterdir():
                if f.is_file() and (f.name.startswith('Z') or f.name.endswith('.dcm') or f.name[0].isdigit()):
                    has_dicom = True
                    break

            if has_dicom:
                return standard, 'standard'

            # Visit format: contains subdirectories
            for sub in sorted(standard.iterdir()):
                if sub.is_dir() and 'CBF' in sub.name.upper():
                    return sub, 'visit_derived'
            for sub in sorted(standard.iterdir()):
                if sub.is_dir() and 'ASL' in sub.name.upper():
                    return sub, 'visit_raw'

    # ASL_special: prefer M0 over iso
    special_dir = DATA / f'{TIMEPOINT}_ASL_special'
    if special_dir.exists():
        for name in ['ASL_3D_tra_M0', 'ASL_3D_tra_iso']:
            candidate = special_dir / name / subject_id
            if candidate.exists():
                return candidate, 'special'

    return None, None


def _copy_derived_cbf(dcm_dir, out):
    """Copy scanner-derived CBF to NIfTI."""
    files = sorted(dcm_dir.iterdir())
    if not files:
        return None
    slices = []
    for f in files:
        try:
            ds = pydicom.dcmread(str(f), force=True)
            slices.append(ds.pixel_array.astype(np.float32))
        except Exception:
            continue
    if not slices:
        return None
    vol = np.stack(slices, axis=-1)
    ds0 = pydicom.dcmread(str(files[0]), force=True)
    aff = np.diag([
        float(ds0.PixelSpacing[0]), float(ds0.PixelSpacing[1]),
        float(getattr(ds0, 'SliceThickness', 4)), 1.0
    ])
    nib.save(nib.Nifti1Image(vol, aff), str(out))
    logger.info(f'CBF (visit DERIVED): {vol.shape}, mean={vol.mean():.1f}')
    return str(out)


def _mosaic_to_cbf(dcm_dir, out):
    """Decode mosaic ASL and compute CBF."""
    from preprocess.asl_special import decode_mosaic

    files = sorted(dcm_dir.iterdir())
    if len(files) < 2:
        return None

    volumes = []
    for f in files:
        try:
            vol, ds = decode_mosaic(f)
            volumes.append(vol)
        except Exception as e:
            logger.warning(f'Mosaic decode failed for {f}: {e}')
            continue

    if len(volumes) < 2:
        return None

    m0 = volumes[0]
    perf = np.mean(volumes[1:], axis=0)

    PLD, tau, T1a = 1.5, 1.5, 1.65
    alpha, lam = 0.85, 0.9
    unit_convert = 6000.0

    m0_safe = np.where(m0 > 0, m0, np.nan)
    cbf = unit_convert * perf * lam * np.exp(PLD / T1a) / \
          (2.0 * alpha * T1a * (1.0 - np.exp(-tau / T1a)) * m0_safe)
    cbf = np.nan_to_num(cbf, nan=0.0, posinf=0.0, neginf=0.0)
    cbf = np.clip(cbf, 0, 200)

    ds0 = pydicom.dcmread(str(files[0]), force=True)
    pix = [float(x) for x in getattr(ds0, 'PixelSpacing', [3, 3])]
    aff = np.diag([pix[0], pix[1], float(getattr(ds0, 'SliceThickness', 4)), 1.0])
    nib.save(nib.Nifti1Image(cbf, aff), str(out))
    logger.info(f'CBF (mosaic): {cbf.shape}, mean={np.nanmean(cbf[m0>0]):.1f}')
    return str(out)


def asl_to_cbf(subject_id):
    """
    Convert ASL DICOM to CBF NIfTI.
    Handles standard (DERIVED/pCASL), mosaic (ASL_special), and visit DERIVED.
    Output path mirrors data directory structure.
    """
    asl_dir, mode = _find_asl_dir(subject_id)
    if not asl_dir:
        return None

    rel_path = asl_dir.relative_to(DATA)
    out_dir = BASE / 'output' / rel_path
    out = out_dir / f'{subject_id}_CBF.nii.gz'
    if is_step_done(out):
        return str(out)
    ensure_dir(out_dir)

    if mode == 'visit_derived':
        return _copy_derived_cbf(asl_dir, out)
    if mode == 'visit_raw':
        return _mosaic_to_cbf(asl_dir, out)
    if mode == 'special':
        return _mosaic_to_cbf(asl_dir, out)

    result = _try_derived_cbf(asl_dir, out)
    if result:
        return result
    return _compute_pcasl_cbf(asl_dir, out)


def _try_derived_cbf(asl_dir, out):
    """Extract scanner-computed CBF from DERIVED PERFUSION images."""
    deriv = []
    for f in sorted(asl_dir.iterdir()):
        try:
            ds = pydicom.dcmread(str(f), stop_before_pixels=True)
            it = [str(x) for x in ds.ImageType]
            if 'DERIVED' in it and any('PERFUSION' in x for x in it):
                deriv.append((int(ds.InstanceNumber), f))
        except Exception:
            continue

    if not deriv:
        return None

    deriv.sort()
    vol = np.stack(
        [pydicom.dcmread(str(f)).pixel_array.astype(np.float32) for _, f in deriv],
        axis=-1
    )
    ds0 = pydicom.dcmread(str(deriv[0][1]))
    aff = np.diag([
        float(ds0.PixelSpacing[0]),
        float(ds0.PixelSpacing[1]),
        float(ds0.SliceThickness),
        1.0
    ])
    nib.save(nib.Nifti1Image(vol, aff), str(out))
    logger.info(f'CBF (DERIVED): {vol.shape}, mean={vol.mean():.1f}')
    return str(out)


def _compute_pcasl_cbf(asl_dir, out):
    """
    Compute CBF from control-label pairs using pCASL formula.
    f = 6000 * ΔM * λ * exp(PLD/T1a) / (2 * α * T1a * (1 - exp(-τ/T1a)) * M0)
    Units: mL/100g/min
    """
    files = sorted(asl_dir.iterdir())
    volumes = []
    for f in files:
        try:
            ds = pydicom.dcmread(str(f), stop_before_pixels=False)
            volumes.append(ds.pixel_array.astype(np.float32))
        except Exception:
            continue

    if len(volumes) < 4:
        return None

    data = np.stack(volumes, axis=-1)
    if data.ndim == 3:
        data = data[:, :, np.newaxis, :]

    nt = data.shape[3]

    # pCASL parameters (GE 3D pCASL)
    PLD = 2.0       # Post-labeling delay (s)
    tau = 1.5        # Label duration (s)
    T1a = 1.65       # Blood T1 at 3T (s)
    alpha = 0.85     # Labeling efficiency
    lam = 0.9        # Partition coefficient (mL/g)
    unit_convert = 6000.0

    # Split control/label pairs
    if nt % 2 == 0:
        ctrl = data[..., 0::2]
        label = data[..., 1::2]
    else:
        ctrl = data[..., 1::2]
        label = data[..., 2::2]
    n_pairs = min(ctrl.shape[3], label.shape[3])
    delta = np.mean(ctrl[..., :n_pairs] - label[..., :n_pairs], axis=3)

    m0 = np.median(data, axis=3)
    m0_safe = np.where(m0 > 0, m0, np.nan)

    cbf = unit_convert * delta * lam * np.exp(PLD / T1a) / \
          (2 * alpha * T1a * (1 - np.exp(-tau / T1a)) * m0_safe)
    cbf = np.nan_to_num(cbf, nan=0.0, posinf=0.0, neginf=0.0)
    cbf = np.clip(cbf, 0, 200)

    ds0 = pydicom.dcmread(str(files[0]))
    aff = np.diag([
        float(ds0.PixelSpacing[0]),
        float(ds0.PixelSpacing[1]),
        float(ds0.SliceThickness),
        1.0
    ])
    nib.save(nib.Nifti1Image(cbf, aff), str(out))
    logger.info(f'CBF (pCASL): {cbf.shape}, mean={np.nanmean(cbf[m0 > 0]):.1f}')
    return str(out)
