"""
ASL → CBF preprocessing (Step 1).
Paper 2022: pCASL quantification using single-compartment model.
"""
import logging
import numpy as np
import nibabel as nib
import pydicom
from pathlib import Path
from preprocess import OUT_ASL, DATA_ASL, is_step_done, ensure_dir

logger = logging.getLogger('preprocess.asl')


def asl_to_cbf(subject_id):
    """
    Convert ASL DICOM to CBF NIfTI.
    Strategy: prefer DERIVED perfusion images (scanner-computed CBF).
    Fallback: compute CBF from control-label pairs using pCASL formula.
    """
    out = OUT_ASL / subject_id / f'{subject_id}_CBF.nii.gz'
    if is_step_done(out):
        return str(out)
    ensure_dir(out.parent)

    asl_dir = DATA_ASL / subject_id
    if not asl_dir.exists():
        return None

    # Try DERIVED images first
    result = _try_derived_cbf(asl_dir, out)
    if result:
        return result

    # Fallback: pCASL formula
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
