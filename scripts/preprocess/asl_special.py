"""
ASL_special MOSAIC decoder for baseline_ASL_special data.
Siemens mosaic format: single 2D image contains all slices in a grid.
"""
import logging
import numpy as np
import pydicom

logger = logging.getLogger('preprocess.asl_special')


def decode_mosaic(dcm_path):
    """Decode a Siemens mosaic DICOM into a 3D volume."""
    ds = pydicom.dcmread(str(dcm_path), force=True)
    pixel_array = ds.pixel_array.astype(np.float32)

    # Mosaic grid dimensions
    rows = int(getattr(ds, 'Rows', pixel_array.shape[0]))
    cols = int(getattr(ds, 'Columns', pixel_array.shape[1]))

    # From CSA header or compute from acquisition matrix
    acq_matrix = getattr(ds, 'AcquisitionMatrix', [0, 0, 0, 0])
    freq_encode = int(acq_matrix[0]) or int(acq_matrix[3])
    phase_encode = int(acq_matrix[1]) or int(acq_matrix[2])

    if not freq_encode or not phase_encode:
        # Fallback: infer from image size (assume square slices)
        n_slices = int(getattr(ds, 'NumberOfSlices', 0)) or int(getattr(ds, 'Private_0051_100b', [0])[0])
        if n_slices:
            # Find tile arrangement
            for n_cols in range(1, 20):
                if cols % n_cols == 0:
                    slice_w = cols // n_cols
                    if slice_w * n_slices <= rows * n_cols:
                        n_rows = (n_slices + n_cols - 1) // n_cols
                        if slice_w * n_rows <= rows:
                            freq_encode = slice_w
                            phase_encode = slice_w
                            break
        else:
            # Assume square slices
            n_cols = cols // int(np.sqrt(cols))
            n_rows = rows // int(np.sqrt(cols))
            freq_encode = cols // n_cols
            phase_encode = rows // n_rows

    # Tile dimensions
    slice_h, slice_w = phase_encode, freq_encode
    n_cols = cols // slice_w if slice_w > 0 else 1
    n_rows = rows // slice_h if slice_h > 0 else 1

    if n_cols == 0 or n_rows == 0:
        raise ValueError(f'Cannot decode mosaic: image={rows}x{cols}, acq_matrix={acq_matrix}')

    # Extract tiles
    n_tiles = n_rows * n_cols
    volumes = []
    for r in range(n_rows):
        for c in range(n_cols):
            tile = np.zeros((slice_h, slice_w), dtype=np.float32)
            src_r = r * slice_h
            src_c = c * slice_w
            if src_r + slice_h <= rows and src_c + slice_w <= cols:
                tile = pixel_array[src_r:src_r + slice_h, src_c:src_c + slice_w]
            volumes.append(tile)

    # Stack into 3D
    vol_3d = np.stack(volumes, axis=-1)  # (H, W, N_slices)

    logger.info(f'Mosaic decoded: {rows}x{cols} -> {slice_h}xslice_w x{n_tiles} slices')
    return vol_3d, ds


def asl_special_to_cbf(subject_id, asl_dir, output_path):
    """
    Compute CBF from ASL_special mosaic data.
    Only 2 DICOM files: InstanceNumber=1 (M0) and InstanceNumber=41 (perfusion ΔM).
    """
    from pathlib import Path
    asl_dir = Path(asl_dir)
    files = sorted(asl_dir.iterdir())

    if len(files) < 2:
        logger.warning(f'ASL_special: only {len(files)} files, need 2')
        return None

    m0_vol = None
    perf_vol = None

    for f in files:
        try:
            ds = pydicom.dcmread(str(f), stop_before_pixels=True)
            inst = int(getattr(ds, 'InstanceNumber', 0))
            if inst == 1:
                m0_vol, _ = decode_mosaic(f)
            elif inst > 1:
                perf_vol, ds_perf = decode_mosaic(f)
        except Exception as e:
            logger.warning(f'Failed to decode {f}: {e}')
            continue

    if m0_vol is None or perf_vol is None:
        logger.warning('ASL_special: missing M0 or perfusion volume')
        return None

    # pCASL parameters for GE 3D ASL (different from standard PLD)
    PLD = 1.5       # Post-labeling delay for this sequence (s)
    tau = 1.5       # Label duration (s)
    T1a = 1.65      # Blood T1 at 3T (s)
    alpha = 0.85    # Labeling efficiency
    lam = 0.9       # Blood-brain partition coefficient (mL/g)
    unit_convert = 6000.0

    # CBF = 6000 * ΔM * λ * exp(PLD/T1a) / (2 * α * T1a * (1 - exp(-τ/T1a)) * M0)
    m0_safe = np.where(m0_vol > 0, m0_vol, np.nan)
    cbf = unit_convert * perf_vol * lam * np.exp(PLD / T1a) / \
          (2.0 * alpha * T1a * (1.0 - np.exp(-tau / T1a)) * m0_safe)
    cbf = np.nan_to_num(cbf, nan=0.0, posinf=0.0, neginf=0.0)
    cbf = np.clip(cbf, 0, 200)

    # Build affine from DICOM
    if ds_perf:
        pix_spacing = [float(x) for x in getattr(ds_perf, 'PixelSpacing', [3, 3])]
        slice_thick = float(getattr(ds_perf, 'SliceThickness', 4))
        affine = np.diag([pix_spacing[0], pix_spacing[1], slice_thick, 1.0])
    else:
        affine = np.eye(4)

    import nibabel as nib
    nib.save(nib.Nifti1Image(cbf, affine), str(output_path))
    logger.info(f'ASL_special CBF: {cbf.shape}, mean={np.nanmean(cbf[m0_vol>0]):.1f}')
    return str(output_path)
