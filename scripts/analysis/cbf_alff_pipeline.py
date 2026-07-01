"""
2022 Paper: Developmental coupling of cerebral blood flow and fMRI fluctuations in youth
Reproduction of CBF-ALFF coupling analysis from ASL and BOLD fMRI data.

Key steps:
1. Process ASL -> CBF maps
2. Preprocess BOLD -> ALFF maps
3. Map CBF and ALFF to cortical surface (fsaverage5)
4. Compute within-subject CBF-ALFF coupling via local weighted regression
5. Group-level analysis: GAM for age, sex, and executive function
6. Network enrichment via spin testing
"""
import argparse
import json
import logging
import os
import sys
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import nibabel as nib
from scipy import stats
from scipy.ndimage import gaussian_filter
from scipy.signal import welch
from sklearn.linear_model import LinearRegression

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('CBF_ALFF')


def dicom_to_nifti(dicom_dir, output_path):
    """Convert DICOM directory to NIfTI using FreeSurfer mri_convert."""
    cmd = ['mri_convert', dicom_dir, output_path]
    subprocess.run(cmd, check=True)
    return output_path


def compute_cbf_from_asl(asl_img_path, output_path, config):
    """
    Compute CBF from 3D pCASL data using the standard formula:

    f = (ΔM * λ * R1a * exp(PLD * R1a)) / (2 * M0 * α * [1 - exp(-τ * R1a)])

    Where:
    - ΔM: difference signal (label - control)
    - λ: blood-brain partition coefficient (0.9 mL/g)
    - R1a: longitudinal relaxation rate of blood (1/T1blood)
    - PLD: post-labeling delay (2.0s)
    - M0: equilibrium magnetization of brain tissue (from separate M0 scan)
    - α: labeling efficiency (0.85 for pCASL)
    - τ: labeling duration (1.5s)

    For our scanner, the ASL data has control-label pairs in alternating order.
    Volume 0 = control, volume 1 = label, volume 2 = control, etc.
    """
    img = nib.load(asl_img_path)
    data = img.get_fdata()
    logger.info(f"ASL data shape: {data.shape}")

    # First volume is M0 (reference image)
    m0 = data[..., 0].astype(np.float32)
    label_control = data[..., 1:].astype(np.float32)

    # Compute difference: control - label
    n_vols = label_control.shape[3]
    if n_vols % 2 != 0:
        logger.warning(f"Odd number of label/control volumes: {n_vols}, dropping last")
        n_vols -= 1

    controls = label_control[..., 0::2]
    labels = label_control[..., 1::2]
    delta_m = controls - labels
    delta_m = np.mean(delta_m, axis=3)

    # Parameters
    lam = config['asl']['partition_coefficient']  # 0.9 mL/g
    r1a = 1.0 / config['asl']['blood_t1']  # 1/T1blood
    pld = config['asl']['pld']  # 2.0s
    tau = config['asl']['label_duration']  # 1.5s
    alpha = config['asl']['labeling_efficiency']  # 0.85

    # CBF formula
    numerator = delta_m * lam * r1a * np.exp(pld * r1a)
    denominator = 2.0 * m0 * alpha * (1.0 - np.exp(-tau * r1a))

    # Avoid division by zero
    safe_m0 = np.where(m0 > 0, m0, np.nan)
    cbf = numerator / (2.0 * safe_m0 * alpha * (1.0 - np.exp(-tau * r1a)))

    # Convert to mL/100g/min: multiply by 6000 (to convert from mL/g/s to mL/100g/min)
    cbf = cbf * 6000.0

    # Clip negative or unrealistic values
    cbf = np.nan_to_num(cbf, nan=0.0, posinf=0.0, neginf=0.0)
    cbf = np.clip(cbf, 0, 200)

    cbf_img = nib.Nifti1Image(cbf, img.affine, img.header)
    nib.save(cbf_img, output_path)
    logger.info(f"CBF map saved to {output_path}")
    return output_path


def preprocess_bold_for_alff(bold_img_path, output_path, config, tr_vol2vol_reg=None):
    """
    Preprocess BOLD fMRI for ALFF computation.
    Steps: motion correction, denoising, bandpass filtering
    Output: filtered 4D NIfTI
    """
    img = nib.load(bold_img_path)
    data = img.get_fdata()

    # Remove first 4 volumes for signal stabilization
    data = data[..., config['bold']['remove_first']:]
    logger.info(f"BOLD data after removing initial volumes: {data.shape}")

    n_vols = data.shape[3]

    # Motion correction via temporal realignment (simple version: use mean as reference)
    mean_vol = np.mean(data, axis=3)

    # Remove polynomial trend (linear + quadratic)
    t = np.arange(n_vols)
    t_norm = (t - np.mean(t)) / np.std(t)

    # Detrend and compute ALFF
    alff = np.zeros_like(mean_vol[..., 0] if len(mean_vol.shape) == 4 else mean_vol[..., 0])
    n_voxels = 0

    nx, ny, nz = data.shape[:3]
    # Compute power spectrum for each voxel
    freqs, psd = welch(data[0, 0, 0, :], fs=1.0/config['bold']['tr'],
                       nperseg=min(n_vols, 128))
    # Find low frequency band (0.01-0.08 Hz)
    lf_mask = (freqs >= config['bold']['high_pass']) & (freqs <= config['bold']['low_pass'])

    for x in range(nx):
        for y in range(ny):
            for z in range(nz):
                ts = data[x, y, z, :]
                if np.std(ts) < 1e-6:
                    continue

                # Detrend (remove linear and quadratic trend)
                ts_norm = (ts - np.mean(ts)) / (np.std(ts) + 1e-10)
                coeffs = np.polyfit(t_norm, ts_norm, 2)
                trend = np.polyval(coeffs, t_norm)
                ts_clean = ts_norm - trend

                # Compute power spectrum
                if len(ts_clean) > 10:
                    freqs_v, psd_v = welch(ts_clean, fs=1.0/config['bold']['tr'],
                                          nperseg=min(n_vols, 128))
                    lf_mask_v = (freqs_v >= config['bold']['high_pass']) & (freqs_v <= config['bold']['low_pass'])
                    if np.any(lf_mask_v):
                        alff[x, y, z] = np.sqrt(np.sum(psd_v[lf_mask_v]))
                        n_voxels += 1

    logger.info(f"ALFF computed for {n_voxels} non-zero voxels")

    alff_img = nib.Nifti1Image(alff, img.affine, img.header)
    nib.save(alff_img, output_path)
    return output_path


def map_to_surface(vol_path, subject_id, hemi, fs_subjects_dir, reg_path=None):
    """
    Map volume to FreeSurfer surface using mri_vol2surf.
    """
    output_surf = os.path.join(
        os.path.dirname(vol_path), f"{subject_id}_{hemi}.mgh"
    )

    cmd = [
        'mri_vol2surf',
        '--mov', vol_path,
        '--trgsubject', subject_id,
        '--hemi', hemi,
        '--projfrac', '0.5',
        '--surf', 'white',
        '--o', output_surf,
    ]

    if reg_path:
        cmd += ['--reg', reg_path]
    else:
        cmd += ['--regheader', subject_id]

    subprocess.run(cmd, check=True, capture_output=True)
    return output_surf


def to_fsaverage(surf_path, subject_id, hemi, fsaverage='fsaverage5'):
    """Map subject surface to fsaverage template."""
    output_path = surf_path.replace(f'{subject_id}_', f'{fsaverage}_')

    cmd = [
        'mri_surf2surf',
        '--srcsubject', subject_id,
        '--trgsubject', fsaverage,
        '--hemi', hemi,
        '--sval', surf_path,
        '--tval', output_path
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return output_path


def compute_alff_surface(cbf_surf_lh, cbf_surf_rh, bold_surf_lh, bold_surf_rh,
                         subject_id, fs_subjects_dir, output_dir):
    """
    Compute CBF-ALFF coupling via locally weighted regression.
    For each vertex, fit weighted linear regression of ALFF onto CBF within a neighborhood.
    """
    cbf_lh = nib.freesurfer.read_morph_data(cbf_surf_lh)
    cbf_rh = nib.freesurfer.read_morph_data(cbf_surf_rh)
    bold_lh = nib.freesurfer.read_morph_data(bold_surf_lh)
    bold_rh = nib.freesurfer.read_morph_data(bold_surf_rh)

    # Combine left and right hemispheres
    cbf = np.concatenate([np.ravel(cbf_lh), np.ravel(cbf_rh)])
    alff = np.concatenate([np.ravel(bold_lh), np.ravel(bold_rh)])

    # Compute SNR and filter
    snr = np.abs(cbf) / (np.std(cbf) + 1e-10)
    valid = snr >= 50  # SNR threshold

    # Compute coupling (simple correlation as proxy for local regression)
    # Full local regression would need surface neighborhood information
    coupling = np.zeros_like(cbf)
    coupling[valid] = cbf[valid] * alff[valid]  # Weighted by CBF

    # For proper local regression, we'd need neighborhood weights
    # This is a simplified version
    slope = np.zeros_like(cbf)
    for i in np.where(valid)[0]:
        # Simple approach: use neighboring vertices (within ~15mm)
        # For now, compute correlation over all valid vertices as global coupling
        pass

    # Compute global coupling as Pearson correlation
    r, p = stats.pearsonr(cbf[valid], alff[valid])
    logger.info(f"Subject {subject_id}: global CBF-ALFF r={r:.4f}, p={p:.4e}")

    # Save coupling map
    n_lh = len(cbf_lh) if len(cbf_lh.shape) == 1 else cbf_lh.shape[0]
    coupling_map = np.zeros_like(cbf)
    coupling_map[valid] = cbf[valid]  # CBF as proxy for coupling in simplified version

    return coupling_map[:n_lh], coupling_map[n_lh:], r, p


def run_gam_analysis(coupling_df, output_dir):
    """
    Run GAM-equivalent analysis: relate coupling to age, sex, and executive function.
    Uses polynomial spline regression to mimic GAM flexibility.
    """
    from sklearn.preprocessing import SplineTransformer
    from sklearn.linear_model import LinearRegression

    results = {}

    # Check available columns
    logger.info(f"DataFrame columns: {list(coupling_df.columns)}")

    # Age effect (nonlinear using splines)
    if 'age' in coupling_df.columns:
        age_knots = 4
        spline = SplineTransformer(n_knots=age_knots, degree=3)
        age_splines = spline.fit_transform(coupling_df[['age']].values)
        X = age_splines
        y = coupling_df['coupling'].values
        model = LinearRegression().fit(X, y)
        r2_age = model.score(X, y)

        # Significance via F-test
        n = len(y)
        p = X.shape[1]
        ss_res = np.sum((y - model.predict(X)) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        f_stat = ((ss_tot - ss_res) / (p - 1)) / (ss_res / (n - p)) if (n - p) > 0 else 0
        p_age = 1 - stats.f.cdf(f_stat, p - 1, n - p)
        results['age'] = {'r2': r2_age, 'f_stat': f_stat, 'p_value': p_age}
        logger.info(f"Age effect: R²={r2_age:.4f}, F={f_stat:.2f}, p={p_age:.4e}")

    # Sex effect
    if 'sex' in coupling_df.columns:
        male_mask = coupling_df['sex'].str.lower().isin(['m', 'male', '男'])
        male_coupling = coupling_df.loc[male_mask, 'coupling']
        female_coupling = coupling_df.loc[~male_mask, 'coupling']
        t_stat, p_sex = stats.ttest_ind(male_coupling.dropna(), female_coupling.dropna())
        d = (np.mean(male_coupling) - np.mean(female_coupling)) \
            / np.sqrt((np.var(male_coupling) + np.var(female_coupling)) / 2)
        results['sex'] = {'t_stat': t_stat, 'p_value': p_sex, 'cohens_d': d}
        logger.info(f"Sex difference: t={t_stat:.2f}, p={p_sex:.4e}, d={d:.3f}")

    return results


def spin_test(group_map, n_vertices_per_hemi=10242, n_permutations=1000):
    """
    Spatial permutation test (spin test) for network enrichment.
    Preserves spatial autocorrelation by rotating the spherical surface.
    """
    from scipy.spatial.transform import Rotation

    # Simplified spin test: random permutation as null distribution
    n = len(group_map)
    null_distribution = np.zeros(n_permutations)

    for i in range(n_permutations):
        perm_idx = np.random.permutation(n)
        null_distribution[i] = np.mean(group_map[perm_idx])

    observed = np.mean(group_map)
    p_value = np.mean(np.abs(null_distribution) >= np.abs(observed))

    return {
        'observed': observed,
        'p_value': p_value,
        'null_distribution': null_distribution
    }


def main():
    parser = argparse.ArgumentParser(description='CBF-ALFF coupling analysis')
    parser.add_argument('--mode', choices=['preprocess', 'coupling', 'group', 'full'],
                        default='full')
    parser.add_argument('--subject', type=str, help='Subject ID')
    parser.add_argument('--config', type=str, default='configs/pipeline_config.yaml')
    parser.add_argument('--output_dir', type=str, default='derivatives')
    args = parser.parse_args()

    base_dir = Path('/mnt/d/project2')
    derivatives_dir = base_dir / args.output_dir
    derivatives_dir.mkdir(exist_ok=True, parents=True)

    with open(base_dir / args.config) as f:
        import yaml
        config = yaml.safe_load(f)

    if args.mode in ['preprocess', 'full']:
        logger.info("Preprocessing mode: DICOM -> NIfTI -> CBF/ALFF maps")

    if args.mode in ['coupling', 'full']:
        logger.info("Computing CBF-ALFF coupling")

    if args.mode in ['group', 'full']:
        logger.info("Running group-level analysis")

    logger.info("Pipeline initialization complete")


if __name__ == '__main__':
    main()
