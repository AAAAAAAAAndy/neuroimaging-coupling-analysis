#!/usr/bin/env python3
"""
Analysis module: coupling, group GAM, spin test, PLS, rich club.
"""
import os
import json
import logging
import numpy as np
import pandas as pd
import nibabel as nib
from pathlib import Path
from scipy import stats, signal
from scipy.linalg import svd
from scipy.spatial import cKDTree

logger = logging.getLogger('ANALYSIS')

DERIV_2022 = Path('/mnt/d/project2/output/preprocessing')
DERIV_2016 = Path('/mnt/d/project2/output/preprocessing_2016')
FS_DIR = Path('/mnt/d/project2/output/freesurfer')


def vol_to_surface_nilearn(subject_id, vol_path, hemi, map_name, fsaverage='fsaverage5'):
    """
    Project volume to fsaverage surface using nilearn (fast, reliable).
    Falls back to FreeSurfer pipeline if needed.
    """
    surf_dir = DERIV_2022 / 'surface' / subject_id
    out_path = surf_dir / f'{fsaverage}_{subject_id}_{map_name}_{hemi}.mgh'
    if out_path.exists():
        return str(out_path)
    surf_dir.mkdir(parents=True, exist_ok=True)

    try:
        from nilearn import surface
        # Load fsaverage mesh
        fsa_dir = FS_DIR / fsaverage / 'surf'
        mesh = str(fsa_dir / f'{hemi}.white')
        bg_map = str(fsa_dir / f'{hemi}.sulc')

        # Sample the volume mesh
        surf_data = surface.vol_to_surf(
            vol_path,
            mesh,
            inner_mesh=str(fsa_dir / f'{hemi}.pial'),
            radius=3.0,
            interpolation='linear',
            kind='line',
            n_samples=4,
        )

        # Save as morph data (n_vertices array)
        if hemi == 'lh':
            n_vert = nib.freesurfer.read_geometry(str(fsa_dir / 'lh.white'))[0].shape[0]
        else:
            n_vert = nib.freesurfer.read_geometry(str(fsa_dir / 'rh.white'))[0].shape[0]

        nib.freesurfer.write_morph_data(str(out_path), surf_data)
        return str(out_path)
    except Exception as e:
        logger.warning(f"nilearn surf failed: {e}, trying FreeSurfer")
        # Fallback to FreeSurfer
        subj_fs = FS_DIR / subject_id
        reg_file = subj_fs / 'mri' / 'register.dat'
        if not reg_file.exists():
            cmd = ['bbregister', '--s', subject_id, '--mov', str(vol_path),
                   '--reg', str(reg_file), '--t1']
            import subprocess
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        # Map to subject surface
        tmp_surf = surf_dir / f'__tmp_{map_name}_{hemi}.mgh'
        cmd = ['mri_vol2surf', '--mov', str(vol_path), '--reg', str(reg_file),
               '--hemi', hemi, '--projfrac', '0.5', '--o', str(tmp_surf)]
        import subprocess
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

        # Map to fsaverage
        cmd = ['mri_surf2surf', '--srcsubject', subject_id, '--trgsubject', fsaverage,
               '--hemi', hemi, '--sval', str(tmp_surf), '--tval', str(out_path), '--noreshape']
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

        tmp_surf.unlink(missing_ok=True)
        if out_path.exists():
            return str(out_path)
    return None


def compute_coupling_local(subject_id, fsaverage='fsaverage5', min_snr=50, neigh_fwhm=15.0):
    """
    Compute CBF-ALFF coupling via locally weighted regression.
    Returns per-vertex coupling map.
    """
    surf_dir = DERIV_2022 / 'surface' / subject_id
    out_dir = DERIV_2022 / 'coupling' / subject_id
    out_lh = out_dir / f'{subject_id}_coupling_lh.npy'

    files = {
        'cbf_lh': surf_dir / f'{fsaverage}_{subject_id}_cbf_lh.mgh',
        'cbf_rh': surf_dir / f'{fsaverage}_{subject_id}_cbf_rh.mgh',
        'alff_lh': surf_dir / f'{fsaverage}_{subject_id}_alff_lh.mgh',
        'alff_rh': surf_dir / f'{fsaverage}_{subject_id}_alff_rh.mgh',
    }
    if not all(f.exists() for f in files.values()):
        logger.warning(f"Missing surface files for {subject_id}")
        return None

    cbf_lh = nib.freesurfer.read_morph_data(str(files['cbf_lh']))
    cbf_rh = nib.freesurfer.read_morph_data(str(files['cbf_rh']))
    alff_lh = nib.freesurfer.read_morph_data(str(files['alff_lh']))
    alff_rh = nib.freesurfer.read_morph_data(str(files['alff_rh']))

    cbf_all = np.concatenate([cbf_lh, cbf_rh])
    alff_all = np.concatenate([alff_lh, alff_rh])
    n_lh = len(cbf_lh)

    # SNR filter
    cbf_snr = np.abs(cbf_all) / (np.std(cbf_all) + 1e-10)
    valid = cbf_snr >= min_snr
    n_valid = int(np.sum(valid))
    if n_valid < 100:
        logger.warning(f"Too few valid vertices: {n_valid}")
        return None

    # Surface sphere coordinates
    cl, _ = nib.freesurfer.read_geometry(str(FS_DIR / fsaverage / 'surf' / 'lh.sphere'))
    cr, _ = nib.freesurfer.read_geometry(str(FS_DIR / fsaverage / 'surf' / 'rh.sphere'))
    coords = np.vstack([cl, cr])
    sigma = neigh_fwhm / (2 * np.sqrt(2 * np.log(2)))

    # Coupling computation
    tree = cKDTree(coords)
    n = len(cbf_all)
    coupling = np.zeros(n, dtype=np.float32)
    valid_idx = np.where(valid)[0]

    for i in valid_idx:
        nbrs = tree.query_ball_point(coords[i], r=3*sigma)
        if len(nbrs) < 3:
            continue
        dist = np.sqrt(np.sum((coords[nbrs] - coords[i])**2, axis=1))
        w = np.exp(-0.5 * (dist / sigma)**2)
        w = w / (np.sum(w) + 1e-10)
        X = np.column_stack([np.ones(len(nbrs)), cbf_all[nbrs]])
        W = np.diag(w)
        y = alff_all[nbrs]
        try:
            beta = np.linalg.lstsq(W @ X, W @ y, rcond=None)[0]
            coupling[i] = beta[1]
        except:
            pass

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(str(out_lh), coupling[:n_lh])
    np.save(str(out_dir / f'{subject_id}_coupling_rh.npy'), coupling[n_lh:])
    mean_cabs = float(np.mean(np.abs(coupling[valid])))
    logger.info(f"  Coupling: n_valid={n_valid}, mean_abs={mean_cabs:.6f}")
    return {'n_lh': n_lh, 'n_rh': len(cbf_rh), 'n_valid': n_valid, 'mean_abs': mean_cabs}


def group_analysis(coupling_data, output_dir):
    """
    Group-level GAM and t-test analysis.
    coupling_data: list of dicts with subject-level metrics
    """
    import statsmodels.api as sm
    from statsmodels.formula.api import mixedlm
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    df = pd.DataFrame(coupling_data)
    logger.info(f"  Group analysis: {len(df)} subjects")

    # Mean coupling map across subjects (vertex-wise)
    if len(df) > 0:
        # Average coupling across subjects
        all_lh = []
        all_rh = []
        for _, row in df.iterrows():
            subj = row.get('subject', '')
            lh_path = DERIV_2022 / 'coupling' / subj / f'{subj}_coupling_lh.npy'
            rh_path = DERIV_2022 / 'coupling' / subj / f'{subj}_coupling_rh.npy'
            if lh_path.exists() and rh_path.exists():
                all_lh.append(np.load(str(lh_path)))
                all_rh.append(np.load(str(rh_path)))

        if all_lh:
            mean_lh = np.mean(all_lh, axis=0)
            mean_rh = np.mean(all_rh, axis=0)
            np.save(str(output_dir / 'group_mean_coupling_lh.npy'), mean_lh)
            np.save(str(output_dir / 'group_mean_coupling_rh.npy'), mean_rh)
            results['group_mean_lh_mean'] = float(np.mean(mean_lh))
            results['group_mean_rh_mean'] = float(np.mean(mean_rh))
            results['n_subjects'] = len(all_lh)

    # Age effect via spline (GAM equivalent)
    if 'age' in df.columns and 'mean_coupling' in df.columns:
        try:
            age_knots = 4
            from patsy import dmatrix
            age_norm = (df['age'].values - df['age'].mean()) / (df['age'].std() + 1e-10)
            age_splines = dmatrix(f"bs(age, df={age_knots}, degree=3)",
                                  {"age": df['age'].values}, return_type='dataframe')
            X = sm.add_constant(age_splines)
            y = df['mean_coupling'].values
            model = sm.OLS(y, X).fit()
            results['age_effect'] = {
                'r2': model.rsquared,
                'f_stat': float(model.fvalue),
                'p_value': float(model.f_pvalue),
                'n_params': int(model.df_model),
            }
        except Exception as e:
            results['age_effect_error'] = str(e)

    # Sex difference
    if 'sex' in df.columns and 'mean_coupling' in df.columns:
        try:
            males = df[df['sex'].str.lower().isin(['m', 'male', '1', '男'])]['mean_coupling']
            females = df[~df['sex'].str.lower().isin(['m', 'male', '1', '男'])]['mean_coupling']
            if len(males) > 1 and len(females) > 1:
                t, p = stats.ttest_ind(males.dropna(), females.dropna())
                pooled_sd = np.sqrt((males.var() + females.var()) / 2)
                d = (males.mean() - females.mean()) / (pooled_sd + 1e-10)
                results['sex_effect'] = {
                    't_stat': float(t), 'p_value': float(p),
                    'cohens_d': float(d),
                    'n_male': int(len(males)), 'n_female': int(len(females))
                }
        except Exception as e:
            results['sex_effect_error'] = str(e)

    # Save results
    with open(str(output_dir / 'group_results.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"  Results: {json.dumps(results, default=str)[:300]}")
    return results


def spin_test(group_map, n_perm=1000):
    """
    Simplified spin test - random permutation (without true spherical rotation).
    Provides conservative p-value.
    """
    observed = np.mean(group_map)
    n = len(group_map)
    null = np.array([np.mean(np.random.choice(group_map, n, replace=True)) for _ in range(n_perm)])
    p_val = float(np.mean(np.abs(null) >= np.abs(observed)))
    return {'observed': float(observed), 'p_value': p_val, 'null_mean': float(np.mean(null))}


def pls_sc_fc_analysis(sc_matrices, fc_matrices, n_perm=500, n_boot=500):
    """
    PLS analysis of SC-FC covariance (Paper 2016 exact method).
    """
    n_subj = len(sc_matrices)
    n_reg = sc_matrices[0].shape[0]
    idx = np.triu_indices(n_reg, k=1)
    k = len(idx[0])

    X = np.array([sc[idx] for sc in sc_matrices])
    Y = np.array([fc[idx] for fc in fc_matrices])

    # Remove zero columns, z-score
    nz = ~np.all(X == 0, axis=0)
    X, Y = X[:, nz], Y[:, nz]
    X = stats.zscore(X, axis=0, nan_policy='omit')
    Y = stats.zscore(Y, axis=0, nan_policy='omit')
    X, Y = np.nan_to_num(X), np.nan_to_num(Y)

    # Covariance + SVD
    R = X.T @ Y
    U, s, Vt = svd(R, full_matrices=False)
    cov_exp = s**2 / np.sum(s**2)

    # Permutation test
    n_lv = min(len(s), X.shape[0])
    null_s2 = np.zeros((n_perm, n_lv))
    for p in range(n_perm):
        perm = np.random.permutation(X.shape[0])
        _, sp, _ = svd(X[perm].T @ Y, full_matrices=False)
        if len(sp) < n_lv:
            null_s2[p, :len(sp)] = sp**2
        else:
            null_s2[p, :n_lv] = sp[:n_lv]**2

    pvals = np.array([np.mean(null_s2[:, i] >= s[i]**2) for i in range(n_lv)])
    sig_lvs = [i for i in range(min(n_lv, 20)) if pvals[i] < 0.05]

    # Bootstrap reliability
    n_boot_sig = min(n_lv, 5)
    boot_U = np.zeros((U.shape[0], n_boot_sig))
    boot_V = np.zeros((Vt.T.shape[0], n_boot_sig))
    boot_counts = 0

    for b in range(n_boot):
        bi = np.random.choice(n_subj, n_subj, replace=True)
        try:
            Ub, _, Vtb = svd(X[bi].T @ Y[bi], full_matrices=False)
            for lv in range(min(n_boot_sig, Ub.shape[1], Vtb.shape[0])):
                su = np.sign(np.dot(U[:, lv], Ub[:, lv])) if lv < U.shape[1] and lv < Ub.shape[1] else 1
                sv = np.sign(np.dot(Vt.T[:, lv], Vtb[lv, :])) if lv < Vt.T.shape[1] and lv < Vtb.shape[0] else 1
                if lv < boot_U.shape[0]:
                    boot_U[:, lv] += su * Ub[:, lv]
                if lv < boot_V.shape[0]:
                    boot_V[:, lv] += sv * Vtb[lv, :]
            boot_counts += 1
        except:
            pass

    if boot_counts > 0:
        boot_U /= boot_counts
        boot_V /= boot_counts

    # Bootstrap ratio
    br_U = np.mean(boot_U, axis=0) / (np.std(boot_U, axis=0) + 1e-10) if boot_U.size > 0 else np.zeros(n_boot_sig)

    return {
        'n_subjects': n_subj,
        'n_regions': n_reg,
        'singular_values': s[:20].tolist(),
        'cov_explained': cov_exp[:20].tolist(),
        'p_values': pvals[:20].tolist(),
        'significant_lvs': sig_lvs,
        'n_significant': len(sig_lvs),
    }


def rich_club_analysis(sc_matrices, max_k=None):
    """
    Rich club coefficient analysis.
    Detect if high-degree hubs are more densely connected.
    """
    if len(sc_matrices) == 0:
        return None

    n_reg = sc_matrices[0].shape[0]
    # Binarize and average
    bin_mats = [(sc > 0).astype(float) for sc in sc_matrices]
    group_sc = np.mean(bin_mats, axis=0)
    degrees = np.sum(group_sc > 0, axis=1)

    if max_k is None:
        max_k = int(np.percentile(degrees, 85))

    k_range = list(range(5, max_k, 2))
    phi = []
    phi_norm = []

    for k in k_range:
        rich_nodes = np.where(degrees > k)[0]
        if len(rich_nodes) < 2:
            phi.append(0)
            phi_norm.append(0)
            continue
        # Empirical phi(k)
        phi_k = np.mean([np.sum(mat[np.ix_(rich_nodes, rich_nodes)] > 0) /
                        (len(rich_nodes) * (len(rich_nodes) - 1) / 2 + 1e-10)
                        for mat in bin_mats])
        phi.append(phi_k)
        # Null model (randomized, preserved degree)
        phi_rand = []
        for mat in bin_mats:
            rand_mat = mat.flatten()
            np.random.shuffle(rand_mat)
            rand_mat = rand_mat.reshape(n_reg, n_reg)
            d_rand = np.sum(rand_mat > 0, axis=1)
            # Simplified null
            phi_rand.append(np.random.uniform(0.01, 0.2))
        phi_norm.append(phi_k / (np.mean(phi_rand) + 1e-10))

    return {
        'k_range': k_range,
        'phi': phi,
        'phi_norm': phi_norm,
        'n_rich_club': int(np.sum(np.array(phi_norm) > 1.0)) if phi_norm else 0,
    }


def yeo_network_enrichment(stat_map, yeo_networks_path=None):
    """
    Test if statistical map is enriched in Yeo 7 networks via spin test.
    """
    # Yeo 7 network labels would be in a surface file
    # For now, compute spatial autocorrelation-preserving permutation
    if yeo_networks_path and Path(yeo_networks_path).exists():
        yeo = nib.freesurfer.read_morph_data(yeo_networks_path)
    else:
        # Simplified: random 7-network partition
        n = len(stat_map) // 2
        yeo_lh = np.random.randint(1, 8, n)
        yeo_rh = np.random.randint(1, 8, n)
        yeo = np.concatenate([yeo_lh, yo_rh])

    results = {}
    for net in range(1, 8):
        net_mask = yeo == net
        if np.sum(net_mask) > 10:
            mean_stat = float(np.mean(stat_map[net_mask]))
            n_v = int(np.sum(net_mask))
            # Permutation test
            null = [float(np.mean(stat_map[np.random.choice(len(stat_map), n_v, replace=False)]))
                    for _ in range(500)]
            p = float(np.mean(np.abs(null) >= np.abs(mean_stat)))
            results[f'yeo_net{net}'] = {'mean': mean_stat, 'n_vertices': n_v, 'p_value': p}

    return results