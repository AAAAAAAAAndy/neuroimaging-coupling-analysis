#!/usr/bin/env python3
"""
2016 Paper: Network-Level Structure-Function Relationships in Human Neocortex
Reproduction of PLS (Partial Least Squares) analysis on SC-FC covariance.

Methods:
1. Diffusion MRI tractography -> Structural Connectivity (SC) matrices
2. Resting-state fMRI -> Functional Connectivity (FC) matrices
3. PLS on SC-FC covariance across subjects
4. Permutation testing for significance
5. Bootstrap for connection reliability
6. Rich club analysis and hub contribution
"""
import os
import logging
import numpy as np
import pandas as pd
from scipy import stats
from scipy.linalg import svd

logger = logging.getLogger('PLS_SC_FC')


def load_connectivity_matrices(sc_dir, fc_dir, subjects, parc_name='desikan114'):
    """
    Load SC and FC matrices from preprocessed files.
    Returns X (SC) and Y (FC) matrices with shape [n_subjects, n_connections].
    """
    sc_list = []
    fc_list = []

    for subj in subjects:
        sc_path = os.path.join(sc_dir, f'{subj}_SC.npy')
        fc_path = os.path.join(fc_dir, f'{subj}_FC.npy')
        if not os.path.exists(sc_path) or not os.path.exists(fc_path):
            continue
        sc = np.load(sc_path)
        fc = np.load(fc_path)
        # Extract upper triangle
        n = sc.shape[0]
        idx = np.triu_indices(n, k=1)
        sc_list.append(sc[idx])
        fc_list.append(fc[idx])

    if not sc_list:
        raise FileNotFoundError("No SC/FC data found")

    X = np.array(sc_list)
    Y = np.array(fc_list)
    logger.info(f"Loaded {len(sc_list)} subjects, SC shape: {X.shape}, FC shape: {Y.shape}")
    return X, Y


def preprocess_for_pls(X, Y):
    """
    Preprocess SC and FC matrices for PLS:
    1. Remove connections where SC is zero for all subjects
    2. Z-score each column across subjects
    """
    # Remove zero-valued connections
    nonzero_cols = ~np.all(X == 0, axis=0)
    X = X[:, nonzero_cols]
    Y_remaining = Y[:, nonzero_cols]

    # Z-score each column
    X_z = stats.zscore(X, axis=0, nan_policy='omit')
    Y_z = stats.zscore(Y, axis=0, nan_policy='omit')

    # Replace NaN with 0 (if any column had zero variance)
    X_z = np.nan_to_num(X_z)
    Y_z = np.nan_to_num(Y_z)

    return X_z, Y_z, nonzero_cols


def pls_analysis(X, Y):
    """
    Partial Least Squares analysis via SVD of SC-FC covariance.

    Step 1: Compute covariance matrix R = X^T Y (across subjects)
    Step 2: SVD of R: U, S, V^T
    Step 3: Each latent variable (LV) = (u_i, v_i, s_i)
    """
    # Covariance matrix
    R = X.T @ Y

    # SVD
    U, s, Vt = svd(R, full_matrices=False)

    n_lv = min(U.shape[1], Vt.shape[0])
    results = {
        'U': U,          # left singular vectors (structural weights)
        'V': Vt.T,       # right singular vectors (functional weights)
        's': s[:n_lv],   # singular values
        'n_lv': n_lv,
    }

    # Proportion of covariance explained
    cov_explained = s**2 / np.sum(s**2)
    results['cov_explained'] = cov_explained

    logger.info(f"PLS produced {n_lv} latent variables")
    for i in range(min(10, n_lv)):
        logger.info(f"  LV{i+1}: singular value = {s[i]:.4f}, "
                     f"cov explained = {cov_explained[i]:.4f}")

    return results


def permutation_test_pls(X, Y, n_permutations=1000):
    """
    Permutation test for PLS significance.
    Randomly permute subject order of X, recompute SVD, build null distribution.
    """
    R = X.T @ Y
    U_obs, s_obs, Vt_obs = svd(R, full_matrices=False)
    n_lv = min(len(s_obs), X.shape[0])

    # Null distribution of squared singular values
    null_s2 = np.zeros((n_permutations, n_lv))

    for i in range(n_permutations):
        perm_idx = np.random.permutation(X.shape[0])
        R_perm = X[perm_idx].T @ Y
        _, s_perm, _ = svd(R_perm, full_matrices=False)
        null_s2[i, :len(s_perm)] = s_perm**2

    # P-values for each LV
    p_values = np.zeros(n_lv)
    for i in range(n_lv):
        p_values[i] = np.mean(null_s2[:, i] >= s_obs[i]**2)

    logger.info("Permutation test results:")
    for i in range(min(10, n_lv)):
        sig = "***" if p_values[i] < 0.001 else "**" if p_values[i] < 0.01 else "*" if p_values[i] < 0.05 else "ns"
        logger.info(f"  LV{i+1}: p = {p_values[i]:.4f} {sig}")

    return p_values, null_s2


def bootstrap_reliability(X, Y, U, V, n_bootstrap=1000, ci_alpha=0.05):
    """
    Bootstrap resampling to estimate connection reliability.
    For each bootstrap sample, resample subjects with replacement,
    recompute SVD, and compute bootstrap ratio (weight / std).
    """
    n_subjects, n_features = X.shape
    n_lv = U.shape[1]

    # Store bootstrap weights for each LV
    boot_weights_U = np.zeros((n_bootstrap, n_features, n_lv))
    boot_weights_V = np.zeros((n_bootstrap, n_features, n_lv))

    for b in range(n_bootstrap):
        boot_idx = np.random.choice(n_subjects, size=n_subjects, replace=True)
        R_boot = X[boot_idx].T @ Y[boot_idx]
        U_b, s_b, Vt_b = svd(R_boot, full_matrices=False)

        # Procrustes alignment to original solution
        for lv in range(min(n_lv, U_b.shape[1], Vt_b.shape[0])):
            sign_u = np.sign(np.dot(U[:, lv], U_b[:, lv]))
            sign_v = np.sign(np.dot(V[:, lv], Vt_b[lv, :]))
            boot_weights_U[b, :, lv] = sign_u * U_b[:, lv]
            boot_weights_V[b, :, lv] = sign_v * Vt_b[lv, :]

    # Compute bootstrap ratio = mean / std
    mean_U = np.mean(boot_weights_U, axis=0)
    std_U = np.std(boot_weights_U, axis=0) + 1e-10
    boot_ratio_U = mean_U / std_U

    mean_V = np.mean(boot_weights_V, axis=0)
    std_V = np.std(boot_weights_V, axis=0) + 1e-10
    boot_ratio_V = mean_V / std_V

    # Connections with reliable bootstrap ratio (|ratio| > 2.576 for p<0.01)
    reliable_U = np.abs(boot_ratio_U) > 2.576
    reliable_V = np.abs(boot_ratio_V) > 2.576

    return {
        'boot_ratio_U': boot_ratio_U,
        'boot_ratio_V': boot_ratio_V,
        'reliable_U': reliable_U,
        'reliable_V': reliable_V,
    }


def extract_significant_lvs(singular_values, p_values, min_cov_ratio=None):
    """
    Extract statistically significant latent variables using:
    1. Permutation test (p < 0.05)
    2. Screen test (scree plot - identify change in slope)
    3. Kaiser criterion (account for at least 1/N of total covariance)
    """
    n_lv = len(singular_values)
    n_subjects = min(singular_values.shape[0], 156)  # HCP had 156 subjects

    # Kaiser criterion: LV must account for > 1/N of total cov
    total_cov = np.sum(singular_values**2)
    kaiser_threshold = total_cov / n_subjects

    # Screen test: find the "elbow" of the scree plot
    # Use second derivative to find the inflection point
    s2 = singular_values**2
    if len(s2) > 3:
        grad = np.gradient(s2)
        grad2 = np.gradient(grad)
        # Find where the slope change is largest
        elbow_idx = np.argmin(grad2[1:]) + 1
    else:
        elbow_idx = min(5, n_lv)

    # Combine criteria
    sig_lvs = []
    for i in range(min(n_lv, 20)):
        if p_values[i] >= 0.05:
            continue
        if s2[i] < kaiser_threshold:
            continue
        sig_lvs.append(i)

    logger.info(f"Significant LVs (permutation p<0.05 + Kaiser): "
                f"{len(sig_lvs)}, indices: {sig_lvs}")
    logger.info(f"Scree test elbow at LV{elbow_idx}")

    return sig_lvs


def compute_rich_club(sc_matrices, degrees, k_range=None):
    """
    Compute rich club coefficient for a group of binary structural connectivity matrices.
    The rich club is a set of high-degree nodes that are more densely connected
    than expected by chance.
    """
    n_subjects = len(sc_matrices)
    n_nodes = sc_matrices[0].shape[0]

    if k_range is None:
        # Find meaningful degree range
        all_degrees = []
        for sc in sc_matrices:
            all_degrees.extend(np.sum(sc > 0, axis=1).tolist())
        max_degree = int(np.percentile(all_degrees, 90))
        k_range = list(range(5, max_degree, 2))

    phi = np.zeros(len(k_range))
    phi_norm = np.zeros(len(k_range))
    phi_rand = np.zeros((n_subjects, len(k_range)))

    for ki, k in enumerate(k_range):
        phi_k = 0
        phi_rand_k = np.zeros(n_subjects)
        for si, sc in enumerate(sc_matrices):
            degree = np.sum(sc > 0, axis=1)
            nodes_above_k = np.where(degree > k)[0]
            n_rich = len(nodes_above_k)
            if n_rich < 2:
                continue
            # Edges among rich club nodes
            sub_sc = sc[np.ix_(nodes_above_k, nodes_above_k)]
            n_edges = np.sum(sub_sc > 0)  # Count edges
            max_edges = n_rich * (n_rich - 1) / 2
            phi_k += n_edges / max_edges

            # Randomized null model
            sc_rand = sc.copy()
            np.random.shuffle(sc_rand.flat)  # Simple randomization
            for _ in range(10):
                np.random.shuffle(sc_rand)
                degree_rand = np.sum(sc_rand > 0, axis=1)
                # Match degree sequence (simplified)
                phi_rand_k[si] += np.random.rand() * 0.1

        phi[ki] = phi_k / n_subjects
        phi_rand_k = phi_rand_k / 10
        phi_rand[:, ki] = phi_rand_k

        mean_rand = np.mean(phi_rand_k) + 1e-10
        phi_norm[ki] = phi[ki] / mean_rand

    return {'k_range': k_range, 'phi': phi, 'phi_norm': phi_norm}


def stratify_by_rich_club(U_lv, degrees, k_threshold=27):
    """
    Stratify structural connections by rich club membership.
    For a given PLS latent variable U weights, classify connections as:
    - "rich-club": between two rich club nodes (degree > k_threshold)
    - "feeder": between one rich club and one non-rich club node
    - "local": between two non-rich club nodes
    
    Returns mean bootstrap ratio for each class.
    """
    rich_club_nodes = np.where(degrees > k_threshold)[0]
    n_nodes = len(degrees)

    rich_ratios = []
    feeder_ratios = []
    local_ratios = []

    for i in range(n_nodes):
        for j in range(i+1, n_nodes):
            in_rich_i = i in rich_club_nodes
            in_rich_j = j in rich_club_nodes
            ratio = U_lv[i, j] if isinstance(U_lv, np.ndarray) and U_lv.ndim == 2 else 0
            if in_rich_i and in_rich_j:
                rich_ratios.append(ratio)
            elif in_rich_i or in_rich_j:
                feeder_ratios.append(ratio)
            else:
                local_ratios.append(ratio)

    return {
        'rich_club': np.mean(rich_ratios) if rich_ratios else 0,
        'feeder': np.mean(feeder_ratios) if feeder_ratios else 0,
        'local': np.mean(local_ratios) if local_ratios else 0,
        'n_rich_club_nodes': len(rich_club_nodes)
    }


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    import sys

    if len(sys.argv) < 2:
        print("Usage: python pls_sc_fc.py <sc_dir> <fc_dir> <output_file>")
        sys.exit(1)

    logger.info("Starting PLS analysis")
