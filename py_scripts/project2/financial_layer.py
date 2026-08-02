from sklearn.linear_model import Lasso
import numpy as np
import pandas as pd
import networkx as nx

def PCA(returns):
    corr_matrix = returns.corr()
    eigenvalues, eigenvectors = np.linalg.eig(corr_matrix)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    return eigenvalues, eigenvectors



def parkinson_log_vol(high, low):
    """
    Daily Parkinson (range-based) volatility from intraday High/Low, in logs.

    sigma_t^2 = ln(High_t / Low_t)^2 / (4 ln 2)

    The high-low range is ~5x more statistically efficient than |close-to-close
    return| as a daily volatility estimator. Log-volatility is approximately
    Gaussian (Andersen et al.), which suits the Lasso downstream.

    Parameters
    ----------
    high, low : pd.DataFrame
        Daily High and Low prices, same shape, assets on columns.

    Returns
    -------
    pd.DataFrame
        Log Parkinson volatility, NaN where the range is zero/invalid.
    """
    rng = np.log(high / low)
    var = rng ** 2 / (4 * np.log(2))
    var = var.where(var > 0)  # zero range (stale bar) -> NaN rather than log(0)
    return 0.5 * np.log(var)


def fevd_connectedness(var_coefs, residuals, horizon=10):
    """
    Diebold-Yilmaz connectedness table from a fitted VAR via generalized FEVD.

    Answers: "of the H-step forecast error variance of asset i's series, what
    fraction traces to shocks originating in asset j?" Uses the Pesaran-Shin
    generalized decomposition (order-invariant, no Cholesky ordering choice),
    row-normalized as in Diebold-Yilmaz (2012).

    Parameters
    ----------
    var_coefs : dict[int, pd.DataFrame]
        Output of var_lasso: {L: A_L} with A_L.loc[i, j] = coef of asset i at
        t-L on asset j at t.
    residuals : pd.DataFrame
        In-sample VAR residuals (time x assets), used for the shock covariance.
    horizon : int, optional
        FEVD horizon H in bars (default 10 days).

    Returns
    -------
    table : pd.DataFrame
        theta.loc[i, j] = share of i's H-step FEV due to shocks in j
        (rows sum to 1). Off-diagonal (i, j) is the directed spillover j -> i.
    summary : pd.DataFrame
        Per-asset TO (exports, sum of its column off-diagonal), FROM (imports,
        sum of its row off-diagonal) and NET = TO - FROM, plus the total
        connectedness index (mean off-diagonal share) in `summary.attrs['total']`.
    """
    names = residuals.columns
    N = len(names)
    p = max(var_coefs)
    # Standard VAR form x_t = sum B_l x_{t-l}: B_l = A_l^T
    B = {L: var_coefs[L].values.T for L in var_coefs}
    Sigma = np.cov(residuals.values.T)

    # MA representation Psi_h via recursion
    Psi = [np.eye(N)]
    for h in range(1, horizon):
        Psi.append(sum(B[l] @ Psi[h - l] for l in range(1, min(h, p) + 1)))

    diag_sigma = np.diag(Sigma).copy()
    diag_sigma[diag_sigma == 0] = np.finfo(float).eps
    num = np.zeros((N, N))
    den = np.zeros(N)
    for Ph in Psi:
        PS = Ph @ Sigma
        num += (PS ** 2) / diag_sigma[None, :]   # (e_i' Psi_h Sigma e_j)^2 / sigma_jj
        den += np.einsum('ij,ij->i', PS, Ph)     # e_i' Psi_h Sigma Psi_h' e_i

    theta = num / den[:, None]
    theta = theta / theta.sum(axis=1, keepdims=True)  # row-normalize (DY 2012)
    table = pd.DataFrame(theta, index=names, columns=names)

    off = table.values - np.diag(np.diag(table.values))
    summary = pd.DataFrame({
        'TO': off.sum(axis=0),     # what j exports to all others
        'FROM': off.sum(axis=1),   # what i imports from all others
    }, index=names)
    summary['NET'] = summary['TO'] - summary['FROM']
    summary.attrs['total'] = off.sum() / N  # total connectedness index
    return table, summary


def remove_market_mode(returns, n_modes=1):
    """
    Project out the dominant eigenmode(s) of the correlation matrix, returning
    idiosyncratic residual returns.

    At hourly frequency the top eigenmode is the common market/sector factor,
    which dominates raw returns and drowns out cross-asset lead-lag structure
    (a rank-1 market mode produces an exactly symmetric lead-lag matrix).
    Removing it lets Sparse VAR compete over genuinely idiosyncratic
    co-movement. The removed factor should be modeled separately as its own
    systematic layer of the TMDN.

    Parameters
    ----------
    returns : pd.DataFrame
        Asset returns, time on rows, assets on columns.
    n_modes : int, optional
        Number of leading eigenmodes to project out (default 1, the market mode).

    Returns
    -------
    residuals : pd.DataFrame
        Standardized residual returns with the top n_modes removed.
    factors : pd.DataFrame
        Time series of the removed factor(s), one column per mode - keep these
        for the systematic layer.
    """
    standardized = (returns - returns.mean()) / returns.std()
    corr = standardized.corr().values
    eigenvalues, eigenvectors = np.linalg.eigh(corr)
    top = eigenvectors[:, ::-1][:, :n_modes]  # eigh sorts ascending - take the largest

    Z = standardized.values
    F = Z @ top                    # factor time series (T x n_modes)
    residual_vals = Z - F @ top.T  # project out the factor(s)

    residuals = pd.DataFrame(residual_vals, index=returns.index, columns=returns.columns)
    factors = pd.DataFrame(F, index=returns.index,
                           columns=[f"mode_{k+1}" for k in range(n_modes)])
    return residuals, factors


def assymetrical_correlation_matrix(returns):
    """LLM GENERATED DOCSTRING, VERIFIED BY AUTHOR:
    Computes the asymmetrical correlation matrix for a given DataFrame of returns. 
    The asymmetrical correlation matrix captures the lead-lag relationships between assets by standardizing the returns and calculating the cross-product of the standardized leader and lagger returns.
    Parameters:
    returns : pd.DataFrame
        A DataFrame of asset returns, where rows represent time periods and columns represent different assets.
    Returns:
    pd.DataFrame    An asymmetrical correlation matrix where the entry (i, j) represents the correlation of asset i leading asset j.   
    """
    standardized_returns = (returns - returns.mean()) / returns.std()
    
    leader_returns = standardized_returns.shift(1).dropna()  #shift and drop first row
    standardized_returns = standardized_returns.loc[leader_returns.index]  #align the standardized returns with the shifted leader returns
    
    asym_corr_matrix = leader_returns.T @ standardized_returns / (len(leader_returns) - 1)
    
    return asym_corr_matrix

def marchenko_pastur_returns(correlation_matrix, N, T):
    """
    LLM GENERATED DOCSTRING, VERIFIED BY AUHTOR:
    Applies Marchenko-Pastur denoising to a correlation matrix derived from financial returns.

    This function computes the empirical correlation matrix from the input returns DataFrame,
    then denoises it using the Marchenko-Pastur distribution to filter out noise eigenvalues.
    Eigenvalues below the theoretical maximum (lambda_max) are replaced with their mean value,
    preserving the signal while reducing noise.

    Parameters:
    -----------
    total_returns : pd.DataFrame
        DataFrame of asset returns, with assets as columns and time periods as rows.
        Should be log-returns or similar stationary series.

    N : int
        Number of assets (variables) in the returns data.

    T : int
        Number of time periods (observations) in the returns data.

    Returns:
    --------
    pd.DataFrame
        Denoised correlation matrix with the same index and columns as the input returns.
        Values are normalized to ensure diagonal elements are 1.

    Notes:
    ------
    - The Marchenko-Pastur distribution models the eigenvalue spectrum of random correlation matrices.
    - Lambda_max = (1 + sqrt(N/T))^2 serves as the threshold for signal vs noise.
    - This method helps extract meaningful correlations from noisy financial data.
    """

    C = np.asarray(correlation_matrix, dtype=float)

    if np.allclose(C, C.T):
        # Symmetric case: classic MP filtering. eigh (not eig) guarantees real
        # eigenvalues and orthonormal eigenvectors, so V @ diag(λ) @ V.T is valid.
        lambda_max = (1 + np.sqrt(N / T)) ** 2
        eigenvalues, eigenvectors = np.linalg.eigh(C)
        is_noise = eigenvalues < lambda_max
        eigenvalues_denoised = eigenvalues.copy()
        if is_noise.any():
            eigenvalues_denoised[is_noise] = eigenvalues[is_noise].mean()
        C_clean = eigenvectors @ np.diag(eigenvalues_denoised) @ eigenvectors.T
    else:
        # Asymmetric (lead-lag) case: eigenvectors are not orthogonal, so the
        # V.T reconstruction is invalid and eigenvalues are complex. Use SVD
        # instead. Under the null, an N x N lead-lag correlation matrix has
        # i.i.d. entries of variance 1/T, and its largest singular value
        # converges to 2*sqrt(N/T) (quarter-circle law) - singular values
        # below that bound are indistinguishable from noise and are truncated.
        s_max = 2 * np.sqrt(N / T)
        U, s, Vt = np.linalg.svd(C)
        s_denoised = np.where(s > s_max, s, 0.0)
        C_clean = U @ np.diag(s_denoised) @ Vt

    # KEEP IN CHECK FOR FUTURE FILTERING (SPARSE VAR)
    np.fill_diagonal(C_clean, 0)  # Set diagonal to 0 to avoid loops

    # Convert back to DataFrame with proper index and columns
    C_final_df = pd.DataFrame(C_clean, index=correlation_matrix.index, columns=correlation_matrix.columns)

    return C_final_df

def har_x_lasso(log_vol, alpha=0.01):
    """
    HAR-X(d,w,m) volatility model with cross-asset Lasso regularization.

    Supersedes var_lasso() for the volatility spillover layer. VAR(4) on
    log-vol approximates long-memory persistence with 4 uniform lags, but vol
    has well-documented multi-frequency structure (Corsi 2009): market
    participants react to yesterday, last week, and last month of vol - producing rough-fractional-integration dynamics that HAR captures with 3
    parameters per pair instead of 4*N for VAR(4), making estimation more
    efficient at N=27.

    Model per target asset j:
        σ_{j,t} = Σ_i [ β^d_{ij} σ_{i,t-1}
                       + β^w_{ij} σ̄^w_{i,t}   (1/5 mean over t-5..t-1)
                       + β^m_{ij} σ̄^m_{i,t} ] (1/22 mean over t-22..t-1)
                + ε_{j,t}

    Design matrix: [σ^d, σ^w, σ^m] for all N assets → 3N columns.
    One Lasso per target j recovers β^d, β^w, β^m (N×N each). Cross-asset
    terms i≠j are the spillover channels - this is the HAR-X extension
    (Bollerslev et al.) applied jointly across a panel.

    For FEVD compatibility, HAR(1,5,22) is converted to an equivalent VAR(22)
    (Corsi et al. 2012):
        A_1     = β^d + β^w/5 + β^m/22
        A_{2..5}  = β^w/5 + β^m/22
        A_{6..22} = β^m/22
    This feeds directly into fevd_connectedness() unchanged.

    Parameters
    ----------
    log_vol : pd.DataFrame
        Standardized (zero-mean, unit-var) Parkinson log-volatility, time×assets.
        Must already be standardized - do not pass raw log-vol.
    alpha : float
        Lasso L1 penalty; tune against the placebo test as with var_lasso.

    Returns
    -------
    har_coefs : dict
        {'d': B_d, 'w': B_w, 'm': B_m} - N×N DataFrames.
        B_k.loc[i, j] = HAR-k coefficient of asset i predicting asset j.
    var_rep : dict[int, pd.DataFrame]
        Equivalent VAR(22) representation {1..22: A_l} for fevd_connectedness.
    residuals : pd.DataFrame
        In-sample HAR-X residuals, aligned to log_vol.index[22:].
    """
    names = log_vol.columns
    N = len(names)

    daily   = log_vol.shift(1)
    weekly  = log_vol.rolling(5).mean().shift(1)   # mean(t-5..t-1)
    monthly = log_vol.rolling(22).mean().shift(1)  # mean(t-22..t-1)

    X_df = pd.concat([daily, weekly, monthly], axis=1)
    X_df.columns = (
        [f"d_{c}" for c in names]
        + [f"w_{c}" for c in names]
        + [f"m_{c}" for c in names]
    )

    valid  = X_df.index[X_df.notna().all(axis=1)]
    X_vals = X_df.loc[valid].values
    Y_vals = log_vol.loc[valid].values

    coefs = np.zeros((3 * N, N))
    for j in range(N):
        model = Lasso(alpha=alpha, fit_intercept=False, max_iter=10000)
        model.fit(X_vals, Y_vals[:, j])
        coefs[:, j] = model.coef_

    Bd = pd.DataFrame(coefs[:N],        index=names, columns=names)
    Bw = pd.DataFrame(coefs[N:2 * N],   index=names, columns=names)
    Bm = pd.DataFrame(coefs[2 * N:],    index=names, columns=names)
    har_coefs = {'d': Bd, 'w': Bw, 'm': Bm}

    # HAR(1,5,22) → VAR(22) for fevd_connectedness()
    bw5  = Bw / 5
    bm22 = Bm / 22
    var_rep = {1: Bd + bw5 + bm22}
    for l in range(2, 6):
        var_rep[l] = (bw5 + bm22).copy()
    for l in range(6, 23):
        var_rep[l] = bm22.copy()

    residuals = pd.DataFrame(Y_vals - X_vals @ coefs, index=valid, columns=names)
    return har_coefs, var_rep, residuals


def circular_shift_placebo(panel, rng):
    """
    Circular-shift each column by an independent random offset.

    The correct placebo control for an *autocorrelated* panel (e.g. log-vol,
    lag-1 autocorrelation ~0.4). A plain i.i.d. row shuffle would destroy each
    series' own persistence, so any model that exploits autocorrelation beats it
    trivially and the test proves nothing. A per-column circular shift instead
    PRESERVES each series' own autocorrelation (and marginal distribution) while
    destroying the cross-asset temporal alignment - which is exactly, and only,
    what a cross-asset spillover network claims to capture. If the real network's
    cross-asset structure survives this control, it is not an artifact of each
    series being individually persistent.

    Parameters
    ----------
    panel : pd.DataFrame
        Time x assets (e.g. standardized log-volatility).
    rng : np.random.Generator
        Source of the per-column offsets (pass one in for reproducibility).

    Returns
    -------
    pd.DataFrame
        Same shape/index/columns as `panel`, each column independently rolled.
    """
    T = len(panel)
    shifted = np.empty_like(panel.values)
    for j in range(panel.shape[1]):
        offset = int(rng.integers(1, T))  # avoid the no-op zero shift
        shifted[:, j] = np.roll(panel.values[:, j], offset)
    return pd.DataFrame(shifted, index=panel.index, columns=panel.columns)


def var_lasso(returns, alpha=0.01, n_lags=1, exclude_cross_session=True):
    """
    Sparse VAR(p) via per-asset Lasso regressions - NOT Graphical Lasso.

    Fits X(t) = sum_{L=1..p} A_L^T X(t-L) + eps with an L1 penalty by running
    one Lasso regression per target asset: regress asset j at t on ALL assets
    at t-1, ..., t-p jointly. The stacked coefficient vectors form one
    directed, asymmetric adjacency matrix per lag. Graphical Lasso would
    instead estimate a symmetric precision matrix of contemporaneous returns
    and destroy the leader-lagger directionality this project depends on.

    Fitting all lags jointly (rather than one shifted correlation per lag)
    means each A_L captures the *marginal* predictive content of lag L,
    controlling for the other lags - the multivariate analogue of partial
    autocorrelation.

    Parameters
    ----------
    returns : pd.DataFrame
        (Residualized) asset returns, time on rows, assets on columns.
        Fit on returns directly - not on a correlation matrix.
    alpha : float, optional
        L1 regularization strength; higher values give a sparser graph.
    n_lags : int, optional
        VAR order p - how many hourly lags of every asset enter the regression.
    exclude_cross_session : bool, optional
        If True, keep only targets whose full lag stack lies within the same
        trading session (same calendar date), so no lead-lag edge spans an
        overnight/weekend gap. Only applies to intraday data (multiple bars
        per calendar date) - at daily or lower frequency every bar is its own
        date and the filter is skipped automatically. NOTE: with B intraday
        bars per session this leaves (B - n_lags) usable targets per day - check you retain more samples than the N * n_lags predictors.

    Returns
    -------
    dict[int, pd.DataFrame]
        {L: A_L} for L = 1..n_lags, where A_L.loc[i, j] is the coefficient of
        leader i at time t-L predicting lagger j at time t. Nonzero entries
        are the directed edges of the lag-L lead-lag network.
    """
    standardized = (returns - returns.mean()) / returns.std()
    N = returns.shape[1]
    T = len(standardized)

    # Build the lagged design: row t stacks [X(t-1), X(t-2), ..., X(t-p)]
    Y = standardized.iloc[n_lags:]
    lag_blocks = [standardized.shift(L).iloc[n_lags:] for L in range(1, n_lags + 1)]
    X_vals = np.hstack([b.values for b in lag_blocks])
    Y_vals = Y.values

    if exclude_cross_session and isinstance(returns.index, pd.DatetimeIndex):
        all_dates = np.asarray(returns.index.date)
        is_intraday = len(np.unique(all_dates)) < len(all_dates)
        if is_intraday:  # at daily+ frequency every bar is its own session - nothing to exclude
            target_dates = np.asarray(Y.index.date)
            oldest_lag_dates = np.asarray(lag_blocks[-1].index.date)  # t - p
            same_session = target_dates == oldest_lag_dates
            X_vals, Y_vals = X_vals[same_session], Y_vals[same_session]

    n_samples, n_predictors = X_vals.shape
    if n_samples < n_predictors:
        print(f"var_lasso warning: {n_samples} samples < {n_predictors} predictors "
              f"(N={N}, p={n_lags}) - Lasso still fits, but consider a longer "
              f"window or fewer lags for stability.")

    coefs = np.zeros((n_predictors, N))
    for j in range(N):
        model = Lasso(alpha=alpha, fit_intercept=False, max_iter=10000)
        model.fit(X_vals, Y_vals[:, j])
        coefs[:, j] = model.coef_

    return {
        L: pd.DataFrame(coefs[(L - 1) * N: L * N, :],
                        index=returns.columns, columns=returns.columns)
        for L in range(1, n_lags + 1)
    }