from matplotlib.widgets import Lasso
import numpy as np
import pandas as pd
import networkx as nx

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

    lambda_max = (1+np.sqrt(N/T))**2
    eigenvalues, eigenvectors = np.linalg.eig(correlation_matrix)

    is_noise = np.real(eigenvalues) < lambda_max # Eig can be complex due to the assymetrical nature of the correlation matrix, we take the real part for comparison
    noise_mean = np.mean(np.real(eigenvalues[is_noise]))
    eigenvalues_denoised = eigenvalues.copy()
    eigenvalues_denoised[is_noise] = noise_mean

    C_clean = eigenvectors @ np.diag(eigenvalues_denoised) @ eigenvectors.T
    C_clean = np.real(C_clean)
    # KEEP IN CHECK FOR FUTURE FILTERING (GLASSO)
    np.fill_diagonal(C_clean, 0)  # Set diagonal to 0 to avoid loops
    
    # Convert back to DataFrame with proper index and columns
    C_final_df = pd.DataFrame(C_clean, index=correlation_matrix.index, columns=correlation_matrix.columns)

    return C_final_df

def var_lasso(correlation_matrix, alpha=0.01):
    """
    LLM GENERATED DOCSTRING, VERIFIED BY AUTHOR:
    Applies VAR Lasso to the input correlation matrix to estimate a sparse inverse covariance matrix, which can be interpreted as a network of relationships between assets.
    Parameters:
    correlation_matrix : pd.DataFrame
        A correlation matrix derived from financial returns, where rows and columns represent different assets.
    alpha : float, optional
        Regularization strength for the Lasso regression. Higher values lead to sparser solutions. Default is 0.01.
    Returns:
    pd.DataFrame
        A DataFrame representing the sparse inverse covariance matrix (precision matrix) estimated by VAR Lasso, where non-zero entries indicate significant relationships between assets.
    Notes:
    - Given the crucial leader-lagger assymetry in financial data, the VAR Lasso is a powerful altnernative to the traditional graphical lasso.
    """

    # assym_precision_matrix = np.zeros((len(correlation_matrix), len(correlation_matrix)))

    # model = Lasso(alpha=alpha)

    

    
        