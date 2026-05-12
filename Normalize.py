import numpy as np
import pandas as pd

def cpm_log1p(X_df: pd.DataFrame) -> pd.DataFrame:
    """Convert raw counts to log1p(CPM). Assumes X_df is samples x genes."""
    counts = X_df.values.astype(np.float64)
    lib_size = counts.sum(axis=1, keepdims=True)
    lib_size[lib_size == 0] = 1.0
    cpm = (counts / lib_size) * 1e6
    return pd.DataFrame(np.log1p(cpm), index=X_df.index, columns=X_df.columns)

def filter_genes(X_df: pd.DataFrame, min_samples_frac: float = 0.10, min_cpm: float = 1.0) -> pd.DataFrame:
    """
    Keep genes that have CPM >= min_cpm in at least min_samples_frac of samples.
    Works on raw counts; internally uses CPM thresholding.
    """
    counts = X_df.values.astype(np.float64)
    lib_size = counts.sum(axis=1, keepdims=True)
    lib_size[lib_size == 0] = 1.0
    cpm = (counts / lib_size) * 1e6

    min_samples = int(np.ceil(min_samples_frac * X_df.shape[0]))
    keep = (cpm >= min_cpm).sum(axis=0) >= min_samples

    kept_genes = X_df.columns[keep]
    return X_df.loc[:, kept_genes]

# Filter first (on raw counts)
X_f = filter_genes(X, min_samples_frac=0.10, min_cpm=1.0)
print("Genes after filtering:", X_f.shape[1])

# Normalize
X_n = cpm_log1p(X_f)
print("Normalized matrix shape:", X_n.shape)
