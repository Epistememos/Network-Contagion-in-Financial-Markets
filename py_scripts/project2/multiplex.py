import numpy as np
import pandas as pd

# Assembles the two estimated TMDN layers into one supra-adjacency object
# A[i, j, alpha, t] and tests whether they are coupled - i.e. whether the
# data-driven financial layer (volatility connectedness) actually lines up with
# the industry-structure supply-chain layer. Until this module, the two layers
# were built and compared informally but never combined into a single object.
#
# Representation choice: a raw 4-D numpy array loses the ticker/layer/time
# labels that make the object interpretable, so we keep a labelled dict of
# layers instead:
#   - financial layer: time-indexed (one FEVD theta table per rolling window)
#   - supply-chain layers (direct, co-exposure): static, broadcast across t
# `supra_at(t)` returns the concrete {layer: NxN DataFrame} slice at one window.


def build_supra_adjacency(vol_tables, vol_ends, sc_adjacency, sc_overlap):
    """
    Assemble the multiplex supra-adjacency A[i, j, alpha, t].

    Parameters
    ----------
    vol_tables : list[pd.DataFrame]
        Financial layer: one Diebold-Yilmaz FEVD theta table per rolling window
        (theta.loc[i, j] = share of i's FEV from shocks in j).
    vol_ends : list[pd.Timestamp]
        Window-end timestamp for each entry of `vol_tables` (the t index).
    sc_adjacency : pd.DataFrame
        Supply-chain DIRECT layer: adjacency.loc[src, tgt] = channel count.
    sc_overlap : pd.DataFrame
        Supply-chain CO-EXPOSURE layer: symmetric shared-customer counts.

    Returns
    -------
    dict
        {
          'layers': ['financial', 'supply_direct', 'supply_coexposure'],
          'tickers': list[str],
          'financial': {'tables': [...], 'ends': DatetimeIndex},  # temporal
          'supply_direct': pd.DataFrame,      # static
          'supply_coexposure': pd.DataFrame,  # static
        }
    """
    if len(vol_tables) != len(vol_ends):
        raise ValueError("vol_tables and vol_ends must have equal length")
    if not vol_tables:
        raise ValueError("vol_tables is empty")

    tickers = list(vol_tables[-1].index)

    # Align the static layers to the financial layer's ticker order/label set.
    sc_direct = sc_adjacency.reindex(index=tickers, columns=tickers).fillna(0.0)
    sc_coexp = sc_overlap.reindex(index=tickers, columns=tickers).fillna(0.0)

    return {
        "layers": ["financial", "supply_direct", "supply_coexposure"],
        "tickers": tickers,
        "financial": {"tables": list(vol_tables),
                      "ends": pd.DatetimeIndex(vol_ends)},
        "supply_direct": sc_direct,
        "supply_coexposure": sc_coexp,
    }


def supra_at(supra, t_index=-1):
    """
    Concrete supra-adjacency slice A[:, :, alpha, t=t_index]: one NxN DataFrame
    per layer at a single window. Static layers are returned as-is (they do not
    vary in t).

    Parameters
    ----------
    supra : dict
        Output of `build_supra_adjacency`.
    t_index : int, optional
        Index into the financial layer's window list (default -1, latest).

    Returns
    -------
    dict[str, pd.DataFrame]
        {'financial': ..., 'supply_direct': ..., 'supply_coexposure': ...}
    """
    return {
        "financial": supra["financial"]["tables"][t_index],
        "supply_direct": supra["supply_direct"],
        "supply_coexposure": supra["supply_coexposure"],
    }


def _off_diagonal_pairs(theta_table):
    """Yield (i, j, theta_ij) for every ordered off-diagonal pair."""
    tickers = list(theta_table.index)
    vals = theta_table.values
    for a in range(len(tickers)):
        for b in range(len(tickers)):
            if a != b:
                yield tickers[a], tickers[b], vals[a, b]


def layer_coupling_test(theta_table, sc_direct_pairs, sc_overlap,
                        n_permutations=5000, rng=None):
    """
    Test whether supply-chain linkage predicts financial-layer connectedness.

    The headline cross-layer result: are company pairs that are linked in the
    supply-chain layer (direct edge OR shared-customer co-exposure) also more
    strongly connected in the volatility layer? Generalises the notebook's
    top-15 eyeball check to an all-pairs comparison with a proper null.

    Significance is assessed by a PERMUTATION test, not a parametric one: the
    off-diagonal theta entries are not independent (network autocorrelation), so
    a t-test / Mann-Whitney p-value would be optimistic. We instead shuffle the
    linked/unlinked labels across pairs and rebuild the null distribution of the
    mean-theta gap - which is valid under the framing that the supply-chain graph
    is a fixed prior and only the label-to-pair assignment is exchangeable.

    Parameters
    ----------
    theta_table : pd.DataFrame
        Financial-layer FEVD table (e.g. a single window's DY connectedness).
    sc_direct_pairs : set[tuple[str, str]]
        Directed/undirected pairs with a documented direct supply-chain edge
        (e.g. from supply_chain_layer.edge_set(..., undirected=True)).
    sc_overlap : pd.DataFrame
        Shared-customer co-exposure counts (symmetric).
    n_permutations : int, optional
        Number of label shuffles for the null distribution.
    rng : np.random.Generator, optional
        RNG for reproducibility (created if not supplied).

    Returns
    -------
    dict
        linked_mean, unlinked_mean, gap, p_value (one-sided: P(null gap >= real
        gap)), n_linked, n_unlinked, dose_response_corr (Pearson corr of theta
        vs shared-customer count over off-diagonal pairs).
    """
    if rng is None:
        rng = np.random.default_rng()

    thetas, linked_flags, overlaps = [], [], []
    for i, j, theta_ij in _off_diagonal_pairs(theta_table):
        thetas.append(theta_ij)
        has_direct = (i, j) in sc_direct_pairs or (j, i) in sc_direct_pairs
        shared = 0.0
        if i in sc_overlap.index and j in sc_overlap.columns:
            shared = sc_overlap.loc[i, j]
        overlaps.append(shared)
        linked_flags.append(bool(has_direct or shared > 0))

    thetas = np.asarray(thetas, dtype=float)
    linked = np.asarray(linked_flags, dtype=bool)
    overlaps = np.asarray(overlaps, dtype=float)

    n_linked = int(linked.sum())
    n_unlinked = int((~linked).sum())
    if n_linked == 0 or n_unlinked == 0:
        raise ValueError("need both linked and unlinked pairs for the test")

    linked_mean = thetas[linked].mean()
    unlinked_mean = thetas[~linked].mean()
    real_gap = linked_mean - unlinked_mean

    # Permutation null: reshuffle which pairs are "linked", keeping the count.
    null_gaps = np.empty(n_permutations)
    k = n_linked
    for p in range(n_permutations):
        perm = rng.permutation(len(thetas))
        sel = perm[:k]
        mask = np.zeros(len(thetas), dtype=bool)
        mask[sel] = True
        null_gaps[p] = thetas[mask].mean() - thetas[~mask].mean()

    # one-sided p: how often does the null produce a gap at least as large?
    p_value = (np.sum(null_gaps >= real_gap) + 1) / (n_permutations + 1)

    # dose-response: does theta scale with shared-customer count?
    if np.std(overlaps) > 0:
        dose_corr = float(np.corrcoef(thetas, overlaps)[0, 1])
    else:
        dose_corr = float("nan")

    return {
        "linked_mean": float(linked_mean),
        "unlinked_mean": float(unlinked_mean),
        "gap": float(real_gap),
        "p_value": float(p_value),
        "n_linked": n_linked,
        "n_unlinked": n_unlinked,
        "dose_response_corr": dose_corr,
    }
