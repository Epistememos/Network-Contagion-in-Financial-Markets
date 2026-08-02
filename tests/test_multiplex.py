import sys
import os

import numpy as np
import pandas as pd
import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from py_scripts.project2 import multiplex as mx


TICKERS = list("ABCDE")


@pytest.fixture
def toy_layers():
    """Small synthetic multiplex: 3 financial windows + two static SC layers."""
    rng = np.random.default_rng(0)
    ends = pd.DatetimeIndex(["2024-01-05", "2024-01-12", "2024-01-19"])
    tables = []
    for _ in range(3):
        m = rng.uniform(size=(5, 5))
        np.fill_diagonal(m, 0.0)
        m = m / m.sum(axis=1, keepdims=True)  # DY row-normalized
        tables.append(pd.DataFrame(m, index=TICKERS, columns=TICKERS))

    sc_adj = pd.DataFrame(0.0, index=TICKERS, columns=TICKERS)
    sc_adj.loc["A", "B"] = 1  # one direct edge
    sc_overlap = pd.DataFrame(0.0, index=TICKERS, columns=TICKERS)
    sc_overlap.loc["A", "C"] = sc_overlap.loc["C", "A"] = 3  # one co-exposure pair
    return tables, ends, sc_adj, sc_overlap


class TestBuildSupraAdjacency:
    def test_structure_and_labels(self, toy_layers):
        tables, ends, sc_adj, sc_overlap = toy_layers
        supra = mx.build_supra_adjacency(tables, ends, sc_adj, sc_overlap)

        assert supra["layers"] == ["financial", "supply_direct", "supply_coexposure"]
        assert supra["tickers"] == TICKERS
        assert len(supra["financial"]["tables"]) == 3
        assert list(supra["financial"]["ends"]) == list(ends)
        assert supra["supply_direct"].shape == (5, 5)
        assert supra["supply_coexposure"].shape == (5, 5)

    def test_static_layers_aligned_to_financial_tickers(self, toy_layers):
        tables, ends, sc_adj, sc_overlap = toy_layers
        # scramble/extend the SC layer's labels; build should reindex to the
        # financial layer's ticker set and fill missing with 0.
        sc_adj2 = sc_adj.reindex(index=list("BACDE"), columns=list("BACDE"))
        supra = mx.build_supra_adjacency(tables, ends, sc_adj2, sc_overlap)
        assert list(supra["supply_direct"].index) == TICKERS
        assert supra["supply_direct"].loc["A", "B"] == 1

    def test_length_mismatch_raises(self, toy_layers):
        tables, ends, sc_adj, sc_overlap = toy_layers
        with pytest.raises(ValueError):
            mx.build_supra_adjacency(tables, ends[:2], sc_adj, sc_overlap)


class TestSupraAt:
    def test_returns_all_layers_for_window(self, toy_layers):
        tables, ends, sc_adj, sc_overlap = toy_layers
        supra = mx.build_supra_adjacency(tables, ends, sc_adj, sc_overlap)

        slice_latest = mx.supra_at(supra, -1)
        assert set(slice_latest.keys()) == {"financial", "supply_direct", "supply_coexposure"}
        pd.testing.assert_frame_equal(slice_latest["financial"], tables[-1])

        slice_first = mx.supra_at(supra, 0)
        pd.testing.assert_frame_equal(slice_first["financial"], tables[0])


class TestLayerCouplingTest:
    def _theta_with_signal(self, linked_pairs, boost, rng):
        """Build a theta table where linked pairs get a fixed additive boost,
        so the coupling test should detect a positive, significant gap."""
        base = rng.uniform(0.0, 0.1, size=(5, 5)).copy()
        np.fill_diagonal(base, 0.0)
        table = pd.DataFrame(base, index=TICKERS, columns=TICKERS)
        for (i, j) in linked_pairs:
            table.loc[i, j] += boost
            table.loc[j, i] += boost
        return table

    def test_detects_constructed_signal(self):
        rng = np.random.default_rng(1)
        direct_pairs = {("A", "B")}
        sc_overlap = pd.DataFrame(0.0, index=TICKERS, columns=TICKERS)
        sc_overlap.loc["A", "C"] = sc_overlap.loc["C", "A"] = 3

        # linked set is {A-B (direct), A-C (co-exposure)}; give those pairs a boost
        theta = self._theta_with_signal([("A", "B"), ("A", "C")], boost=0.5, rng=rng)
        res = mx.layer_coupling_test(theta, direct_pairs, sc_overlap,
                                     n_permutations=2000, rng=np.random.default_rng(2))

        assert res["gap"] > 0
        assert res["linked_mean"] > res["unlinked_mean"]
        assert res["p_value"] < 0.05
        assert res["n_linked"] > 0 and res["n_unlinked"] > 0

    def test_null_when_no_association(self):
        rng = np.random.default_rng(3)
        direct_pairs = {("A", "B")}
        sc_overlap = pd.DataFrame(0.0, index=TICKERS, columns=TICKERS)
        sc_overlap.loc["A", "C"] = sc_overlap.loc["C", "A"] = 3

        # theta independent of linkage: no boost -> gap should be non-significant
        base = rng.uniform(0.0, 0.1, size=(5, 5)).copy()
        np.fill_diagonal(base, 0.0)
        theta = pd.DataFrame(base, index=TICKERS, columns=TICKERS)

        res = mx.layer_coupling_test(theta, direct_pairs, sc_overlap,
                                     n_permutations=2000, rng=np.random.default_rng(4))
        assert res["p_value"] > 0.05

    def test_dose_response_is_finite_with_varied_overlap(self):
        rng = np.random.default_rng(5)
        direct_pairs = set()
        sc_overlap = pd.DataFrame(0.0, index=TICKERS, columns=TICKERS)
        sc_overlap.loc["A", "B"] = sc_overlap.loc["B", "A"] = 1
        sc_overlap.loc["A", "C"] = sc_overlap.loc["C", "A"] = 5

        base = rng.uniform(0.0, 0.1, size=(5, 5)).copy()
        np.fill_diagonal(base, 0.0)
        theta = pd.DataFrame(base, index=TICKERS, columns=TICKERS)

        res = mx.layer_coupling_test(theta, direct_pairs, sc_overlap,
                                     n_permutations=500, rng=np.random.default_rng(6))
        assert np.isfinite(res["dose_response_corr"])

    def test_raises_when_all_pairs_linked_or_none(self):
        rng = np.random.default_rng(7)
        base = rng.uniform(0.0, 0.1, size=(5, 5)).copy()
        np.fill_diagonal(base, 0.0)
        theta = pd.DataFrame(base, index=TICKERS, columns=TICKERS)
        empty_overlap = pd.DataFrame(0.0, index=TICKERS, columns=TICKERS)

        # no linked pairs at all -> cannot form both groups
        with pytest.raises(ValueError):
            mx.layer_coupling_test(theta, set(), empty_overlap,
                                   n_permutations=100, rng=rng)
