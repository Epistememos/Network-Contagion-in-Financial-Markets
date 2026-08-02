import sys
import os

import numpy as np
import pandas as pd
import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from py_scripts.project2 import financial_layer as fl


# ── Shared fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def factor_returns():
    """Returns with one dominant common factor + idiosyncratic noise, so the
    top eigenmode is unambiguous and residualization/denoising have something
    real to remove."""
    rng = np.random.default_rng(0)
    T, N = 300, 6
    market = rng.normal(size=T)
    loadings = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5])
    idio = rng.normal(scale=0.3, size=(T, N))
    values = market[:, None] * loadings[None, :] + idio
    cols = [f"A{i}" for i in range(N)]
    idx = pd.date_range("2022-01-03", periods=T, freq="B")
    return pd.DataFrame(values, index=idx, columns=cols)


@pytest.fixture
def random_returns():
    rng = np.random.default_rng(1)
    T, N = 60, 4
    idx = pd.date_range("2022-01-03", periods=T, freq="B")
    cols = ["A", "B", "C", "D"]
    return pd.DataFrame(rng.normal(size=(T, N)), index=idx, columns=cols)


# ── PCA ──────────────────────────────────────────────────────────────────────

class TestPCA:
    def test_eigenvalues_sorted_descending(self, factor_returns):
        eigenvalues, _ = fl.PCA(factor_returns)
        vals = np.real(eigenvalues)
        assert np.all(np.diff(vals) <= 1e-9), "eigenvalues must be sorted descending"

    def test_eigenvectors_orthonormal(self, factor_returns):
        _, eigenvectors = fl.PCA(factor_returns)
        V = np.real(eigenvectors)
        gram = V.T @ V
        np.testing.assert_allclose(gram, np.eye(gram.shape[0]), atol=1e-6)

    def test_top_eigenvalue_dominates_for_common_factor(self, factor_returns):
        eigenvalues, _ = fl.PCA(factor_returns)
        vals = np.real(eigenvalues)
        # a strong common factor should make the top eigenvalue much larger
        # than an even split across N=6 assets (which would be 1.0 each)
        assert vals[0] > 2.0


# ── parkinson_log_vol ────────────────────────────────────────────────────────

class TestParkinsonLogVol:
    def test_matches_closed_form(self):
        high = pd.DataFrame({"X": [110.0, 105.0]})
        low = pd.DataFrame({"X": [100.0, 100.0]})
        result = fl.parkinson_log_vol(high, low)

        rng = np.log(high / low)
        expected_var = rng ** 2 / (4 * np.log(2))
        expected = 0.5 * np.log(expected_var)

        np.testing.assert_allclose(result.values, expected.values)

    def test_zero_range_is_nan_not_error(self):
        high = pd.DataFrame({"X": [100.0]})
        low = pd.DataFrame({"X": [100.0]})
        result = fl.parkinson_log_vol(high, low)
        assert np.isnan(result.iloc[0, 0])

    def test_output_shape_matches_input(self):
        high = pd.DataFrame(np.random.uniform(101, 110, size=(20, 3)))
        low = pd.DataFrame(np.random.uniform(90, 100, size=(20, 3)))
        result = fl.parkinson_log_vol(high, low)
        assert result.shape == high.shape


# ── remove_market_mode ───────────────────────────────────────────────────────

class TestRemoveMarketMode:
    def test_shapes(self, factor_returns):
        residuals, factors = fl.remove_market_mode(factor_returns, n_modes=1)
        assert residuals.shape == factor_returns.shape
        assert factors.shape == (len(factor_returns), 1)
        assert list(residuals.columns) == list(factor_returns.columns)

    def test_residual_orthogonal_to_removed_factor(self, factor_returns):
        # Projection removes the component along the top eigenvector exactly,
        # so the residual should have (numerically) zero loading on it.
        residuals, _ = fl.remove_market_mode(factor_returns, n_modes=1)
        standardized = (factor_returns - factor_returns.mean()) / factor_returns.std()
        corr = standardized.corr().values
        _, eigenvectors = np.linalg.eigh(corr)
        top = eigenvectors[:, ::-1][:, :1]

        projection = residuals.values @ top
        np.testing.assert_allclose(projection, np.zeros_like(projection), atol=1e-8)

    def test_removes_variance_for_strong_common_factor(self, factor_returns):
        standardized = (factor_returns - factor_returns.mean()) / factor_returns.std()
        residuals, _ = fl.remove_market_mode(factor_returns, n_modes=1)
        # a strong common factor should account for a majority of standardized variance
        assert residuals.var().mean() < standardized.var().mean()


# ── assymetrical_correlation_matrix ──────────────────────────────────────────

class TestAsymmetricalCorrelationMatrix:
    def test_shape_and_labels(self, random_returns):
        result = fl.assymetrical_correlation_matrix(random_returns)
        assert result.shape == (4, 4)
        assert list(result.index) == list(random_returns.columns)
        assert list(result.columns) == list(random_returns.columns)

    def test_matches_manual_leader_lagger_computation(self, random_returns):
        result = fl.assymetrical_correlation_matrix(random_returns)

        standardized = (random_returns - random_returns.mean()) / random_returns.std()
        leader = standardized.shift(1).dropna()
        lagger = standardized.loc[leader.index]

        # Recompute independently via explicit pairwise dot products rather
        # than the function's own matrix-multiply path.
        expected = pd.DataFrame(index=random_returns.columns, columns=random_returns.columns, dtype=float)
        for i in random_returns.columns:
            for j in random_returns.columns:
                expected.loc[i, j] = (leader[i].values * lagger[j].values).sum() / (len(leader) - 1)

        np.testing.assert_allclose(result.values, expected.values.astype(float), rtol=1e-10)

    def test_not_symmetric_in_general(self, random_returns):
        result = fl.assymetrical_correlation_matrix(random_returns)
        assert not np.allclose(result.values, result.values.T)


# ── marchenko_pastur_returns ─────────────────────────────────────────────────

class TestMarchenkoPasturReturns:
    def test_symmetric_branch_zeroes_diagonal(self):
        corr = pd.DataFrame(np.eye(5), index=list("ABCDE"), columns=list("ABCDE"))
        result = fl.marchenko_pastur_returns(corr, N=5, T=1000)
        np.testing.assert_allclose(np.diag(result.values), np.zeros(5))

    def test_symmetric_branch_preserves_signal_eigenvalue(self):
        # Block structure: strong common factor among 5 assets -> one large
        # eigenvalue that should survive MP thresholding at this T/N ratio.
        rng = np.random.default_rng(2)
        N, T = 5, 250
        market = rng.normal(size=T)
        idio = rng.normal(scale=0.2, size=(T, N))
        data = market[:, None] + idio
        returns = pd.DataFrame(data, columns=list("ABCDE"))
        corr = returns.corr()

        lambda_max = (1 + np.sqrt(N / T)) ** 2
        raw_eigenvalues = np.linalg.eigh(corr.values)[0]
        assert raw_eigenvalues.max() > lambda_max, "test setup should produce a signal eigenvalue"

        denoised = fl.marchenko_pastur_returns(corr, N=N, T=T)
        # off-diagonal signal should not have been wiped out by denoising
        off_diag = denoised.values[~np.eye(N, dtype=bool)]
        assert np.abs(off_diag).max() > 0.1

    def test_asymmetric_branch_thresholds_pure_noise(self):
        # Small, purely random asymmetric matrix at a T/N ratio where every
        # singular value should fall below the quarter-circle bound.
        rng = np.random.default_rng(3)
        N, T = 5, 2000
        noise = pd.DataFrame(rng.normal(scale=1.0 / np.sqrt(T), size=(N, N)),
                             index=list("ABCDE"), columns=list("ABCDE"))
        result = fl.marchenko_pastur_returns(noise, N=N, T=T)
        np.testing.assert_allclose(result.values, np.zeros((N, N)), atol=1e-9)

    def test_asymmetric_branch_zeroes_diagonal(self):
        rng = np.random.default_rng(4)
        noise = pd.DataFrame(rng.normal(size=(5, 5)), index=list("ABCDE"), columns=list("ABCDE"))
        # break symmetry explicitly
        noise.iloc[0, 1] += 5.0
        result = fl.marchenko_pastur_returns(noise, N=5, T=50)
        np.testing.assert_allclose(np.diag(result.values), np.zeros(5))


# ── var_lasso ────────────────────────────────────────────────────────────────

class TestVarLasso:
    def test_returns_one_adjacency_per_lag(self, random_returns):
        result = fl.var_lasso(random_returns, alpha=0.01, n_lags=2)
        assert set(result.keys()) == {1, 2}
        for A in result.values():
            assert A.shape == (4, 4)
            assert list(A.index) == list(random_returns.columns)
            assert list(A.columns) == list(random_returns.columns)

    def test_high_alpha_shrinks_everything_to_zero(self, random_returns):
        result = fl.var_lasso(random_returns, alpha=100.0, n_lags=1)
        for A in result.values():
            np.testing.assert_allclose(A.values, np.zeros(A.shape))

    def test_daily_frequency_ignores_session_exclusion(self, random_returns):
        # random_returns has one bar per business day -> is_intraday is False,
        # so exclude_cross_session should be a no-op regardless of its value.
        with_exclusion = fl.var_lasso(random_returns, alpha=0.05, n_lags=1, exclude_cross_session=True)
        without_exclusion = fl.var_lasso(random_returns, alpha=0.05, n_lags=1, exclude_cross_session=False)
        np.testing.assert_allclose(with_exclusion[1].values, without_exclusion[1].values)


# ── fevd_connectedness ───────────────────────────────────────────────────────

class TestFevdConnectedness:
    def test_rows_sum_to_one(self):
        rng = np.random.default_rng(5)
        names = list("ABC")
        coefs = {1: pd.DataFrame(rng.normal(scale=0.1, size=(3, 3)), index=names, columns=names)}
        residuals = pd.DataFrame(rng.normal(size=(100, 3)), columns=names)

        table, _ = fl.fevd_connectedness(coefs, residuals, horizon=5)
        row_sums = table.sum(axis=1).values
        np.testing.assert_allclose(row_sums, np.ones(3), atol=1e-9)

    def test_no_cross_dependence_gives_identity(self):
        # zero VAR coefficients + (population) diagonal residual covariance ->
        # each asset's forecast-error variance is entirely explained by its own
        # shocks. Uses a large sample since fevd_connectedness works off the
        # *empirical* np.cov, which is never exactly diagonal at finite T.
        names = list("ABC")
        coefs = {1: pd.DataFrame(np.zeros((3, 3)), index=names, columns=names)}
        rng = np.random.default_rng(6)
        residuals = pd.DataFrame(rng.normal(scale=[1.0, 2.0, 3.0], size=(200_000, 3)), columns=names)

        table, summary = fl.fevd_connectedness(coefs, residuals, horizon=5)
        np.testing.assert_allclose(table.values, np.eye(3), atol=1e-3)
        np.testing.assert_allclose(summary["TO"].values, np.zeros(3), atol=1e-3)
        np.testing.assert_allclose(summary["FROM"].values, np.zeros(3), atol=1e-3)
        assert summary.attrs["total"] == pytest.approx(0.0, abs=1e-3)

    def test_summary_columns_and_net(self):
        rng = np.random.default_rng(7)
        names = list("ABC")
        coefs = {1: pd.DataFrame(rng.normal(scale=0.1, size=(3, 3)), index=names, columns=names)}
        residuals = pd.DataFrame(rng.normal(size=(100, 3)), columns=names)

        _, summary = fl.fevd_connectedness(coefs, residuals, horizon=5)
        assert set(summary.columns) >= {"TO", "FROM", "NET"}
        np.testing.assert_allclose(
            summary["NET"].values, (summary["TO"] - summary["FROM"]).values
        )
        assert "total" in summary.attrs


# ── har_x_lasso ──────────────────────────────────────────────────────────────

@pytest.fixture
def standardized_log_vol():
    """Standardized (zero-mean, unit-var) synthetic log-volatility panel, which
    is the input contract har_x_lasso expects."""
    rng = np.random.default_rng(11)
    T, N = 300, 5
    idx = pd.date_range("2022-01-03", periods=T, freq="B")
    cols = list("ABCDE")
    raw = pd.DataFrame(rng.normal(size=(T, N)), index=idx, columns=cols)
    return (raw - raw.mean()) / raw.std()


class TestHarXLasso:
    def test_return_contract(self, standardized_log_vol):
        names = list(standardized_log_vol.columns)
        N = len(names)
        har_coefs, var_rep, residuals = fl.har_x_lasso(standardized_log_vol, alpha=0.05)

        assert set(har_coefs.keys()) == {"d", "w", "m"}
        for block in har_coefs.values():
            assert block.shape == (N, N)
            assert list(block.index) == names
            assert list(block.columns) == names

        assert sorted(var_rep.keys()) == list(range(1, 23))
        for A in var_rep.values():
            assert A.shape == (N, N)

        # residuals aligned to the rows that survive the 22-day monthly burn-in
        assert list(residuals.columns) == names
        assert residuals.index.equals(standardized_log_vol.index[22:])
        assert not residuals.isna().any().any()

    def test_var22_reconstruction_identity(self, standardized_log_vol):
        # The HAR(1,5,22) -> VAR(22) mapping is the core invariant:
        #   A_1     = Bd + Bw/5 + Bm/22
        #   A_2..5  = Bw/5 + Bm/22
        #   A_6..22 = Bm/22
        har_coefs, var_rep, _ = fl.har_x_lasso(standardized_log_vol, alpha=0.05)
        Bd, Bw, Bm = har_coefs["d"].values, har_coefs["w"].values, har_coefs["m"].values

        np.testing.assert_allclose(var_rep[1].values, Bd + Bw / 5 + Bm / 22, atol=1e-12)
        for l in range(2, 6):
            np.testing.assert_allclose(var_rep[l].values, Bw / 5 + Bm / 22, atol=1e-12)
        for l in range(6, 23):
            np.testing.assert_allclose(var_rep[l].values, Bm / 22, atol=1e-12)

    def test_high_alpha_zeroes_all_blocks(self, standardized_log_vol):
        har_coefs, var_rep, _ = fl.har_x_lasso(standardized_log_vol, alpha=100.0)
        for block in har_coefs.values():
            np.testing.assert_allclose(block.values, np.zeros(block.shape))
        # and therefore every VAR lag is zero too
        for A in var_rep.values():
            np.testing.assert_allclose(A.values, np.zeros(A.shape))

    def test_feeds_fevd_connectedness(self, standardized_log_vol):
        # Integration: the VAR(22) rep + residuals must produce a valid FEVD
        # table (rows sum to 1) with no reshaping.
        _, var_rep, residuals = fl.har_x_lasso(standardized_log_vol, alpha=0.05)
        table, summary = fl.fevd_connectedness(var_rep, residuals, horizon=10)
        np.testing.assert_allclose(table.sum(axis=1).values, np.ones(table.shape[0]), atol=1e-9)
        assert "total" in summary.attrs
