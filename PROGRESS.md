# Research Progress Log

Consolidated research and implementation history. Replaces `LOG.md` (formal update log) and `Journal.md` (informal notes), both of which now redirect here.

---

## Project 1 - Equity Market Contagion via Minimum Spanning Trees

### 2025-11-16 - Repository setup
Defined project scope: modelling contagion in financial markets using graph-theoretic methods on equity return correlations.

### 2025-12-27 - MST construction
Implemented minimum spanning tree extraction from equity correlation matrices. Validated metric-space properties (symmetry, triangle inequality) for a 10-stock sample. Confirmed MST construction is consistent with hierarchical market structure assumptions (Mantegna 1999). MST preserves strongest pairwise dependencies in a connected, cycle-free structure.

### 2025-12-27 - Ultrametric distance
Replaced generic distance matrix with subdominant ultrametric representation. Needed for consistent hierarchical graph interpretation across rolling windows.

### 2025-12-31 - Dynamic tree analytics
Added rolling tree analysis functions: medoid, average path length, total weight, survival ratio, Jaccard similarity. Rationale: static MSTs miss how market structure evolves; rolling metrics reveal regime changes.

### 2026-03-25 - Account switch
Original GitHub account (Gagsterslol) locked. Continued from Epistememos. Synced history from backup.

### 2026-03-26 - Marchenko–Pastur denoising
Added MP spectral filtering to the correlation matrix before MST construction. Eigenvalues below λ_max = (1+√(N/T))² are noise under the Marchenko–Pastur distribution; replaced with their mean to preserve trace. Rationale: raw finite-sample correlation matrices are noisy; MP separates signal eigenstructure from random-matrix noise (Laloux et al. 1999).

### 2026-03-26 - Refactor into notebooks and visualization
Separated data ingestion, MST generation, and analysis logic into modules. Moved from scripts to `MST.ipynb` for reproducibility.

### 2026-03-29 - Shock contagion routine + ultrametric fix
- Added `contagion.py` with `simulate_shock()`: measures asset-level impact of single-asset perturbations propagated across the MST.
- Fixed subdominant ultrametric calculation (was miscalculating for certain tree topologies).
- Finding: correlation-based MSTs are noise-sensitive - MP denoising is critical before graph formation.

### 2026-04-02 - MST.ipynb stabilized
Full pipeline runs end-to-end. Project 1 milestone complete.

---

## Project 2 - Temporal Multiplex Directed Networks (TMDN)

### 2026-04-04 - TMDN architecture
Defined supra-adjacency tensor A(i, j, α, t) and three candidate layers: financial (return lead-lag), ownership (13F filings), supply chain (revenue flows). Chose semiconductor industry as the laboratory - tightly coupled supply chain with documented contagion episodes (2019 US–China tech restrictions, 2022 CHIPS Act, TSMC fab constraints).

### 2026-04-05 - Asymmetric lead-lag matrix + MP attempt
- Built asymmetric lead-lag matrix: entry (i,j) = corr(R_i,t, R_j,t+1).
- Applied MP filtering to eigenvalues and built graph from denoised matrix.
- Problem: used the symmetric reconstruction V diag(λ) V^T, which is only valid when eigenvectors are orthogonal. Asymmetric matrices have complex eigenvalues and non-orthogonal eigenvectors - reconstruction was corrupting the matrix.

### 2026-04-06 - Graphical Lasso ruled out
Evaluated GL as a sparsification method. Decision: incompatible - GL estimates a symmetric precision matrix of contemporaneous returns, destroying the temporal directionality the TMDN depends on by construction.

### 2026-04-09 - Fixed asymmetric MP + complex eigenvalue handling
- Switched to SVD with quarter-circle law s_max = 2√(N/T) as noise threshold (correct for asymmetric random matrices under the null).
- Fixed silent imaginary-part discarding during eigendecomposition.
- T/N ratio was ~2 at this point - model critically under-specified.

### 2026-04-12 - Statistical brick wall; informal field notes
*From the working journal, written post-facto around 2026-06-07:*
> "Got stuck on returns issue. I had to replace stocks since I switched from Daily Close to hourly data, leading the non-American stocks to pollute my data. Instead I focused on American equivalents or cutting them out. I'm still facing problems with returns despite having forward filled missing values. Gonna put some focus on Z-CPY now. I have to figure out what the issue is and why I get no graphs for WFE at some point tho."

Documented core constraint: low T/N ratio (~2), Graphical Lasso incompatible, Sparse VAR growing as O(N²). Assessed paths: longer window, recency weighting, higher intraday frequency. Chose hourly as next experiment.

### 2026-04-19 - README: "Mathematical Despair"
Documented the wall in public. Investigated cluster-based VAR with differentiated intra/inter-cluster regularization as a way to reduce effective sample complexity without increasing T.

### 2026-05-03 - Graph generation refactor
Cleaned up adjacency-estimation pipeline; separated adjacency construction from graph-level analysis.

### 2026-06-07 - Switched to hourly; attempted return fixes
Changed sampling to `INTERVAL="1h"`. Rationale: 6 intraday bars per session → more samples per 252-day window. Dropped SSNLF (Samsung OTC ADR - near-zero hourly volume). Fixed ticker STNE → STM (STNE is StoneCo, a Brazilian fintech, not STMicroelectronics).

---

## Phase 3 - Null Result and Pivot (July 2026)

### 2026-07-16 - Return lead-lag is null; pivot to vol spillovers

Full rewrite of the financial layer and notebook. Multiple failures and fixes converged on a genuine empirical null.

**Hourly data problems:**
- `shift(1)` across session boundaries creates overnight/weekend log-returns (~17h gaps), contaminating intraday lead-lag. Fixed: drop first bar of each trading day for intraday data.
- 6 intraday bars/session → only 2 usable target rows/day after VAR(4) session filtering. ~70 samples vs 116 predictors per 252-day window.
- ADVNF (Advantest OTC ADR): all-NaN column silently erased the entire WFE sphere via `dropna()`. Dropped.

**VAR fix - MP truncation was destroying directionality:**
- MP truncation at hourly T/N kept only k=1 eigenmode, making residuals rank-1. A rank-1 series has an exactly symmetric lead-lag matrix - directionality destroyed.
- Fix: invert the logic. Instead of keeping the market mode, project it out (`remove_market_mode()`). Idiosyncratic residuals retain cross-asset structure; the market factor becomes its own systematic layer.

**Other bugs fixed:**
- `from matplotlib.widgets import Lasso` was importing a mouse-lasso UI widget, not sklearn's regressor. Fixed: `from sklearn.linear_model import Lasso`.
- `var_lasso()` was a stub; generalized to proper Sparse VAR(p) returning one adjacency per lag. Session exclusion auto-skips at daily frequency.

**Daily frequency - also failed:**
Switched to `INTERVAL="1d"`, `START="2021-11-01"`, 27 assets (ARM/ALAB dropped - IPO history too short). 248 samples vs 108 predictors - comfortable regime. Placebo test result:

| alpha | real edges | placebo edges | ratio | OOS R² |
|---|---|---|---|---|
| 0.01 | 2255 | 2261 | 1.00 | -0.86 |
| 0.02 | 1781 | 1800 | 0.99 | -0.49 |
| 0.05 | 887 | 895 | 0.99 | -0.16 |

Real/placebo ratio ≈ 1.0 across all α. OOS R² negative everywhere. Verdict: cross-predictability of returns in liquid large-cap semis has been arbitraged away. Consistent with Lo–MacKinlay (1990) finding lead-lag only in small vs large caps.

**Pivot to volatility spillovers:**
- Volatility cannot be directly arbitraged ("you can't short vol predictability"), so persistence survives.
- Log-vol lag-1 autocorrelation ≈ 0.40 vs ≈ 0 for returns.
- Added `parkinson_log_vol()`: range-based estimator (Parkinson 1980), ~5× more efficient than |return|.
- Added `fevd_connectedness()`: Diebold-Yilmaz (2012) generalized FEVD, Pesaran-Shin order-invariant decomposition, H=10d horizon.
- Walk-forward OOS R²: VAR cross-asset +0.31, AR own-lags +0.31 full sample. Key fix: was standardizing test segment by test-window statistics (leaks regime information). Fixed: standardize by training-window stats.
- Rolling DY connectedness (186 windows, 252d window, 5d step): peaked 89.5% in Dec 2022 semi bear market. CDNS↔SNPS EDA pair strongest edge, AMAT–LRCX–KLAC equipment cluster. Net transmitters: STM, AMAT, ASX, TER. Net vulnerable: INTC, TOELY, QCOM.

### 2026-07-18 - HAR-X replaces VAR(4) for volatility spillovers

**Why the switch:**

VAR(4) approximates vol persistence with 4 uniform lags, but volatility has well-documented long-memory multi-frequency structure (Corsi 2009, "A Simple Approximate Long-Memory Model of Realized Volatility"): participants simultaneously process yesterday's vol, last week's average, and last month's average. This produces rough-fractional-integration dynamics (d ≈ 0.4) that uniform VAR lags systematically underfit.

HAR(d,w,m) captures this with 3 parameters per pair instead of 4·N for VAR(4) - parsimonious and better motivated. The cross-asset extension (HAR-X, Bollerslev et al.) adds all other assets' daily/weekly/monthly components as predictors for each target, with Lasso selecting which survive. Recent literature (Two-Step Regularized HARX, arXiv Jan 2026) confirms HARX-Lasso outperforms VAR-Lasso for multi-asset realized vol spillovers.

**Technical implementation:**

Model: σ_{j,t} = Σ_i [ β^d_{ij} σ_{i,t-1} + β^w_{ij} σ̄^w_{i,t} + β^m_{ij} σ̄^m_{i,t} ] + ε

Design matrix: [daily, weekly, monthly] × N assets → 3N columns. One Lasso per target. The HAR(1,5,22) is converted to an equivalent VAR(22) for `fevd_connectedness()` (unchanged):
- A_1 = β^d + β^w/5 + β^m/22
- A_{2..5} = β^w/5 + β^m/22
- A_{6..22} = β^m/22

New function: `har_x_lasso()` in `financial_layer.py`. Walk-forward validation updated to compare HAR-X cross-asset vs HAR-AR own-lags benchmark. Rolling connectedness loop updated.

**Network visualizations added:** directed spillover graph (sphere-grouped circular layout, node shape = transmitter/receiver, top-20 θ edges as arrows), TO-vs-FROM risk map, and HAR coefficient heatmaps (β^d/β^w/β^m) showing at which frequency each spillover channel operates.

**Sensitivity grid - structure is robust, forecast lift is not.** Swept α ∈ {0.01, 0.02, 0.05, 0.10} × FEVD horizon H ∈ {5, 10, 20} against the chosen operating point (α=0.05, H=10):
- Top-10 edge overlap: 9–10/10 across the *entire* grid. CDNS↔SNPS, AMAT–LRCX–KLAC, LRCX→MU, STM→IFNNY are not artifacts of the specific hyperparameter choice.
- Mean connectedness index: 84.0–84.8% everywhere (<1pp range) - stable regardless of α or H.
- OOS R² delta (HAR-X cross-asset − HAR-AR own-lags) is **negative at every α**, full-sample, worsening as α grows (α=0.10 → −6.3%). Per-window breakdown still shows HAR-X beating HAR-AR in specific regimes (e.g. +2.6pp at the 2026-05-08 window), but those gains don't survive full-sample averaging.
- Conclusion: the *network structure* (who transmits to whom) is a robust, real feature of the vol data - but as a pure forecasting tool, cross-asset complexity isn't paying for itself over the full sample. The FEVD connectedness network should be framed as a risk-mapping tool, not sold on OOS R² alone. α=0.05 remains a defensible, non-cherry-picked operating point.

---

## Standing Methodological Decisions

| Decision | Reason |
|---|---|
| Daily frequency, not hourly | Hourly: 6 bars/session → ~2 usable rows/day for VAR(4); samples < predictors |
| Residualize market mode, not truncate | MP truncation at k=1 → rank-1 returns → exactly symmetric lead-lag, destroying directionality |
| Sparse VAR (Lasso per asset), not Graphical Lasso | GL estimates symmetric precision matrix of contemporaneous returns; destroys directionality |
| SVD + quarter-circle law for asymmetric matrices | V V^T reconstruction only valid for symmetric matrices with orthonormal eigenvectors |
| Standardize OOS by training-window stats | Test-window standardization leaks regime information; can fake or mask predictive signal |
| Placebo-gate every network | Lasso produces sparse "networks" from pure noise; placebo ratio and OOS R² are the validity gate |
| HAR-X instead of VAR(4) for vol | Parsimonious 3-param-per-pair capture of vol's multi-frequency long-memory structure |
| Sensitivity-grid every operating point before trusting it | Confirms structure (edges, connectedness) isn't a tuning artifact; separately reveals whether forecast-lift claims survive full-sample averaging |
| Present the FEVD network as risk-mapping, not forecasting | Cross-asset HAR-X underperforms own-lags HAR-AR full-sample at every α tested; edge structure is robust but doesn't imply OOS forecast improvement |
