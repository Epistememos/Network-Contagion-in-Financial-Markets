# Network Contagion in Financial Markets

A Python research repository exploring how shocks propagate through equity markets, using correlation networks, minimum spanning trees, and temporal multiplex directed networks - with the semiconductor industry as the laboratory.

## The two projects

### Project 1 - Market structure & contagion via Minimum Spanning Trees

Builds a hierarchical map of the market from equity return correlations:

- Correlation matrices → ultrametric distances → **MST extraction**, preserving the strongest dependencies in a connected, cycle-free graph
- **Marchenko–Pastur denoising** of the correlation spectrum before graph construction
- Tree analytics across rolling windows: medoid, average path length, total weight, survival ratio, Jaccard similarity
- A **shock contagion routine** measuring asset-level impact of single-asset perturbations

Code: [`py_scripts/project1/`](py_scripts/project1/), notebook: [`notebooks/MST.ipynb`](notebooks/MST.ipynb)

### Project 2 - Temporal Multiplex Directed Networks (TMDN)

Models the semiconductor industry (~27 US-listed names across foundries, fabless designers, memory, wafer-fab equipment, OSAT, and analog/auto/power) as a sequence of directed network layers over time, via a supra-adjacency tensor A(i, j, α, t).

The financial layer is estimated with a pipeline of:

1. **Data quality gates** - empty/stale-ticker detection (thinly traded OTC ADRs silently destroy entire spheres via `dropna`)
2. **Market-mode residualization** - the top eigenmode (~50–65% of variance) is projected out and kept as its own systematic layer; lead-lag structure is estimated on the idiosyncratic residuals
3. **HAR-X with Lasso** - Heterogeneous AutoRegression with cross-asset terms: each asset's vol is regressed on all other assets' daily (t-1), weekly (5-day avg), and monthly (22-day avg) volatility components, with Lasso selecting which cross-asset channels survive. Three parameters per pair capture vol's long-memory multi-frequency structure more efficiently than VAR(p). (*not* Graphical Lasso, which is symmetric and destroys directionality)
4. **Generalized FEVD connectedness** - HAR(1,5,22) is converted to an equivalent VAR(22) for the Diebold–Yilmaz FEVD; θ(i,j) = share of asset i's 10-day forecast-error variance attributable to shocks from j
5. **Placebo testing** - returns are gated with an i.i.d. time-shuffle; the volatility layer uses a **per-asset circular-shift** placebo (preserves each series' own autocorrelation, destroys cross-asset alignment) since an i.i.d. shuffle is invalid for autocorrelated log-vol
6. **Supply-chain cross-check** - a hand-curated industry adjacency (direct customer/supplier edges + shared-customer co-exposure) checks whether the financial layer's top edges have a plausible economic rationale, or are noise that happened to survive the placebo test
7. **Multiplex coupling** - the financial and supply-chain layers are assembled into the supra-adjacency tensor A(i, j, α, t) and tested for coupling: do supply-chain-linked pairs have systematically higher volatility connectedness? (permutation test)

Code: [`py_scripts/project2/financial_layer.py`](py_scripts/project2/financial_layer.py), [`py_scripts/project2/supply_chain_layer.py`](py_scripts/project2/supply_chain_layer.py), [`py_scripts/project2/multiplex.py`](py_scripts/project2/multiplex.py); notebook: [`notebooks/TMDN.ipynb`](notebooks/TMDN.ipynb)

## Key findings so far

**Return lead-lag among liquid semiconductor stocks is dead.** At both hourly and daily frequency, residualized or raw, the Sparse VAR networks built on *returns* are statistically indistinguishable from time-shuffled placebos (real/placebo edge ratio ≈ 1.0) with negative out-of-sample R² in every configuration. Cross-predictability of returns in liquid large caps has been arbitraged away - a null result, honestly measured.

**Volatility spillovers are alive and economically coherent.** Following Diebold–Yilmaz (2012), a **HAR-X** model (Corsi 2009) applied to daily **Parkinson (range-based) log-volatility** - strongly persistent (lag-1 autocorrelation ≈ 0.40 vs ≈ 0 for returns) - produces:

- Positive walk-forward out-of-sample R² (+0.35 full sample, as of the 2026-07-25 run) for the HAR-AR own-lags benchmark; HAR-X cross-asset model adds lift in some sub-windows but not full-sample (see Robustness below)
- A rolling **generalized-FEVD connectedness network**: θ(i, j) = share of asset *i*'s 10-day forecast-error variance attributable to shocks from asset *j*
- Structure the math had no way of knowing: the EDA duopoly (Cadence ↔ Synopsys) as the strongest edge, a tight AMAT–LRCX–KLAC equipment cluster, STM → Infineon, and a total connectedness index that peaked (89.5%) in the December 2022 semi bear market
- Net vol **transmitters** (STM, AMAT, ASX, TER) sit upstream in the supply chain; net **receivers** include INTC and TOELY

**A second layer - supply chain - exists as both a structural cross-check and a genuine multiplex layer.** `supply_chain_layer.py` encodes the semiconductor industry's known structure (equipment → fab, EDA/test → chip designer, foundry → fabless customer, OSAT → chip company) plus a shared-customer co-exposure view, which explains peer clusters that direct edges miss (AMAT-LRCX-KLAC don't sell to each other; CDNS/SNPS are competitors, not customer/supplier - both pairs move together because they share the same customers). **Caveat:** this layer is currently built from general industry-structure knowledge, not from individually-cited 10-K filings - scraping and verifying each edge against SEC EDGAR filings is planned follow-up work, not yet done. Ownership (13F) remains future work.

**The two layers are coupled - the multiplex is more than the sum of its layers.** `multiplex.py` assembles the supra-adjacency tensor A(i, j, α, t) (financial layer temporal, supply-chain layers static) and tests the headline cross-layer question over *all* pairs: supply-chain-linked pairs have significantly higher volatility connectedness than unlinked pairs (mean θ 0.032 vs 0.030, gap +0.003, permutation p ≈ 0.01 over 5,000 shuffles), and θ rises with the number of shared customers (dose-response corr ≈ +0.28). The data-driven financial network tracks real industry structure, not just statistical co-movement.

The volatility spillover network is the working financial layer of the TMDN.

### Robustness (as of the 2026-08-01 run, data through 2026-07-31)

Four independent checks, documented in full in the notebook's "Financial Layer Robustness Summary", "Supply-Chain Rationale & Overall Verdict", and "Multiplex Tensor & Cross-Layer Coupling" sections:

- **Return null result is decisive, not marginal.** Placebo edge ratio 0.96-0.98 across α ∈ {0.01, 0.02, 0.03, 0.05} - the real return network has *fewer* surviving edges than shuffled noise at every regularization level, plus OOS R² of -0.14 to -0.72.
- **Vol-layer structure survives a 4×3 hyperparameter grid** (α × FEVD horizon): 9-10/10 top-edge overlap and connectedness confined to 84.0-84.7% everywhere. But the OOS R² delta (HAR-X cross-asset vs. HAR-AR own-lags) is negative at *every* configuration and degrades monotonically with α - the layer adds no forecast lift and should be read as risk-mapping, not prediction.
- **Vol-layer placebo test now run (previously an open gap).** A per-asset circular-shift null (valid for autocorrelated log-vol, unlike an i.i.d. shuffle) collapses the total connectedness index from a real 80.3% to 18.7% ± 1.7% (real +36.5 sd above the placebo band, non-overlapping), and placebos reproduce only ~0.4/10 of the real top edges. The cross-asset connectedness is real, not an autocorrelation artifact. (Magnitude metrics like coefficient mass/concentration are deliberately *not* used - destroying the common structure spuriously concentrates the placebo FEVD, so they move the "wrong" way; total connectedness is the honest structure-level statistic.)
- **Supply-chain economic plausibility: 73% hit rate, but indirect.** Of the top-15 vol-spillover edges, 11/15 have a documented rationale, but only 2/15 are direct customer/supplier links - the dominant mechanism is shared-customer co-exposure (peer clusters exposed to the same capex/demand cycle), not commercial linkage. 4/15 edges have no rationale under either view.

## Repository structure

```
├── notebooks/
│   ├── MST.ipynb        # Project 1: MST construction, tree metrics, contagion
│   └── TMDN.ipynb       # Project 2: data gates, residualization, HAR-X,
│                        #   placebo tests, Diebold–Yilmaz vol spillovers,
│                        #   network/risk-map/coefficient visualizations
├── py_scripts/
│   ├── project1/        # mst.py, analysis.py, contagion.py
│   └── project2/
│       ├── financial_layer.py   # PCA, market-mode removal, asymmetric lead-lag,
│       │                        #   MP/SVD denoising, har_x_lasso (HAR-X),
│       │                        #   var_lasso (Sparse VAR(p)), circular_shift_placebo,
│       │                        #   parkinson_log_vol, fevd_connectedness
│       ├── supply_chain_layer.py # hand-curated industry adjacency (direct edges
│       │                        #   + shared-customer co-exposure); TODO: verify
│       │                        #   each edge against SEC EDGAR 10-K filings
│       └── multiplex.py         # supra-adjacency tensor A[i,j,alpha,t] assembly
│                                #   + cross-layer coupling test (permutation)
├── data/                # MST support utilities
├── src/                 # entry-point script (Project 1)
├── tests/               # test_financial_layer.py + test_multiplex.py (pytest, 34 tests)
└── PROGRESS.md          # full research log: decisions, pivots, null results, methodology
```

## Setup

Python 3.10+ recommended.

```bash
# macOS / Linux
python3 -m venv venv
source venv/bin/activate

# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1
```

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

If pip is configured against a private index you can't reach, install from public PyPI explicitly:

```bash
pip install --index-url https://pypi.org/simple -r requirements.txt
```

Verify:

```bash
python -c "import numpy, pandas, networkx, yfinance, matplotlib, sklearn, plotly, pytest; print('OK')"
```

Run the unit tests:

```bash
pytest tests/
```

## Running

Market data is downloaded on the fly from Yahoo Finance via `yfinance` (internet required; no local datasets).

```bash
# Project 1 entry point
python src/main.py

# The main work lives in the notebooks
jupyter lab notebooks/TMDN.ipynb
```

`TMDN.ipynb` runs top to bottom: universe definition → price/return acquisition and quality gates → market-mode residualization → Sparse VAR + placebo validation → volatility spillover layer with rolling connectedness.

## Methodological notes (read before extending)

- **Every network must pass the placebo test - but match the null to the data.** For returns, an i.i.d. time-shuffle is the right control (returns have ~0 autocorrelation). For volatility it is *not*: log-vol's ~0.40 autocorrelation makes an i.i.d. shuffle trivially beatable, so the vol layer uses a per-asset **circular shift** (preserves own autocorrelation, destroys cross-asset alignment) via `fl.circular_shift_placebo`. And choose the test statistic carefully: for the vol layer, use the **total connectedness index** (structure-level), not coefficient mass/concentration - destroying the common structure spuriously concentrates the placebo FEVD, so magnitude metrics move the wrong way.
- **Standardize out-of-sample data with training-window statistics.** Standardizing the test segment by its own stats leaks regime information and can mask (or fake) predictive signal.
- **Symmetric-matrix tools don't transfer to asymmetric matrices.** Eigenvector-transpose reconstruction and Marchenko–Pastur thresholds assume symmetry; the lead-lag matrix requires SVD with the quarter-circle bound s_max = 2·√(N/T) instead (handled automatically in `marchenko_pastur_returns`).
- **Denoising can destroy the signal you're after.** Hard MP truncation keeps ~1 eigenmode at these T/N ratios, making returns rank-1 - and a rank-1 series has an exactly symmetric lead-lag matrix. Hence residualization (remove the market mode) rather than truncation (keep only the market mode).
- **Watch ticker quality.** OTC ADRs (Samsung, Advantest) have no usable intraday data and stale daily closes; one all-NaN column wipes out an entire sphere through `dropna`. The data-quality cell asserts on coverage and staleness before anything downstream runs.
- **HAR-X, not VAR(p), for volatility.** Vol has long-memory multi-frequency persistence (Corsi 2009); VAR(4) underfits the monthly component. HAR(d,w,m) captures daily/weekly/monthly components with 3 parameters per pair instead of p·N for VAR(p). For FEVD, HAR(1,5,22) converts to an equivalent VAR(22) via A₁ = β^d + β^w/5 + β^m/22, A₂₋₅ = β^w/5 + β^m/22, A₆₋₂₂ = β^m/22.

## References

- Mantegna (1999) - hierarchical structure in financial markets (MST)
- Laloux, Cipelletti, Bouchaud, Potters (1999) - random matrix theory and correlation denoising
- Lo & MacKinlay (1990) - lead-lag effects in stock returns
- Basu & Michailidis (2015) - regularized estimation of sparse high-dimensional VAR
- Diebold & Yilmaz (2012) - volatility spillovers via generalized FEVD
- Parkinson (1980) - range-based volatility estimation
- Corsi (2009) - HAR model for long-memory realized volatility
