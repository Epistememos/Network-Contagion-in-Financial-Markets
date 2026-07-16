# Network Contagion in Financial Markets

A Python research repository exploring how shocks propagate through equity markets, using correlation networks, minimum spanning trees, and temporal multiplex directed networks — with the semiconductor industry as the laboratory.

## The two projects

### Project 1 — Market structure & contagion via Minimum Spanning Trees

Builds a hierarchical map of the market from equity return correlations:

- Correlation matrices → ultrametric distances → **MST extraction**, preserving the strongest dependencies in a connected, cycle-free graph
- **Marchenko–Pastur denoising** of the correlation spectrum before graph construction
- Tree analytics across rolling windows: medoid, average path length, total weight, survival ratio, Jaccard similarity
- A **shock contagion routine** measuring asset-level impact of single-asset perturbations

Code: [`py_scripts/project1/`](py_scripts/project1/), notebook: [`notebooks/MST.ipynb`](notebooks/MST.ipynb)

### Project 2 — Temporal Multiplex Directed Networks (TMDN)

Models the semiconductor industry (~27 US-listed names across foundries, fabless designers, memory, wafer-fab equipment, OSAT, and analog/auto/power) as a sequence of directed network layers over time, via a supra-adjacency tensor A(i, j, α, t).

The financial layer is estimated with a pipeline of:

1. **Data quality gates** — empty/stale-ticker detection (thinly traded OTC ADRs silently destroy entire spheres via `dropna`)
2. **Market-mode residualization** — the top eigenmode (~50–65% of variance) is projected out and kept as its own systematic layer; lead-lag structure is estimated on the idiosyncratic residuals
3. **Sparse VAR(p)** — one Lasso regression per target asset over all assets' lags jointly, yielding one directed, asymmetric adjacency matrix per lag (*not* Graphical Lasso, which is symmetric by construction and destroys directionality)
4. **Placebo testing** — every candidate network must beat a time-shuffled control and show positive out-of-sample R², or it is treated as noise

Code: [`py_scripts/project2/financial_layer.py`](py_scripts/project2/financial_layer.py), notebook: [`notebooks/TMDN.ipynb`](notebooks/TMDN.ipynb)

## Key findings so far

**Return lead-lag among liquid semiconductor stocks is dead.** At both hourly and daily frequency, residualized or raw, the Sparse VAR networks built on *returns* are statistically indistinguishable from time-shuffled placebos (real/placebo edge ratio ≈ 1.0) with negative out-of-sample R² in every configuration. Cross-predictability of returns in liquid large caps has been arbitraged away — a null result, honestly measured.

**Volatility spillovers are alive and economically coherent.** Following Diebold–Yilmaz (2012), the same Sparse VAR machinery applied to daily **Parkinson (range-based) log-volatility** — which is strongly persistent (lag-1 autocorrelation ≈ 0.40 vs ≈ 0 for returns) — produces:

- Positive walk-forward out-of-sample R² (+0.31 full sample), with the cross-asset network adding predictive power over an own-lags AR benchmark
- A rolling **generalized-FEVD connectedness network**: θ(i, j) = share of asset *i*'s 10-day forecast-error variance attributable to shocks from asset *j*
- Structure the math had no way of knowing: the EDA duopoly (Cadence ↔ Synopsys) as the strongest edge, a tight AMAT–LRCX–KLAC equipment cluster, STM → Infineon, and a total connectedness index that peaked (89.5%) in the December 2022 semi bear market
- Net vol **transmitters** (STM, AMAT, ASX, TER) sit upstream in the supply chain; net **receivers** include INTC and TOELY

The volatility spillover network is the working financial layer of the TMDN; ownership and supply-chain layers are future work.

## Repository structure

```
├── notebooks/
│   ├── MST.ipynb        # Project 1: MST construction, tree metrics, contagion
│   └── TMDN.ipynb       # Project 2: data gates, residualization, Sparse VAR,
│                        #   placebo tests, Diebold–Yilmaz vol spillovers
├── py_scripts/
│   ├── project1/        # mst.py, analysis.py, contagion.py
│   └── project2/
│       └── financial_layer.py   # PCA, market-mode removal, asymmetric lead-lag,
│                                #   MP/SVD denoising, var_lasso (Sparse VAR(p)),
│                                #   parkinson_log_vol, fevd_connectedness
├── data/                # MST support utilities
├── src/                 # entry-point script (Project 1)
├── tests/               # test code
├── LOG.md               # research log: what was tried, what worked, why
└── Journal.md           # informal day-to-day notes
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
python -c "import numpy, pandas, networkx, yfinance, matplotlib, sklearn, plotly; print('OK')"
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

- **Every network must pass the placebo test.** Shuffle time order (within sessions for intraday data), refit, and compare — a Lasso will happily produce a sparse "network" from pure noise. The permanent placebo cell in `TMDN.ipynb` is the validity gate for any new configuration.
- **Standardize out-of-sample data with training-window statistics.** Standardizing the test segment by its own stats leaks regime information and can mask (or fake) predictive signal.
- **Symmetric-matrix tools don't transfer to asymmetric matrices.** Eigenvector-transpose reconstruction and Marchenko–Pastur thresholds assume symmetry; the lead-lag matrix requires SVD with the quarter-circle bound s_max = 2·√(N/T) instead (handled automatically in `marchenko_pastur_returns`).
- **Denoising can destroy the signal you're after.** Hard MP truncation keeps ~1 eigenmode at these T/N ratios, making returns rank-1 — and a rank-1 series has an exactly symmetric lead-lag matrix. Hence residualization (remove the market mode) rather than truncation (keep only the market mode).
- **Watch ticker quality.** OTC ADRs (Samsung, Advantest) have no usable intraday data and stale daily closes; one all-NaN column wipes out an entire sphere through `dropna`. The data-quality cell asserts on coverage and staleness before anything downstream runs.

## References

- Mantegna (1999) — hierarchical structure in financial markets (MST)
- Laloux, Cipelletti, Bouchaud, Potters (1999) — random matrix theory and correlation denoising
- Lo & MacKinlay (1990) — lead-lag effects in stock returns
- Basu & Michailidis (2015) — regularized estimation of sparse high-dimensional VAR
- Diebold & Yilmaz (2012) — volatility spillovers via generalized FEVD
- Parkinson (1980) — range-based volatility estimation
