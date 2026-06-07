# Network-Contagion-in-Financial-Markets

## Project 1 — Equity market contagion via MST

UPDATE 1:
- Implemented minimum spanning tree extraction from equity correlation matrices.
- Validated metric-space properties for a 10-stock sample and confirmed MST construction is consistent with hierarchical market structure assumptions.
- Rationale: MST extraction provides a parsimonious market graph that preserves strongest pairwise dependencies while enforcing a connected, cycle-free structure.

UPDATE 2:
- Replaced generic distance matrix input with an ultrametric distance representation to support hierarchical graph validation.
- Added tree analysis functions to compute medoid, average path length, total weight, survival ratio, and Jaccard similarity across rolling windows.
- Rationale: ultrametric distance ensures the graph aligns with hierarchical clustering assumptions and enables consistent interpretation of MST topology over time.

UPDATE 3:
- Refactored codebase to separate data ingestion, MST generation, and analysis logic.
- Prepared the pipeline for additional metric extraction and subsequent model extensions.
- Rationale: modular separation improves reproducibility, facilitates targeted debugging, and supports future extension of analytic metrics.

UPDATE 4:
- Developed a shock contagion routine to measure asset-level impact from single-asset perturbations.
- Identified noise sensitivity in correlation-based MST construction, and flagged the need for improved denoising before graph formation.
- Rationale: evaluating perturbation impacts exposes weak edges and highlights the influence of measurement noise on inferred market structure.

UPDATE 5:
- Defined the temporal multiplex directed network concept for future analysis.
- Specified candidate network layers for semiconductor analysis: price returns, ownership relationships, and revenue flow.
- Rationale: explicit layer selection establishes the framework for multi-layer network modeling and clarifies the data sources required for future work.

## Project 2 — Temporal Multiplex Directed Networks (TMDN)

UPDATE 1:
- Collected semiconductor equity returns and constructed an asymmetric lead-lag matrix with asset values at time t predicting asset values at time t+1.
- Applied Marchenko-Pastur filtering to eigenvalues and built graph structures from denoised matrices.
- Noted excessive graph density under raw lead-lag estimation.
- Rationale: asymmetric lead-lag estimation retains temporal directionality, while MP filtering aims to reduce noise in high-dimensional covariance structure.

UPDATE 2:
- Evaluated Graphical Lasso and determined it is incompatible with asymmetric lead-lag matrices without discarding directionality.
- Identified low sample-to-variable ratio (T/N ≈ 2) and assessed alternatives: extending the observation window, applying recency weighting, or increasing intraday sampling frequency.
- Rationale: preserving directionality is essential for TMDN, and the low T/N ratio signals that the model is under-specified for dense asymmetric estimation.

UPDATE 3:
- Investigated Sparse VAR for asymmetric dependency estimation.
- Documented scalability constraints: relation count grows as O(N²), so a 100-asset universe implies ~10,000 potential edges versus ~1,400 for 38 assets.
- Considered cluster-based VAR with differentiated intra-cluster/inter-cluster regularization to reduce effective sample complexity.
- Rationale: Sparse VAR provides a structured alternative to dense precision estimation, and clustering helps control the combinatorial explosion of possible edges.

UPDATE 4:
- Refined the denoising pipeline to use PCA on sliding windows, apply MP thresholding to eigenmodes, reconstruct denoised returns, and estimate sparse VAR adjacency matrices for graph construction.
- Rationale: windowed PCA + MP filtering isolates dominant signal components before sparse asymmetric dependency estimation, reducing the impact of noise on inferred relationships.
