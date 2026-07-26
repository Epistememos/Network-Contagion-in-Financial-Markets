import pandas as pd
import numpy as np

# Hand-curated from general semiconductor industry-structure knowledge (company
# segment classification, typical customer relationships), not from individual
# 10-K filings read and cited during this pass - no licensed supply-chain feed
# or per-filing verification backs these edges yet. TODO: replace with edges
# scraped/verified from actual 10-K business-description and customer-
# concentration sections (SEC EDGAR) to ground each edge in a citable source.
# Until then, edges are a coarse qualitative prior, unweighted by revenue share -
# useful to sanity-check the data-driven financial layer against, not to be
# read as a measured or individually-sourced flow.
#
# Channels:
#   equipment - front-end wafer fab equipment vendor -> fab operator
#   test      - back-end automated test equipment vendor -> chip company
#   eda       - EDA/design software vendor -> chip designer
#   foundry   - contract foundry -> fabless customer
#   osat      - outsourced assembly/test provider -> chip company

_FRONT_END_EQUIPMENT = ["ASML", "AMAT", "LRCX", "KLAC", "TOELY"]
_TEST_EQUIPMENT = ["TER"]
_EDA = ["CDNS", "SNPS"]
_OSAT = ["ASX", "AMKR"]

# Everyone who fabricates wafers in-house (foundries + IDMs) - buyers of front-end tools
_FAB_OPERATORS = ["TSM", "INTC", "UMC", "GFS", "MU", "TXN", "ADI", "STM", "ON", "MCHP", "IFNNY"]

# Chip designers who buy EDA tools and back-end test equipment (fabless + IDMs)
_CHIP_DESIGNERS = ["NVDA", "AMD", "AVGO", "QCOM", "MRVL", "TXN", "ADI", "NXPI",
                   "STM", "ON", "MCHP", "IFNNY", "INTC", "MU"]

# Documented foundry -> fabless customer relationships (primary/majority foundry only)
_FOUNDRY_CUSTOMERS = {
    "TSM": ["NVDA", "AMD", "AVGO", "QCOM", "MRVL"],
    "UMC": ["NXPI"],
    "GFS": ["AMD"],
}

# Companies that routinely outsource assembly/test to OSAT vendors
_OSAT_CUSTOMERS = ["NVDA", "AMD", "AVGO", "QCOM", "MRVL", "TXN", "ADI", "NXPI",
                   "STM", "ON", "MCHP", "IFNNY"]


def build_supply_chain_edges():
    """
    Directed (source, target, channel) edges: source is the upstream supplier,
    target is the downstream customer whose costs/output depend on source.

    Returns
    -------
    list[tuple[str, str, str]]
    """
    edges = []

    for vendor in _FRONT_END_EQUIPMENT:
        for fab in _FAB_OPERATORS:
            edges.append((vendor, fab, "equipment"))

    for vendor in _TEST_EQUIPMENT:
        for designer in _CHIP_DESIGNERS:
            edges.append((vendor, designer, "test"))

    for vendor in _EDA:
        for designer in _CHIP_DESIGNERS:
            edges.append((vendor, designer, "eda"))

    for foundry, customers in _FOUNDRY_CUSTOMERS.items():
        for customer in customers:
            edges.append((foundry, customer, "foundry"))

    for vendor in _OSAT:
        for customer in _OSAT_CUSTOMERS:
            edges.append((vendor, customer, "osat"))

    return edges


def supply_chain_adjacency(tickers):
    """
    Directed supply-chain adjacency restricted to `tickers`.

    Parameters
    ----------
    tickers : list[str]
        Universe to restrict the edge list to (e.g. the 27 semis in the notebook).

    Returns
    -------
    adjacency : pd.DataFrame
        adjacency.loc[source, target] = number of distinct channels connecting
        source -> target (0 if no documented relationship).
    edges_df : pd.DataFrame
        One row per (source, target, channel) edge, restricted to `tickers`.
    """
    edges = build_supply_chain_edges()
    edges_df = pd.DataFrame(edges, columns=["source", "target", "channel"])
    edges_df = edges_df[edges_df["source"].isin(tickers) & edges_df["target"].isin(tickers)]

    adjacency = pd.DataFrame(0, index=tickers, columns=tickers)
    counts = edges_df.groupby(["source", "target"]).size()
    for (src, tgt), n in counts.items():
        adjacency.loc[src, tgt] = n

    return adjacency, edges_df


def shared_customer_overlap(tickers):
    """
    Symmetric co-exposure network: weight(a, b) = number of common downstream
    customers a and b both sell to.

    Direct supply-chain edges miss peer clusters like AMAT-LRCX-KLAC (equipment
    vendors don't sell to each other) or CDNS-SNPS (EDA competitors, not
    customer/supplier). Both pairs move together because they share the same
    customer base and are exposed to the same capex/demand cycle - this
    bipartite-projection view captures that indirect channel.

    Parameters
    ----------
    tickers : list[str]

    Returns
    -------
    pd.DataFrame
        Symmetric N x N overlap count, zero diagonal.
    """
    _, edges_df = supply_chain_adjacency(tickers)
    customers_by_vendor = edges_df.groupby("source")["target"].apply(set)
    vendors = customers_by_vendor.index.tolist()

    overlap = pd.DataFrame(0, index=tickers, columns=tickers)
    for i, a in enumerate(vendors):
        for b in vendors[i + 1:]:
            shared = len(customers_by_vendor[a] & customers_by_vendor[b])
            if shared:
                overlap.loc[a, b] = shared
                overlap.loc[b, a] = shared
    return overlap


def edge_set(tickers, undirected=True):
    """
    Set of (a, b) pairs with a documented supply-chain relationship, for overlap
    checks against a data-driven network (e.g. the FEVD spillover table).

    Parameters
    ----------
    undirected : bool, optional
        If True (default), returns both (a, b) and (b, a) for every edge - a vol
        shock can propagate against the physical goods flow (e.g. a demand shock
        at a fabless customer feeds back into its foundry), so direction-agnostic
        overlap is the more defensible validity check.
    """
    _, edges_df = supply_chain_adjacency(tickers)
    pairs = set(zip(edges_df["source"], edges_df["target"]))
    if undirected:
        pairs |= {(b, a) for a, b in pairs}
    return pairs
