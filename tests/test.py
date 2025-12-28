import sys
import os
import numpy as np
import pandas as pd
import networkx as nx
import itertools
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import mst


def test_mst_discrete_properties(mst):
    print("=== MST PROP. VALIDATION SUITE ===")
    
    #Ultrametric:  d(i,j) <= Max{d(i,k),d(k,j)
    nodes = list(mst.nodes)
    
    # Obtaining ultrametric from mst
    def ultrametric_form(mst_graph):
        #initializing entries to 0.0 (lowest possible value)
        U = pd.DataFrame(0.0, index=nodes, columns=nodes)
        for p, q in itertools.combinations(nodes, 2):
            path = nx.shortest_path(mst_graph, source=p, target=q, weight='weight')
            #iterating through every edge of the path to find the max weight
            max_w = max(mst_graph[u][v]['weight'] for u,v in zip(path[:-1], path[1:]))
            #replacing symmetrically the ulltrametric distance
            U.loc[p,q] = max_w
            U.loc[q,p] = max_w
        return U

    ultrametric = ultrametric_form(mst)

    violations = 0
    #checking the ultrametric property for every triplets
    for i,j,k in itertools.combinations(nodes, 3):
        d_kj = ultrametric.loc[k,j]
        d_ik = ultrametric.loc[i,k]
        d_ij = ultrametric.loc[i,j]
        #Ultrametric property
        if d_ij > (max(d_ik, d_kj) + 1e-9):
            violations += 1
    print(f"[Ultrametric] Inequality violations: {violations}")

    # Topology: Tree Properties. (Connected with N-1 edges => no cycles as well)
    v = len(mst.nodes)
    e = len(mst.edges)
    is_connected = nx.is_connected(mst)
    print(f"[Topology] Connected: {is_connected}, Edges: {e} (Expected: {v-1})")

if __name__ == "__main__":
    test_mst_discrete_properties( mst.mst)
    
    
        