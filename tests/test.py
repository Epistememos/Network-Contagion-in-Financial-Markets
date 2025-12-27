import sys
import os
import numpy as np
import networkx as nx
import itertools
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import mst

def test_mst_discrete_properties(dist_matrix, mst):
    print("=== MST PROP. VALIDATION SUITE ===")

    #Metric: Test triangle inequality (N.B. Ultrametric is stronger than that)
    nodes = dist_matrix.columns
    triplets = list(itertools.combinations(nodes,3)) #making triplets using itertools
    violations = 0
    for i,j,k in triplets:
        d_kj = dist_matrix.loc[k,j]
        d_ik = dist_matrix.loc[i,k]
        d_ij = dist_matrix.loc[i,j]
        #used d_ij >= d_ik + d_kj initially, AI suggests tolerance instead due to floating point arithmetic (Suggestion Approved)
        if d_ij > (d_ik + d_kj + 1e-9):
            violations += 1
    print(f"[Metric] Triangle Inequality violations: {violations}")

    # Topology: Tree Props. (Connected with N-1 edges => no cycles as well)
    v = len(mst.nodes)
    e = len(mst.edges)
    is_connected = nx.is_connected(mst)
    print(f"[Topology] Connected: {is_connected}, Edges: {e} (Expected: {v-1})")

if __name__ == "__main__":

    test_mst_discrete_properties(mst.dist_matrix, mst.mst)
    
    
        