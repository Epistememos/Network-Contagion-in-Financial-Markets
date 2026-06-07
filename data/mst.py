import numpy as np
import pandas as pd
import networkx as nx

def mst (returns):

    #OBTAINING DISTANCE MATRIX
    #correlation matrix (p_ij) (.corr() -> Pearson correlation coefficient)
    #Pearson used in the Mantegna paper -> Pearson deals with continuous data (while Spearman and Kendall is for ordinal)
    corr_matrix = returns.corr()
    
    #corr_matrix -> dist_matrix ( d(i, j ) = 1 − ρ_ij^2 )
    #not sure which one to use, I've read online that sqrt(2(1-p)) is favorable, one because it preserves direction, 2. because its euclidian as well
    dist_matrix = (1-(corr_matrix ** 2))
    dist_matrix_euclidian = np.sqrt(2 * (1 - corr_matrix))

    #BUILDING MST
    #graph G, nodes := stocks, edges := dist_matrix entry
    G = nx.from_pandas_adjacency(dist_matrix) 
    mst = nx.minimum_spanning_tree(G)

    #ANALYSIS
    #Medoid (Central stock with the most influence according to correlation)
    #closeness centrality: reciprocal of the avg. shortest path distance
    closeness = nx.closeness_centrality(mst) 
    #def of closeness => max of closeness := medoid
    medoid = max(closeness, key=closeness.get)
    #diameter
    diameter = nx.diameter(mst)
    size = mst.size("weight")

    return [mst, medoid, diameter, size]