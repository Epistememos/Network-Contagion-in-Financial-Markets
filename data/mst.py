import yfinance as yf
import numpy as np
import pandas as pd
import networkx as nx

#OBTAINING DATA
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'AVGO', 'COST', 'PEP']
data = yf.download(tickers, start="2023-01-01")['Close']

#OBTAINING RETURNS
#Yi = lnPi(t) − lnPi(t − 1)
returns = np.log(data /data.shift(1)).dropna()

#OBTAINING DISTANCE MATRIX
#correlation matrix (p_ij) (.corr() -> Pearson correlation coefficient)
#Pearson used in the Mantegna paper -> Pearson deals with continuous data (while Spearman and Kendall is for ordinal)
corr_matrix = returns.corr()
#corr_matrix -> dist_matrix ( d(i, j ) = 1 − ρ_ij^2 )
dist_matrix = np.sqrt(2 * (1 - corr_matrix))

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