import sys
import os
import yfinance as yf
import numpy as np
import pandas as pd
import networkx as nx
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import mst

def rolling_window_mst ():
    #OBTAINING DATA
    results = []
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'AVGO', 'COST', 'PEP']
    data = yf.download(tickers, start="2023-01-01", progress=False)['Close']
    #OBTAINING RETURNS
    #Yi = lnPi(t) − lnPi(t − 1)
    returns = np.log(data /data.shift(1)).dropna()
    window = 60
    rolling_window_data = returns.rolling(window).dropna()

    for i in range(window, len(returns)):
        window_slice = returns.iloc[i-window : i]
        current_date = returns.index[i]
        tree, medoid, diameter, size = mst.mst(window_slice)
        results.append((current_date, tree, medoid, diameter, size))
    return results

def analysis (results):
    analysis_results = {
        'medoid_changes': [],
        'diameter_changes': [],
        'edge_changes': [],        
    }
    
    for i in range(1, len(results)):
        date = results[i][0]
        current_mst = results[i][1]
        previous_mst = results[i-1][1]
        current_medoid = results[i][2]
        previous_medoid = results[i-1][2]
        current_diameter = results[i][3]
        previous_diameter = results[i-1][3]
        current_size = results[i][4]
        previous_size = results[i-1][4]

        #tracking changes in medoid
        if current_medoid != previous_medoid:
            analysis_results['medoid_changes'].append([i, current_medoid])
        #changes in diameter can showcase market crashes
        if current_diameter != previous_diameter:
            analysis_results['diameter_changes'].append([i, current_diameter])

        #survival ratio & jaccard
        prev_edges = set(previous_mst.edges())
        curr_edges = set(current_mst.edges())
        edge_similar = len(prev_edges & curr_edges)
        survival_ratio = edge_similar/(len(curr_edges))
        jaccard_sim= edge_similar/(len(prev_edges or curr_edges))

        diameter_diff = current_diameter - previous_diameter
        size_diff = current_size - previous_size
        
    return (analysis_results)