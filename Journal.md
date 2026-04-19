This is an informal way for me to share my thoughts as I evolve through these projects. 

PROJECT 1 MST for 

UPDATE 1: I successfully implemented a MST to represent the correlation between stocks. As per Mantegna's paper on a hiearchical structure in financial markets, the MST offers an ultrametric space that enables us to model the market as a hierarchical classification of assets. In day 2, I will look to shift from a static snapshot to a dynamic one. We will keep limiting ourselves to the 10 stocks listed. Day 1 passed the test on the metric space property (I have yet to check for ultrametric) 

UPDATE 2: Decided to check for ultrametric instead on moving to dynamic. I had to change from distance matrix to a ultrametric distance matrix to be able to test that the graph respected those properties. I also went into making an analysis function for obtaining the properties of the trees and analysing them. 

UPDATE 3: I decided to refactor to respect the single responsibility principle. I want to build up on the analysis, so this will enable minimal code refactoring. I got the data to analyse.

UPDATE 4: I looked at what signals I could analyze, they didn't seem to provide new information. It's like adding noise instead of lowering it. The MST tool doesn't seem to provide enough flexibility to offer a nuanced analysis of the market. I created a shock contagion function to see how it impacts the different stocks when one stock changes. I'm not too sure of the strength of the correlations, they seem to be derived from noise despite applying a Marchenko-Pastur Filter.

UPDATE 5: 
Brainstorming:
https://www.sciencedirect.com/science/article/abs/pii/S0378437122008640
I saw this article about how someone analyzed taxi rides to predict the price of real estate. From what I understand, they broke down cab users according to time of the day. I didn't read the full thing, but I figure they're trying to isolate the commuters to see where the meaningful repetitive flow of people goes to. This proxy-based forecasting would enable to predict the trends of real estate prices.

How can we use multiple layer architecture and apply it to supply chains to predict manufacturing capabilities of companies.

Looking deeper into the subject, I found the existence of Temporal Multiplex Directed Networks. 
It's like if I took a graph with nodes and egdges, made that graph evolve through time and had different types of edges. This results in the layered architecture. 

For the semiconductor industry, the layers I'm considering are: the underlying price, the ownership and the flow of revenue between companies. I considered patents relationships, but it's hard to obtain up to date info on that.

PROJECT 2 - TMDN for Semiconductors

UPDATE 1: Half of the financial layer done. I acquired the returns of a vast amount of stocks in different spheres of the semiconductor industry. Those will be the focus of our analysis. Then, I build a lead-lag correlation matrix from those assets. This results in an assymetrical matrix where rows are the stocks at time T and the columns are the stocks at time T+1. The resulting matrix showcases how the returns of one stock impacts the returns of another over 1 day. Then we reapply the Marchenko-Pastur filter and treat only the real part of the resulting complex eigen values. We then construct graphs from the resulting matrices. 
I noticed that the graph is too dense. This is normal given that every stock is correlated to the other, but I found out about Graphical Lasso, a penalty method that allows us to transform the correlation matrix into a sparse precision one. It keeps only the conditionally dependent relationships (direct ones) reducing the others to 0.

UPDATE 2: I tried applying Graphical Lasso and noticed that it dealt with symmetrical correlation matrices. I build an assymetrical leader-lag correlation matrix. Hence, I can either induce symmetry by averaging the two assymetrical entries, which would result in a loss of directionality. This implies a loss of the leader-lagger relationship wich is crucial for our model. Or I can apply a different method. I need something that keeps conditional dependent relations while filtering out independent ones on assymetrical correlation matrices. Also, I realized my sample to variable ratio was small (T/N of 2), which is bad for forming the correlation matrix. I figured out two options, either elongate the sample size to 152 days instead of 76 and put more weight on more recent ones or sample intraday prices and risk having intraday noise.

