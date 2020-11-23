    #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 15:09:48 2020

@author: Dezheng Xu
Description: Homework4
"""

import datetime
import pandas as pd
import pandas_datareader.data as web
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


#Problem1: Covariance Matrix Decompsition

#2 Generate a sequence of daily returns for each asset for each date.
def daily_returns(df):
    
    df1 = df.pct_change()
    
    return np.log(df1+1)

#3 Calculate the covariance matrix of daily returns and perform an eigenvalue decom-position on the matrix
def covar(df):
    
    return df.cov()

def eigenvalues(df):
    
    a,b = np.linalg.eig(df)
    return a

def eigenvector(df):
    
    a,b = np.linalg.eig(df)
    return b

#4 Calculate account for 50% of the variance
def perc(v):
    i = 0
    while sum(v[:i+1])/sum(v) < 0.5:
        i += 1
    return i 
#Calculate account for 90% of the variance
def perc2(v):
    i = 0
    while sum(v[:i+1])/sum(v) < 0.9:
        i += 1
    return i 

#return stream that represents the residual return after the principal components that correspond to these eigenvalues have been removed.
def rs(df1):
    pca = PCA(n_components = 49)    
    fr = pca.fit_transform(df1.iloc[1:])
    rm = df1.iloc[1:].values
    beta = np.dot(np.dot(np.linalg.inv(np.dot(fr.T,fr)),fr.T), rm)
    residual = rm - np.dot(fr, beta)
    r_data = pd.DataFrame(residual, index = df1.iloc[1:].index)
    plt.plot(residual)
    plt.title("Fig2. return stream ")
    plt.xlabel("Date")
    plt.ylabel("Residual Return")
    plt.show()
    return r_data
    
#Problem 2: Portfolio Construction
#1 use svd to invert GCG in a nice, stable way
def GCG_inv(df2):
    
    g = np.ones((len(v), 2))
    g1 = pd.DataFrame(g)
    g1[1][17:] = 0
    G = np.matrix(g1.T)
    C_inv = np.matrix(df2).I
    GCG_inv =  (G * C_inv * (G.T)).I
    return GCG_inv

# Calculated portfolio weight by  Lagrangian
def w(m):
    g = np.ones((len(v), 2))
    g1 = pd.DataFrame(g)
    g1[1][17:] = 0
    G = np.matrix(g1.T)
    C_inv = np.matrix(df2).I
    constant = np.matrix((1, 0.1))
    m1 = np.mean(df1)
    Rmean = np.matrix(m1)
    Lambda = m.T * (G * C_inv * Rmean.T - 2*a*constant.T)
    Weight = 1/2 * C_inv * (Rmean.T - (G.T) * Lambda)
    return Weight
if __name__ == '__main__':
    
    #1
    #get the data from 100 companies in Yahoo Finance
    #start = datetime.datetime(2015, 2, 14) 
    #end = datetime.date.today()
    #tickers = ['AIG','AMAT', 'BABA', 'BIDU', 'GOOGL', 'FB','ATHM','ABEV', 'AMZN',	'V', 'TSLA', 	'VIS',	'AAPL',	'ATVI',	'AZN', 'BA',	'BAC',	'AMD',	'HFC',	'BMY',	'C',	'CELG',	'CHTR',	'CLF',	'JAZZ',	'COP',	'COST',	'COTY',	'CTL',	'CVS',	'CX',	'CZR',	'DHI',	'EA',	'EFC',	'EMR',	'ET',	'F',	'FB',	'FCX',	'FDX',	'AR',	'GE',	'GGAL',	'GGB',	'GOLD',	'GOOG',	'GRPN',	'HAL',	'HBAN',	'IBM',	'IBN',	'INFY',	'INTC',	'ITUB',	'JNJ',	'JPM',	'KGC',	'KMI',	'KO',	'LLY',	'MDLZ',	'MGM',	'LVS',	'MRVL',	'MS',	'MSFT',	'NFLX',	'NKE',	'NLY',	'NOK',	'NTAP',	'NVDA',	'ORCL',	'PBR-A',	'PFE',	'QCOM',	'RF',	'RIG',	'RTN',	'S',	'SBUX',	'SIRI',	'SIX',	'SLB',	'SO',	'SPG',	'SR',	'SWN',	'T',	'TEVA',	'TGT',	'TSM',	'TWTR',	'TXN',	'VZ',	'WFC',	'WMB',	'XOM',	'YELP',	'VLO']
    #prices = web.DataReader(tickers, 'yahoo', start, end)
    #df = pd.DataFrame(prices['Adj Close'])
    #df.to_csv("100companies.csv",index_label="Date")  
        
    f = pd.DataFrame(df.iloc[:, 1: 101])
    f.index = df['Date']
    df1 = daily_returns(f)
    print(df1)
    df2 = covar(df1)
    print(df2)
    v = eigenvalues(df2)
    print(v)
    plt.plot(v)
    plt.title("Fig1. eigenvalues")
    plt.xlabel("Companies")
    plt.ylabel("Eigenvalues Changing")
    plt.show()
    rs(df1)
    b = eigenvector(df2)
    m = GCG_inv(df2)
    a= 1
    W = w(m)
    plt.plot(W)
    plt.title("Fig3. Portfolio Weight")
    plt.xlabel("Companies")
    plt.ylabel("Weight")
    plt.show()
    
  
    
    
    
    
    
    
    
    
    