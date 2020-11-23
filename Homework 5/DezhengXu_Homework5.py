#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 08:27:50 2020

@author: DezhengXu

Description: MF796 Homework5 include 3 questions

"""

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import pandas as pd
import math
import time
from scipy import interpolate
import cmath
from scipy.optimize import minimize
from scipy.optimize import root
import seaborn as sns
from sklearn.linear_model import LinearRegression
from scipy.optimize import fsolve


import tablib


#1 Implementation of Breeden-Litzenberger

def call_K(delta, sigma, T):
    k = 100 * np.exp((sigma ** 2) / 2 * T - norm.ppf(delta) * sigma * (T ** 0.5) )
    return np.array(k)

def Put_K(delta, sigma, T):
    k = 100 * np.exp((sigma ** 2) / 2 * T + norm.ppf(delta) * sigma * (T ** 0.5) )
    return np.array(k)

def B_S(K, sigma, T):
    d1 = (np.log(100 / K)+(sigma ** 2/ 2) * T)/(sigma*T**0.5)
    d2 = d1- sigma*T**0.5
    return(norm.cdf(d1)*100-norm.cdf(d2)*K)

def BL_Density (K_l, v_l, T):
    density = []
    
    for i in range(1,len(K_l)-1):
        ck_h = B_S(K_l[i-1],v_l[i-1],T)
        ck = B_S(K_l[i],v_l[i],T)
        ckh = B_S(K_l[i+1],v_l[i+1],T)
        density += [(ck_h-2*ck+ckh)/0.01]
    return density
def BL_Density2 (K_l, v_l, T):
    density = []
    
    for i in range(1,len(K_l)-1):
        ck_h = B_S(K_l[i-1],v_l,T)
        ck = B_S(K_l[i],v_l,T)
        ckh = B_S(K_l[i+1],v_l,T)
        density += [(ck_h-2*ck+ckh)/0.01]
    return density

def e1(density, S):
    price = 0
    for i in range(0,len(S)-2):
        price += density[i]*ef1(S[i])*0.1
    
    return price
def ef1(s):
    if s <= 110:
        return 1
    else:
        return 0

def e2(density, S):
    price = 0
    for i in range(0,len(S)-2):
        price += density[i]*ef2(S[i])*0.1
    
    return price

def ef2(s):
    if s >= 105:
        return 1
    else:
        return 0

def e3(density, S):
    price = 0
    for i in range(0,len(S)-2):
        price += density[i]*max(0,S[i]-100)*0.1
    
    return price

#3 Hedging Under Heston Model:
class FFT:
    def __init__(self,lst):
        self.sigma = lst[0]
        self.eta0 = lst[1]
        self.kappa = lst[2]
        self.rho = lst[3]
        self.theta = lst[4]
        self.S0 = 267.15
        self.r = 0.015
        self.q = 0.0177
        self.T = 0.25
        
    def Heston_fft(self,alpha,n,B,K):
        """ Define a function that performs fft on Heston process
        """
        bt = time.time()
        r = self.r
        T = self.T
        S0 = self.S0
        N = 2**n
        Eta = B / N
        Lambda_Eta = 2 * math.pi / N
        Lambda = Lambda_Eta / Eta
        
        J = np.arange(1,N+1,dtype = complex)
        vj = (J-1) * Eta
        m = np.arange(1,N+1,dtype = complex)
        Beta = np.log(S0) - Lambda * N / 2
        km = Beta + (m-1) * Lambda
        
        ii = complex(0,1)
        
        Psi_vj = np.zeros(len(J),dtype = complex)
        
        for zz in range(0,N):
            u = vj[zz] - (alpha + 1) * ii
            numer = self.Heston_cf(u)
            denom = (alpha + vj[zz] * ii) * (alpha + 1 + vj[zz] * ii)
            
            Psi_vj [zz] = numer / denom
            
        # Compute FTT
        xx = (Eta/2) * Psi_vj * np.exp(-ii * Beta * vj) * (2 - self.dirac(J-1))
        zz = np.fft.fft(xx)
        
        # Option price
        Mul = np.exp(-alpha * np.array(km)) / np.pi
        zz2 = Mul * np.array(zz).real
        k_List = list(Beta + (np.cumsum(np.ones((N, 1))) - 1) * Lambda)
        Kt = np.exp(np.array(k_List))
       
        Kz = []
        Z = []
        for i in range(len(Kt)):
            if( Kt[i]>1e-16 )&(Kt[i] < 1e16)& ( Kt[i] != float("inf"))&( Kt[i] != float("-inf")) &( zz2[i] != float("inf"))&(zz2[i] != float("-inf")) & (zz2[i] is not  float("nan")):
                Kz += [Kt[i]]
                Z += [zz2[i]]
        tck = interpolate.splrep(Kz , np.real(Z))
        price =  np.exp(-r*T)*interpolate.splev(K, tck).real
        et = time.time()
        
        runt = et-bt

        return(price,runt)
    
    def dirac(self,n):
        """ Define a dirac delta function
        """
        y = np.zeros(len(n),dtype = complex)
        y[n==0] = 1
        return y
        
    def Heston_cf(self,u):
        """ Define a function that computes the characteristic function for variance gamma
        """
        sigma = self.sigma
        eta0 = self.eta0
        kappa = self.kappa
        rho = self.rho
        theta = self.theta
        S0 = self.S0
        r = self.r
        T = self.T
        q = self.q
        
        ii = complex(0,1)
        
        l = cmath.sqrt(sigma**2*(u**2+ii*u)+(kappa-ii*rho*sigma*u)**2)
        w = np.exp(ii*u*np.log(S0)+ii*u*(r-q)*T+kappa*theta*T*(kappa-ii*rho*sigma*u)/sigma**2)/(cmath.cosh(l*T/2)+(kappa-ii*rho*sigma*u)/l*cmath.sinh(l*T/2))**(2*kappa*theta/sigma**2)
        y = w*np.exp(-(u**2+ii*u)*eta0/(l/cmath.tanh(l*T/2)+kappa-ii*rho*sigma*u))
        
        return y  
   
        
def BS3(K, S, sigma, r, q, T):
    d1 = (np.log(S/K)+(r-q+sigma**2/2)*T)/(sigma*T**0.5)
    d2 = d1- sigma*T**0.5
    return(norm.cdf(d1)*S-norm.cdf(d2)*K*np.exp(-r*T))
    

if __name__ == '__main__':
    
    #for call 
    #1
    """
    T = 1/12
    sigma = np.array([0.3225, 0.2473, 0.2021, 0.1824])
    delta = np.array([0.1, 0.25, 0.40, 0.50])
    p1 = Put_K(delta, sigma, T)
    print(Put_K(delta, sigma, T))
    T = 3/12
    sigma = np.array([0.2836, 0.2178, 0.1818, 0.1645])
    delta = np.array([0.1, 0.25, 0.40, 0.50])
    p2 = Put_K(delta, sigma, T)
    print(Put_K(delta, sigma, T))
    
    T = 1/12
    sigma = np.array([0.1824, 0.1574, 0.1370, 0.1148])
    delta = np.array([0.5, 0.40, 0.25, 0.10])
    c1 = call_K(delta, sigma, T)
    print(call_K(delta, sigma, T))
    T = 3/12
    sigma = np.array([0.1645, 0.1462, 0.1256, 0.1094])
    delta = np.array([0.5, 0.40, 0.25, 0.10])
    c2 = call_K(delta, sigma, T)
    print(call_K(delta, sigma, T))
    df1 = pd.DataFrame(np.random.random(size = (7, 2)), columns =['1M', '3M'] )
    df1['Expiry/Strike'] = np.array([0.1, 0.25, 0.40, 0.5, 0.40, 0.25, 0.10])
    df1.index = df1['Expiry/Strike'] 
    df1['1M'] = p1.tolist() + c1[1:].tolist()
    df1['3M'] = p2.tolist() + c2[1:].tolist()
    df2 = pd.DataFrame(df1.iloc[:, 0:2 ])
    df3 = pd.DataFrame(np.random.random(size = (7, 2)), columns =['1M', 'sigma'] )
    df3['1M'] = p1.tolist() + c1[1:].tolist()
    df3['sigma'] = np.array([0.3225, 0.2473, 0.2021, 0.1824,0.1574, 0.1370, 0.1148])
    df4 = pd.DataFrame(np.random.random(size = (7, 2)), columns =['sigma', '3M'] )
    df4['3M'] = p2.tolist() + c2[1:].tolist()
    df4['sigma'] = np.array([0.2836, 0.2178, 0.1818, 0.1645,0.1462, 0.1256, 0.1094])
    sns.pairplot(df3, x_vars='1M', y_vars='sigma', size=7, aspect=0.8, kind='reg')
    regr1 = LinearRegression()
    x1 =np.array(df3['1M'])
    x1 = x1.reshape(-1,1)
    y1 = df3['sigma']
    regr1.fit(x1,y1)
    print(regr1.coef_, regr1.intercept_)
    sns.pairplot(df4, x_vars='3M', y_vars='sigma', size=7, aspect=0.8, kind='reg')
    regr = LinearRegression()
    x =np.array(df4['3M'])
    x = x.reshape(-1,1)
    y = df4['sigma']
    regr.fit(x,y)
    print(regr.coef_, regr.intercept_)
    """
    """
    #2
    K_l = np.linspace(75,112.5,375)
    v_1 = 1.5578 - 0.0138 * K_l
    v_3 = 0.9324 - 0.0077 * K_l
    B1 = BL_Density (K_l, v_1, 1/12)
    B3 = BL_Density (K_l, v_3, 1/4)
    df1 = pd.DataFrame(np.array([B1,B3]).T,index =np.linspace(75,112.5,373), columns = ['1M', '3M'])
    df1.plot()
    plt.title("Fig.1 Risk Neutral Density for 1 & 3 Month Options")
    plt.xlabel("K")
    plt.ylabel("Density")
    plt.show()
    K_l2 = np.linspace(60,140,800)
    density1 = BL_Density2(K_l2, 0.1824, 1/12)
    density3 = BL_Density2(K_l2, 0.1645, 3/12)
    df1 = pd.DataFrame(np.array([density1,density3]).T,index =np.linspace(60,140,798), columns = ['1M', '3M'])
    df1.plot()
    plt.title("Fig.2 Risk Neutral Density for 1 & 3 Month Options")
    plt.xlabel("K")
    plt.ylabel("Density")
    plt.show()
    vol_2 = (v_1+v_3)/2
    density2 = np.array(BL_Density(K_l, vol_2, 2/12))
    S = np.linspace(75,112.5,375)
    
    print(e1(B1,S))
    print(e2(B3,S))
    print(e3(density2,S))
    """
    kappa = 3.52
    theta = 0.05
    sigma = 1.18
    rho = -0.775
    eta0 = 0.034
    S0 = 267.15
    r = 0.015
    q = 0.0177
    expT = 0.25
    K = 275
    alpha = 1
    n = 15
    B = 1000
    parm = [sigma,eta0,kappa,rho,theta,S0]
    call_heston = FFT(parm).Heston_fft(alpha,n,B,K)[0]
    vol = root(lambda x: BS3(K, S0, x, r, q, expT)-call_heston,0.3).x[0]
    print(vol)
    
    ds = 0.01
    dv = 0.05 * eta0
    
    parm1 = [sigma,eta0,kappa,rho,theta,S0+ds]
    parm2 = [sigma,eta0,kappa,rho,theta,S0-ds]
    parm3 = [sigma,eta0+dv,kappa,rho,theta+dv,S0]
    parm4 = [sigma,eta0-dv,kappa,rho,theta-dv,S0]
    delta_heston = (FFT(parm1).Heston_fft(alpha,n,B,K)[0]-FFT(parm2).Heston_fft(alpha,n,B,K)[0])/(2*ds)
    vega_heston = (FFT(parm3).Heston_fft(alpha,n,B,K)[0]-FFT(parm4).Heston_fft(alpha,n,B,K)[0])/(2*dv)
    d1 = (np.log(S0/K)+(r-q+vol**2/2)*expT)/(vol*(expT)**0.5)
    delta_bs = np.exp(-q*expT)*norm.cdf(d1)
    vega_bs = np.exp(-q*expT)*S0*np.sqrt(expT)*norm.cdf(d1)
    print(delta_bs)
    print(delta_heston)
    print(vega_heston)
    print(vega_bs)

    K = S0
    straddle_p = 2 * FFT(parm3).Heston_fft(alpha,n,B,K)[0] + K*np.exp(-r*expT)-(S0)*np.exp(-q*expT)
    straddle_m = 2 * FFT(parm4).Heston_fft(alpha,n,B,K)[0] + K*np.exp(-r*expT)-(S0)*np.exp(-q*expT)
    vega_heston_straddle = (straddle_p-straddle_m)/(2*dv)
    print(vega_heston_straddle)























