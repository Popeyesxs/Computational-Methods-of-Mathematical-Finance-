#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 16:42:58 2020

@author: popeye
"""
import numpy as np
import math
from scipy import interpolate
import cmath
import matplotlib.pyplot as plt

class FFT:
    def __init__(self,lst):
        self.sigma = lst[0]
        self.eta0 = lst[1]
        self.kappa = lst[2]
        self.rho = lst[3]
        self.theta = lst[4]
        self.S0 = lst[5]
        self.r = lst[6]
        self.q = lst[7]
        self.T = lst[8]
        
    def Heston_fft(self,alpha,n,B,K):
        """ Define a function that performs fft on Heston process
        """
        
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
        
        return price
    
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
    
    
    def EL(self,N,M):
        sigma = self.sigma
        eta0 = self.eta0
        kappa = self.kappa
        rho = self.rho
        theta = self.theta
        S0 = self.S0
        r = self.r
        T = self.T
        q = self.q
        dt = T/M
        S = []
        for i in range(N):
            st = [S0] 
            eta = [eta0]
            s = S0
            e = eta0
            for j in range(M):
                z1 = np.random.normal()
                X = np.random.normal()
                z2 = rho * z1 + ((1 - rho**2) ** 0.5) * X           
                e1 = e + kappa *(theta - e) * dt + sigma * (max(e, 0) * dt) ** 0.5 * z2
                s1 = s + (max(e, 0) ** 0.5) *s* (dt**0.5) *z1 + (r - q) * s * dt
                s = s1
                e = e1
                eta += [e]
                st += [s]
            S += [st]
        return np.array(S)
        

def cal_Euro(S, r, T, K, M, N):
        
    payoff = np.maximum(S[:,M-1]-K,0)
    price = np.exp(-r*T)*sum(payoff)/N
            
    return price 

def cal_UpAndOut(S, r, T, K1, K2):
    max_S = np.max(S,axis=1)
    indicator = np.where(max_S<K2, 1, 0)
    payoff = np.maximum(S[:,-1]-K1, 0)*indicator
    opt_price = np.mean(payoff)*np.exp(-r*T)
    return opt_price

def cal_UpAndOut_vr(S, r, T, K1, K2, euro):
    pay_euro = np.maximum(S[:, -1] - K1, 0)

    max_S = np.max(S, axis=1)
    indicator = np.where(max_S < K2, 1, 0)
    pay_UAO = np.maximum(S[:, -1] - K1, 0) * indicator

    cov_mat = np.cov(pay_euro, pay_UAO)
    c = - cov_mat[0][1]/cov_mat[0][0]

    payoff = np.mean(pay_UAO) + c * (pay_euro - euro)

    opt_price = np.mean(payoff)*np.exp(-r*T)
    return opt_price


if __name__ == '__main__':
    # (1)
    r = 0.015
    q = 0.0177
    S0 = 282

    kappa = 3.51
    theta = 0.052
    sigma = 1.17
    rho = -0.77
    eta0 = 0.034
    expT = 1
    K = 285
    alpha = 1
    
    n = 15
    B = 1000
    parm = [sigma,eta0,kappa,rho,theta,S0,r,q,expT]
    #2
    N = 10000
    M = 252
    call_sim = FFT(parm).EL(N,M)   
    #3
    price_sim = cal_Euro(call_sim, r, expT, K, M,N)
    call_heston = FFT(parm).Heston_fft(alpha,n,B,K)
    
    #4
    
    K1 = 285
    K2 = 315
    Nl = [100,200,500,1000,5000,100000,15000,20000]
    P = []
    for N in Nl:
        simu_S = FFT(parm).EL(N,252)  
        price = cal_UpAndOut(simu_S, r, expT, K1, K2)
        P += [price]
    print(P)
    
    K1 = 285
    K2 = 315
    Ns = np.logspace(1,4.5,10)
    UpAndOut=[]
    UpAndOut_cv = []
    for N in Ns:
        temp = 0
        temp_cv = 0
        for _ in range(10):
            simu_S = FFT(parm).EL(int(N),252)
            price = cal_UpAndOut(simu_S, r, expT, K1, K2)
            price_cv = cal_UpAndOut_vr(simu_S, r, expT, K1, K2, price_sim)

            temp += price
            temp_cv += price_cv
        UpAndOut.append(temp/10)
        UpAndOut_cv.append(temp_cv/10)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xscale("log")
    ax.set_title('Option Price vs simulated paths')
    ax.set_ylabel('Price of Up-And-Out')
    ax.set_xlabel('Number of simulated paths (N)')
    ax.plot(Ns, UpAndOut, Ns, UpAndOut_cv,"o-")
    plt.legend(['Up-And-Out without control variate', 'Up-And-Out with control variate'])

    plt.show()

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    