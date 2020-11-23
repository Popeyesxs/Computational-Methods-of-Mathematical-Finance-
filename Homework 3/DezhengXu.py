    #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 10:05:38 2020

@author: Dezheng Xu

Descrption: Homework 3

"""
# 1 Option Pricing via FFT Techniques
import time
import math
import numpy as np
import pandas as pd
import cmath
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import norm
from scipy.optimize import root
from scipy import interpolate
#Problem 1: Evaluation of a known integral using various quadrature

#(1)

class BSOption:
    
    def __init__(self, s, x, t, sigma, rf, div):
        
        """s (the current stock price in dollars),
           k (the option strike price),
           t (the option maturity time in years),
           sigma (the annualized standard deviation of returns),
           rf (the risk-free rate),
           div(the asset pays dividends)
        """
        self.s = s
        self.x = x
        self.t = t
        self.rf = rf
        self.sigma = sigma
        self.div = div
    
    def __repr__(self):
        """return a beautifully-formatted string representation of the BSOption object
        """
        s = "s = $%.2f, x = $%.2f, t = %.2f (years), sigma = %.3f, rf = %.3f, div = %.3f" %(self.s, self.x, self.t, self.sigma, self.rf, self.div)
        return s

    def d1(self):
        """the class BSOption to calculate d1
        """
        f = (self.rf - self.div + (self.sigma ** (2)) / 2 ) * self.t
        return (1/(self.sigma * (self.t ** (0.5)))) *(math.log(self.s/self.x) + f)
    def d2(self):
        """the class BSOption to calculate d2
        """
        d1 = self.d1()
        return  d1 - self.sigma * (self.t **(0.5))
    def nd1(self):
        """the class BSOption to calculate N(d1)
        """
        d1 = self.d1()
        return norm.cdf(d1)
    def nd2(self):
        """the class BSOption to calculate N(d2)
        """
        d2 = self.d2()
        return norm.cdf(d2)

class BSEuroCallOption(BSOption):
    def __init__(self, s, x, t, sigma, rf, div):
        """initialize a new BSEuroCallOption object of this class
        """
        
        
        super().__init__(s, x, t, sigma, rf, div)
    def value(self):
        """implements the pricing algorithm for a European-style call option.
        """
        nd1 = super().nd1()
        nd2 = super().nd2()
        f1 = nd1 * self.s * math.e ** (-self.div * self.t)
        f1 -= nd2 * self.x * math.e ** (-self.rf * self.t)
        return f1
    
    def __repr__(self):
        """return a beautifully-formatted string representation of the BSEuroCallOption object
        """
        s = "BSEuroCallOption, value = $%.2f, \n" %(self.value())
        s += "parameters = (s = $%.2f, x = $%.2f, t = %.2f (years), sigma = %.3f, rf = %.3f, div = %.2f) " %(self.s, self.x, self.t, self.sigma, self.rf, self.div)
        return s

    def delta(self):
        """create a hedging portfolio to offset the option’s price risk
        """
        return super().nd1()

#(a) Exploring FFT Technique Parameters

class FFT:
    def __init__(self, sigma, eta, kappa, rho, theta, S0, r, T):
        
        self.sigma = sigma
        self.eta = eta
        self.kappa = kappa
        self.rho = rho
        self.theta = theta
        self.S0 = S0
        self.r = r
        self.T = T

# Make sure the delta is 0 or 1     
    def p(self,n):
        """ indictor function equal to 1 if j =1 and 0 otherwise
        """
        p = np.zeros(len(n),dtype = complex)
        p[n==0] = 1
        return p

#Estimated Intergration     
    def Intergration (self,u):
        
        sigma = self.sigma
        eta = self.eta
        kappa = self.kappa
        rho = self.rho
        theta = self.theta
        S0 = self.S0
        r = self.r
        T = self.T
        
        ii = complex(0,1)
        
        l = cmath.sqrt( sigma ** 2 * ( u ** 2 + ii * u ) + ( kappa - ii * rho * sigma * u) ** 2)
        w = np.exp(ii * u * np.log(S0) + ii * u * (r - 0) * T + kappa * theta * T * (kappa - ii * rho * sigma * u)/ sigma ** 2)/(cmath.cosh(l * T/2) + (kappa - ii * rho * sigma * u) / l * cmath.sinh(l * T/2)) ** (2 * kappa * theta/sigma**2)
        E = w * np.exp(-(u ** 2 + ii * u) * eta/( l / cmath.tanh( l * T/2) + kappa - ii * rho * sigma * u))
        
        return E

# Function of FFT     
    def x(self,alpha,n,B):
        
        N = 2**n
        v = B / N
        vk = 2 * math.pi / N
        k = vk / v
        
        J = np.arange(1,N+1,dtype = complex)
        vj = (J-1) * v
        
        
        ii = complex(0,1)
        
        Psi_vj = np.zeros(len(J),dtype = complex)
        
        for zz in range(0,N):
            u = vj[zz] - (alpha + 1) * ii
            numer = self.Intergration(u)   
            denom = (alpha + vj[zz] * ii) * (alpha + 1 + vj[zz] * ii)
            
            Psi_vj [zz] = numer / denom
            
        
        xj = (v / 2) * Psi_vj * np.exp(-ii * (np.log(self.S0) - k * N / 2) * vj) * (2 - self.p(J-1))
        y = np.fft.fft(xj)
        return y 

# Use y to get opton price
    def price(self, K, B, n, alpha):
        
        bt = time.time()
        r = self.r
        T = self.T
        S0 = self.S0
        N = 2**n
        v = B / N
        vk = 2 * math.pi / N
        k = vk / v
       
        m = np.arange(1,N+1,dtype = complex)
        Beta = np.log(S0) - k * N / 2
        km = Beta + (m-1) * k
        Mul = np.exp(-alpha * np.array(km)) / np.pi
        yj = Mul * np.array(self.x(alpha, n, B)).real
        k_List = list(Beta + (np.cumsum(np.ones((N, 1))) - 1) * k)
        Kt = np.exp(np.array(k_List))
       
        Kz = []
        Z = []
        for i in range(len(Kt)):
            if( Kt[i]>1e-16 )&(Kt[i] < 1e16)& ( Kt[i] != float("inf"))&( Kt[i] != float("-inf")) &( yj[i] != float("inf"))&(yj[i] != float("-inf")) & (yj[i] is not  float("nan")):
                Kz += [Kt[i]]
                Z += [yj[i]]
        tck = interpolate.splrep(Kz , np.real(Z))
        price =  np.exp(-r*T)*interpolate.splev(K, tck).real
        et = time.time()
        rt = et - bt

        return(price,rt)

# In order to find alpha I wanna choice the lowest alpha until the most stable price generate
    
    def table(self,K, B, n, alphas):
        p = np.array([self.price(K, B, n, a)[0] for a in alphas])
        df1 = pd.DataFrame(p, index = alphas, columns = ['price'])
        df1.plot()
        return df1

#ii Comment on what values seem to lead to the most accurate prices, and the efficiency of each parameterization        
    def figure1(self,nl,Bl,K):
        z = np.zeros((len(nl),len(Bl)))
        e = np.zeros((len(nl),len(Bl)))
        M = []
        x, y= np.meshgrid(nl, Bl)
        
        for i in range(len(nl)):
            for j in range(len(Bl)):
                m1 = self.price(K, Bl[j], nl[i], 1)
                z[i][j] = m1[0]
                e[i][j] = 1/((m1[0]-21.27)**2*m1[1])
                M += [(e[i][j],nl[i],Bl[j])]
        print(max(M))       
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.plot_surface(x, y, z.T, rstride=1, cstride=1, cmap='rainbow')
     
        plt.show()
        
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.plot_surface(x, y, e.T, rstride=1, cstride=1, cmap='rainbow')
        plt.show()
        
        

#(b)Exploring Heston Parameters

def plot_vol_K(price_list,K_list):
    vol = []
    
    for i in range(len(K_list)):
        result = root(lambda x: BSEuroCallOption(150, K_list[i], 0.25, x, 0.025, 0.00).value()-price_list[i],0.3)
        vol += [result.x]
        
    vol = np.array(vol)
    plt.plot(K_list,vol)
    plt.title("Fig. Implied Volatility vs Strike K")
    plt.xlabel("Strike K")
    plt.ylabel("Implied Volatility")
    plt.show()
    
def plot_vol_T(price_list,t_list):
    vol = []
    
    for i in range(len(t_list)):
        result = root(lambda x: BSEuroCallOption(150, 150, t_list[i], x, 0.025, 0.00).value()-price_list[i],0.3)
        vol += [result.x]
        
    vol = np.array(vol)
    plt.plot(t_list,vol)
    plt.title("Fig. Implied Volatility vs Expiry T")
    plt.xlabel("Expiry T")
    plt.ylabel("Implied Volatility")
    plt.show()
        



        
if __name__ == '__main__':
    # a
    #alpha = 1.5
    #sigma = 0.2
    #eta = 0.08
    #kappa = 0.7
    #rho = -0.4
    #theta = 0.1
    #S0 = 250
    #K = 260
    #r = 0.02
    #expT = 0.5
    
    #n = 10
    #B = 250 * 1.5
    #a = FFT(sigma,eta,kappa,rho,theta,S0,r,expT)
    #a.price(K, B, n, alpha)     
    
    
    #alphas = np.linspace(0.1,5,num = 50)
    #a.table()
    #ns = np.array([7,8,9,10,11,12,13,14])
    #Bs = np.linspace(250*1.3,250*1.5,100)
    #a.figure1(ns,Bs,K)
    # b 
    alpha = 1.5
    sigma = 0.4
    eta = 0.09
    kappa = 0.5
    rho = 0.25
    theta = 0.12
    S0 = 150
    r = 0.025
    expT = 0.25
    
    n = 9
    B = 150*2.7
    b = FFT(sigma,eta,kappa,rho,theta,S0,r,expT)
    
    #plot volatility as a function of strike
   

    
    K_list = np.linspace(80,230,60)
    b_price = []
    for j in K_list:
        b_price += [b.price(j, B, n, alpha)[0]]
    b_price = np.array(b_price)
    
    plot_vol_K(b_price,K_list)
    
    #ii
    alpha = 1.5
    sigma = 0.4
    eta = 0.09
    kappa = 0.5
    rho = 0.25
    theta = 0.12
    S0 = 150
    r = 0.025
    expT = 0.25
    
    n = 9
    B = 150*2.7
    
    t_list = np.linspace(1/12,2,50)
    c_price = []
    for t in t_list:
        temp = FFT(sigma,eta, kappa,rho,theta,S0,r,t)
        c_price += [temp.price(150, B, n ,alpha)[0]]
    c_price = np.array(c_price)
    print(c_price)
    plot_vol_T(c_price,t_list)
    
    #iii
    #change eta to 0.2
    alpha = 1.5
    sigma = 0.4
    eta = 0.2
    kappa = 0.5
    rho = 0.25
    theta = 0.12
    S0 = 150
    r = 0.025
    expT = 0.25
    
    n = 9
    B = 150*2.7
    
    t_list = np.linspace(1/12,2,50)
    c_price = []
    for t in t_list:
        temp = FFT(sigma,eta, kappa,rho,theta,S0,r,t)
        c_price += [temp.price(150, B, n ,alpha)[0]]
    c_price = np.array(c_price)
    print(c_price)
    plot_vol_T(c_price,t_list)
    
    
    alpha = 1.5
    sigma = 0.4
    eta = 0.2
    kappa = 0.5
    rho = 0.25
    theta = 0.12
    S0 = 150
    r = 0.025
    expT = 0.25
    n = 9
    B = 150*2.7
    b = FFT(sigma,eta,kappa,rho,theta,S0,r,expT)
    K_list = np.linspace(80,230,60)
    b_price = []
    for j in K_list:
        b_price += [b.price(j, B, n, alpha)[0]]
    b_price = np.array(b_price)
    
    plot_vol_K(b_price,K_list)
    
    #change kappa to 0.2
    alpha = 1.5
    sigma = 0.4
    eta = 0.09
    kappa = 0.2
    rho = 0.25
    theta = 0.12
    S0 = 150
    r = 0.025
    expT = 0.25
    
    n = 9
    B = 150*2.7
    
    t_list = np.linspace(1/12,2,50)
    c_price = []
    for t in t_list:
        temp = FFT(sigma,eta, kappa,rho,theta,S0,r,t)
        c_price += [temp.price(150, B, n ,alpha)[0]]
    c_price = np.array(c_price)
    print(c_price)
    plot_vol_T(c_price,t_list)
    
    
    alpha = 1.5
    sigma = 0.4
    eta = 0.09
    kappa = 0.2
    rho = 0.25
    theta = 0.12
    S0 = 150
    r = 0.025
    expT = 0.25
    n = 9
    B = 150*2.7
    b = FFT(sigma,eta,kappa,rho,theta,S0,r,expT)
    K_list = np.linspace(80,230,60)
    b_price = []
    for j in K_list:
        b_price += [b.price(j, B, n, alpha)[0]]
    b_price = np.array(b_price)
    
    plot_vol_K(b_price,K_list)
    
    #change rho to -0.2
    alpha = 1.5
    sigma = 0.4
    eta = 0.09
    kappa = 0.5
    rho = -0.2
    theta = 0.12
    S0 = 150
    r = 0.025
    expT = 0.25
    
    n = 9
    B = 150*2.7
    
    t_list = np.linspace(1/12,2,50)
    c_price = []
    for t in t_list:
        temp = FFT(sigma,eta, kappa,rho,theta,S0,r,t)
        c_price += [temp.price(150, B, n ,alpha)[0]]
    c_price = np.array(c_price)
    print(c_price)
    plot_vol_T(c_price,t_list)
    
    
    alpha = 1.5
    sigma = 0.4
    eta = 0.09
    kappa = 0.5
    rho = -0.2
    theta = 0.12
    S0 = 150
    r = 0.025
    expT = 0.25
    n = 9
    B = 150*2.7
    b = FFT(sigma,eta,kappa,rho,theta,S0,r,expT)
    K_list = np.linspace(80,230,60)
    b_price = []
    for j in K_list:
        b_price += [b.price(j, B, n, alpha)[0]]
    b_price = np.array(b_price)
    
    plot_vol_K(b_price,K_list)
    
    #change theta to 0.2
    alpha = 1.5
    sigma = 0.4
    eta = 0.09
    kappa = 0.5
    rho = 0.25
    theta = 0.2
    S0 = 150
    r = 0.025
    expT = 0.25
    
    n = 9
    B = 150*2.7
    
    t_list = np.linspace(1/12,2,50)
    c_price = []
    for t in t_list:
        temp = FFT(sigma,eta, kappa,rho,theta,S0,r,t)
        c_price += [temp.price(150, B, n ,alpha)[0]]
    c_price = np.array(c_price)
    print(c_price)
    plot_vol_T(c_price,t_list)
    
    
    alpha = 1.5
    sigma = 0.4
    eta = 0.09
    kappa = 0.5
    rho = 0.2
    theta = 0.2
    S0 = 150
    r = 0.025
    expT = 0.25
    n = 9
    B = 150*2.7
    b = FFT(sigma,eta,kappa,rho,theta,S0,r,expT)
    K_list = np.linspace(80,230,60)
    b_price = []
    for j in K_list:
        b_price += [b.price(j, B, n, alpha)[0]]
    b_price = np.array(b_price)
    
    plot_vol_K(b_price,K_list)


    
  

    
    