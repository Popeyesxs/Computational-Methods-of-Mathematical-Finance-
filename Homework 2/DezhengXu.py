#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 15:38:08 2020

@author: Dezheng Xu

Description: MF796 Homework2
"""
from scipy.stats import norm
import numpy as np
import math 
import matplotlib.pyplot as plt
import pandas as pd
#Problem 1: Evaluation of a known integral using various quadrature

#(1)

class BSEuroCallOption:
    
    def __init__(self, s, k, t, r, sigma):
        
        """s (the current stock price in dollars),
           k (the option strike price),
           t (the option maturity time in years),
           sigma (the annualized standard deviation of returns),
           rf (the annualized rate of return),
        """
        self.s = s
        self.k = k
        self.t = t
        self.r = r
        self.sigma = sigma
    
    def __repr__(self):
        """return a beautifully-formatted string representation of the BSOption object
        """
        s = "s = $%.2f, k = $%.2f, t = %.2f (years), sigma = %.3f, r = %.3f" %(self.s, self.k, self.t, self.sigma, self.r)
        return s

    def d1(self):
        """the class BSOption to calculate d1
        """
        f = (self.r + (self.sigma ** (2)) / 2 ) * self.t
        return (1/(self.sigma * (self.t ** (0.5)))) *(math.log(self.s/self.k) + f)
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
    def mean(self):
        """ the expectation of lnS
        """
        return np.log(self.s) + (self.r - 0.5 * self.sigma ** 2) * self.t
    
    def std(self):
        """ the standard deviation of lnS
        """
        return (self.t * self.sigma ** 2) ** 0.5
    
#Use the evaluating intergrals for the log-normal density in Black-Scholes
    
    def integration(self, x):
        """use the log-normal density in Black-Scholes model get p
        """
        m = self.mean()
        sd = self.std()
        f = np.exp(-self.r*self.t) * (np.exp(x)-self.k)
        p = 1 / ( sd * ( 2 * np.pi) ** 0.5 ) * np.exp(-1 * (x-m) **2 / (2 * sd ** 2))
        return f * p
        
    
    def value(self):
        """implements the pricing algorithm for a European-style call option.
        """
        nd1 = self.nd1()
        nd2 = self.nd2()
        f1 = nd1 * self.s 
        f2 = nd2 * self.k * math.e ** (-self.r * self.t)
        return f1 - f2

#a	Left Riemann rule
    def left_riemann(self,n):
        """ b is upper bound
            a is lower bound
            by Left Riemann rule get x
        """
        b=self.mean()+4*self.std()
        a=np.log(self.k)
        x=np.array([a+(b-a)/n * i for i in range(n)]) # range(n) start from 0 without the last one
        f1=self.integration(x)
        result=(b-a)/n*f1
        return np.sum(result)

#b Midpoint rule
    def midpoint_rule(self, n):
        """ b is upper bound
            a is lower bound
            by midpoint rul get x
        """
        b=self.mean()+4*self.std()
        a=np.log(self.k)
        x=np.array([a+(b-a)/n * (i + 1/2) for i in range(n)])
        f2=self.integration(x)
        result=(b-a)/n*f2
        return np.sum(result)
        
#c Gauss Lengendre
    def gauss_legendre(self,n):
        """ b is upper bound
            a is lower bound
            by Gauss lengendre to get intergration
        """
        x,p=np.polynomial.legendre.leggauss(n)
        b=self.mean()+4*self.std()
        a=np.log(self.k)
        result = ((b - a) / 2) * p * self.integration(((b-a)/2) * x +((b+a)/2)) 
        return np.sum(result)
        
# error by compare the results from Black-Scholes with these three methods.
    def error(self):
        """ from the question we estimated the intergration use n = 5, 10, 50, 100
            the error can be cacualte by range(1, 101)
        """
        BS = self.value()
        df1 = pd.DataFrame(np.random.random(size = (100, 9)), columns = ['error1', 'error2', 'error3', 'N','N ^ -1', 'N ^ -2', 'N ^ -3', 'N ^ -N', 'N ^ -2N' ])
        error1 = []
        error2 = []
        error3 = []
        for i in range(1,101):
            error1 += [abs(self.left_riemann(i)-BS)]
            error2 += [abs(self.midpoint_rule(i)-BS)]
            error3 += [abs(self.gauss_legendre(i)-BS)]
        df1['error1'] = np.array(error1)
        df1['error2'] = np.array(error2)
        df1['error3'] = np.array(error3)
        df1['N'] = np.array([n for n in range(1, 101)])
        df1['N ^ -1'] = 1 / df1['N']
        df1['N ^ -2'] = 1 / df1['N'] ** 2
        df1['N ^ -3'] = 1 / df1['N'] ** 3
        df1['N ^ -N'] = 1 / df1['N'] ** df1['N']
        df1['N ^ -2N'] = 1 / df1['N'] ** (2 * df1['N'])
        return df1

#problem 2  Caculation of Contingent Options
#（1）vanilla option       
class VanillaCallOption:
    
    def __init__(self, s, k, t, r, sigma):
        
        """s (the current stock price in dollars),
           k (the option strike price),
           t (the option maturity time in years),
           sigma (the annualized standard deviation of returns),
           rf (the annualized rate of return),
        """
        self.s = s
        self.k = k
        self.t = t
        self.r = r
        self.sigma = sigma
    
#Use the evaluating intergrals for the log-normal density in Black-Scholes
    
    def integration(self, x):
        """use the log-normal density in Black-Scholes model get p
        """
        f = np.exp(-self.r*self.t) * (x-self.k)
        p = 1 / ( self.sigma * ( 2 * np.pi) ** 0.5 ) * np.exp(-1 * (x-self.s) **2 / (2 * self.sigma ** 2))
        return f * p
        
    
    def midpoint_rule(self, n):
        """ b is upper bound
            a is lower bound
            by midpoint rul get x
        """
        b=self.s+4*self.sigma
        a=self.k
        x=np.array([a+(b-a)/n * (i + 1/2) for i in range(n)])
        f2=self.integration(x)
        result=(b-a)/n*f2
        return np.sum(result)

#(2) contingent option
class ContingentCallOption:
    
    def __init__(self, s, k, t1, t2, r, sigma1, sigma2, rho, con):
        
        """s (the current stock price in dollars),
           k (the option strike price),
           t1 (the option maturity time in 1 year),
           t2 (the option maturity time in 6 months),
           sigma1 (the annualized standard deviation of returns),
           sigma1 (the 6-months standard deviation of returns),
           rho (correlation between S1 and S2),
           r (the annualized rate of return),
           con (contingent on SPY at 6 months being below 365)
        """
        self.s = s
        self.k = k
        self.t1 = t1
        self.t2 = t2
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.rho = rho
        self.r = r
        self.con = con
#Use the evaluating intergrals for the log-normal density in Black-Scholes
    
    def integration(self, x, y):
        
        f1 = np.exp(-self.r*self.t1)
        f2 = x-self.k
        f3 = 1/(2 * np.pi * self.sigma1 * self.sigma2 * (1 - self.rho ** 2) ** 0.5)
        f4 = -1/(2 * (1 - self.rho ** 2)) * ((x - self.s) ** 2 / self.sigma1 ** 2 - 2 * self.rho*(x-self.s)*(y-self.s)/(self.sigma1*self.sigma2)+(y-self.s)**2/self.sigma2**2)
        return f1 * f2 * f3 * np.exp(f4)
    

    def midpoint_rule(self,n):
        b2 = self.con
        a2 = self.s - 4 * self.sigma2
        b1 = self.s + 4 * self.sigma1
        a1 = self.k
        s1 = np.array([a1 + (b1 - a1)/n*(i+1/2) for i in range(n)])
        s2 = np.array([a2 + (b2 - a2)/n*(i+1/2) for i in range(n)])
        result = 0
        for i in s2:
            p = self.integration(s1 , i)
            result += (b2 - a2 ) / n * (np.sum(( b1 - a1)/ n * p))
        return result
if __name__=='__main__': 
    
    a = BSEuroCallOption(10,12,1/4,0.04,0.2)
    n = 100 
    df = a.error()
    df2 = pd.DataFrame([df.loc[i] for i in [4, 9, 49 , 99]])
    df3 = pd.DataFrame([df['error1'], df['N ^ -1'], df['N ^ -2'], df['N ^ -3']]).T
    df4 = pd.DataFrame([df['error2'], df['N ^ -1'], df['N ^ -2'], df['N ^ -3']]).T
    df5 = pd.DataFrame([df['error3'], df['N ^ -N'], df['N ^ -2N']]).T
    df3[2:20].plot()
    df4[2:20].plot()
    df5[2:20].plot()
    #a = ContingentCallOption(321.73,370,1,0.5,0,20,15,0.95,365)
    #a1 = ContingentCallOption(321.73,370,1,0.5,0,20,15,0.8,365)
    #a2 = ContingentCallOption(321.73,370,1,0.5,0,20,15,0.5,365)
    #a3 = ContingentCallOption(321.73,370,1,0.5,0,20,15,0.2,365)
    #a4 = ContingentCallOption(321.73,370,1,0.5,0,20,15,0.95,360)
    #a5 = ContingentCallOption(321.73,370,1,0.5,0,20,15,0.95,350)
    #a6 = ContingentCallOption(321.73,370,1,0.5,0,20,15,0.95,340)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

