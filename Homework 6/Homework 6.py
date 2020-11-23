#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 14:14:07 2020

@author: Dezheng Xu

Description: Homework 6 Numerical PDEs

"""
import numpy as np
import matplotlib.pyplot as plt

#5
def A(Smax, Smin, ht, hs):
    s = np.arange(Smin, Smax+hs, hs)
    a = 1 - sigma ** 2 * s ** 2 * ht/hs ** 2 - r * ht
    l = sigma ** 2 * s ** 2 * ht / hs ** 2/2 - r * s * ht/hs/2
    u = sigma ** 2 * s ** 2 * ht/hs ** 2/2 + r * s * ht/hs/2
    A = np.diag(a[1:300])
    
    l1 = l[2:300]
    u1 = u[1:300-1]
    
    for i in range(len(l1)):
        A[i+1][i] = l1[i]
        A[i][i+1] = u1[i]
        
    eig_vals,eig_vecs = np.linalg.eig(A)
    print('A Matrix:\n',A)
    print('Eigvalues:\n',eig_vals)
    print('Absolute Eigvalues:\n',abs(eig_vals))
    
    plt.plot(eig_vals)
    plt.title('Eigenvalues')
    plt.ylabel('Eigenvalues')
    plt.show()
    
    plt.plot(abs(eig_vals))
    plt.title('Absolute Eigenvalues')
    plt.ylabel('Absolute Eigenvalues')
    plt.show()
    return A
#6 The call spread without the right of early exercise
def bs(S,K1,K2,T,r,sigma,Smin,Smax,ht, hs,option):
    s = np.arange(Smin, Smax+hs, hs)
    u = sigma ** 2 * s ** 2 * ht/hs ** 2/2 + r * s * ht/hs/2
    long  = s - K1 # Long Call
    long[long < 0] = 0
    short = s - K2 # Short Call
    short[short < 0] = 0
    c_e = long - short
    c = c_e[1:300]
        
    # Adding early exercise feature
    for i in range(10000):
        c = A.dot(c)
        c[0] = c[0] + u[1] * (K1-K2) * np.exp(-r*i*ht)
        if option == 'american':
            # early exercise
            c = [max(x,y) for x,y in zip(c,c_e[1:300])]
                            
    return np.interp(S, s[1:300], c)


if __name__ == '__main__':
    S = 312.86
    K1 = 315
    K2 = 320
    T = 142/252
    r = 0.72/100
    sigma = 0.28485
    Smin = 0
    Smax = 600
    ht = T/10000
    hs = 2
    A = A(Smax, Smin, ht, hs)
    callspread_euro = bs(S,K1,K2,T,r,sigma,Smin,Smax,ht, hs,'european')
    print(callspread_euro)
    callspread_amer = bs(S,K1,K2,T,r,sigma,Smin,Smax,ht, hs,'american')
    print(callspread_amer)
    premium = callspread_amer- callspread_euro
    print(premium)





