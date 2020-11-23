#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 16:11:36 2020

@author: Dezheng Xu
Description: MF796 Homework
"""
from scipy.stats import norm
import numpy as np
import math 

#function 1
class MCStockSimulator:
    
    """encapsulates the data and methods required to simulate stock returns and values
    """
    
    def __init__(self, s, t, r, sigma, nper_per_year, beta):
        """s (the current stock price in dollars),
           t (the option maturity time in years),
           r (the annualized rate of return on this stock),
           sigma (the annualized standard deviation of returns),
           beta is positive constant,
           nper_per_year (the number of discrete time periods per year)
        """
        self.s = s
        self.t = t
        self.r = r
        self.sigma = sigma
        self.nper_per_year = nper_per_year
        self.beta = beta
        
    def __repr__(self):
        
        """ print out an object 
        """
        s = "StockSimulator (s = $%.2f, t = %.2f (years), r = %.2f, sigma = %.2f, beta = %.1f, nper_per_year = %d )" %(self.s, self.t, self.r, self.sigma, self.beta, self.nper_per_year)
        return s
    
    def generate_simulated_stock_values(self):
        """generate and return a np.array (numpy array) containing a sequence of stock values, 
           corresponding to a random sequence of stock return
        """
        
        sr =  np.zeros(int(self.t*self.nper_per_year)+1)
        sr[0] = self.s
        for i in range(1, int(self.nper_per_year * self.t)+1):
            
            sr[i] = sr[i-1]+ np.random.normal() * self.sigma * sr[i-1]**self.beta
        
        return np.array(sr)

#function 2 
class MCStockOption(MCStockSimulator):
    
    """ encapsulate the idea of a Monte Carlo stock option, and will contain some 
        additional data members( that are not part of class MCStockSimulator and
        are required to run stock-price simulations and calculate the option’s payoff.
        However, this class will not be a concrete option type, and thus will not
        implement the option’s value algorithm, which differs for each concrete option type.
    """
    
    def __init__(self, s, x, t, r, sigma, nper_per_year, num_trials, beta):
        
        """ s, which is the initial stock price
            x, which is the option’s exercise price
            t, which is the time to maturity (in years) for the option
            r, which is the (expected) mean annual rate of return on the underlying stock
            sigma, which is the annual standard deviation of returns on the underlying stock
            nper_per_year, which is the number of discrete time periods per year with which to evaluate the option, and
            num_trials, which is the number of trials to run when calculating the value of this option
            beta is positive constant,
        """
        super().__init__(s, t, r, sigma, nper_per_year, beta)
        
        self.x = x
        self.num_trials = num_trials
        
    def __repr__ (self):
        """create a nicely formatted printout of the MCStockOption object
        """
        s = "MCStockOption, s=%.2f, x=%.2f, t=%.2f, r=%.2f, sigma=%.2f, beta = %.1f, nper_per_year=%d, num_trials=%d" %(self.s, self.x, self.t, self.r, self.sigma, self.beta, self.nper_per_year, self.num_trials)
        return s

    def value(self):
        """return the value of the option
        """
        print("Base class MCStockOption has no concrete implementation of .value()." )
        return 0

#function 3
       
class MCEuroCallOption(MCStockOption):
    
    """a class definition for the class MCEuroCallOption which inherits from the base class MCStockOption.
    """
    
    def __init__(self, s, x, t, r, sigma, nper_per_year, num_trials, beta):
        
        """initialize a MCEuroCallOption object of this class
        """
        
        super().__init__(s, x, t, r, sigma, nper_per_year, num_trials, beta) 
    
    def __repr__ (self):
        
        """create a nicely formatted printout of the MCEuroCallOption object
        """
        s = "MCEuroCallOption, s=%.2f, x=%.2f, t=%.2f, r=%.2f, sigma=%.2f, beta = %.1f, nper_per_year=%d, num_trials=%d" %(self.s, self.x, self.t, self.r, self.sigma, self.beta, self.nper_per_year, self.num_trials)
        
        return s
    
    def value(self):
        
        """the value of the European put option is calculated
        """
       
        
        trials = np.array([max(self.generate_simulated_stock_values()[-1] - self.x, 0)  * math.exp(- self.r * self.t) for i in range(self.num_trials)] )
        self.stdev = np.std(trials)
        return np.mean(trials)
    
#function 4
    
class BSOption:
    def  __init__(self, s, x, t, sigma, rf, div):
        """initialize a new BSOption object of this class
           s (the current stock price in dollars),
           x (the option strike price),
           t (the option maturity time in years),
           sigma (the annualized standard deviation of returns),
           rf (the annualized risk free rate of return),
           div (the annualized dividend rate; assume continuous dividends rate),
        """
        self.s = s
        self.x = x
        self.t = t
        self.sigma = sigma
        self.rf = rf
        self.div = div
        
    def __repr__(self):
        """return a beautifully-formatted string representation of the BSOption object
        """
        s = "s = $%.2f, x = $%.2f, t = %.2f (years), sigma = %.3f, rf = %.3f, div = %.2f" %(self.s, self.x, self.t, self.sigma, self.rf, self.div)
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
    
    def value(self):
        
        """calculate value for base class BSOption
        """
        
        print("Cannot calculate value for base class BSOption." )
        return 0
    
    def delta(self):
        
        """calculate delta for base class BSOption
        """
        
        print("Cannot calculate delta for base class BSOption." )
        return 0
        
    
#function 5
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
        f2 = nd2 * self.x * math.e ** (-self.rf * self.t)
        return f1 - f2
    
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
    

class MCpayoff(MCStockSimulator):
    """estimate the payoff of the delta neutral portfolio
    """
    def __init__(self, s, t, r, sigma, nper_per_year, beta):
        """s (the current stock price in dollars),
           t (the option maturity time in years),
           r (the annualized rate of return on this stock),
           sigma (the annualized standard deviation of returns),
           beta is positive constant,
           nper_per_year (the number of discrete time periods per year)
        """
        super().__init__(s, t, r, sigma, nper_per_year, beta)
    
    def payoff(self):
        
        """estimate the payoff of the delta neutral portfolio
        """
        payoff = [max(self.generate_simulated_stock_values()[-1] - self.s, 0) - (self.generate_simulated_stock_values()[-1] - self.s) * delta for i in range(10000)]
        return np.mean(payoff)
        
if __name__ == '__main__':
    MCcall = MCEuroCallOption(100, 100, 1, 0.0, 0.25, 1, 1000000, 1)
    BScall = BSEuroCallOption(100, 100, 1, 0.25, 0, 0)
    delta = BScall.delta()
    MCpayoff(100, 1, 0, 0.25, 1,1).payoff()
    MCpayoff(100, 1, 0, 0.25, 1,0.5).payoff()
    MCpayoff(100, 1, 0, 0.4, 1,1).payoff()
        
    
    
    
    



    


