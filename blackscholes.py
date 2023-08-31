import numpy as np
from scipy.stats import norm
import pandas_datareader.data as web
import pandas as pd
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import norm

class BsOption:
    def __init__(self, S, K, T, r, sigma, q=0):
        self.S = S
        self.K = K
        self.T = T
        self.r = r 
        self.sigma = sigma
        self.q = q
        
    
    @staticmethod
    def N(x):
        return norm.cdf(x)
    
    @property
    def params(self):
        return {'S': self.S, 
                'K': self.K, 
                'T': self.T, 
                'r':self.r,
                'q':self.q,
                'sigma':self.sigma}
    
    def d1(self):
        return (np.log(self.S/self.K) + (self.r -self.q + self.sigma**2/2)*self.T) \
                                / (self.sigma*np.sqrt(self.T))
    
    def d2(self):
        return self.d1() - self.sigma*np.sqrt(self.T)
    
    def _call_value(self):
        return self.S*np.exp(-self.q*self.T)*self.N(self.d1()) - \
                    self.K*np.exp(-self.r*self.T) * self.N(self.d2())
                    
    def _put_value(self):
        return self.K*np.exp(-self.r*self.T) * self.N(-self.d2()) -\
                self.S*np.exp(-self.q*self.T)*self.N(-self.d1())
    
    def price(self, type_ = 'C'):
        if type_ == 'C':
            return self._call_value()
        if type_ == 'P':
            return self._put_value() 
        if type_ == 'B':
            return  {'call': self._call_value(), 'put': self._put_value()}
        else:
            raise ValueError('Unrecognized type')
            
        
        
        
        
if __name__ == '__main__':
    K = 100
    r = 0.1
    T = 1
    sigma = 0.3
    S = 100
    print(BsOption(S, K, T, r, sigma).price('B'))

# N = norm.cdf

# def BS_CALL(S, K, T, r, sigma):
#     d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
#     d2 = d1 - sigma * np.sqrt(T)
#     return S * N(d1) - K * np.exp(-r*T)* N(d2)

# def BS_PUT(S, K, T, r, sigma):
#     d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
#     d2 = d1 - sigma* np.sqrt(T)
#     return K*np.exp(-r*T)*N(-d2) - S*N(-d1)

# K = 100
# r = 0.1
# T = 1
# Sigmas = np.arange(0.1, 1.5, 0.01)
# S = 100

# calls = [BS_CALL(S, K, T, r, sig) for sig in Sigmas]
# puts = [BS_PUT(S, K, T, r, sig) for sig in Sigmas]
# plt.plot(Sigmas, calls, label='Call Value')
# plt.plot(Sigmas, puts, label='Put Value')
# plt.xlabel('$\sigma$')
# plt.ylabel(' Value')
# plt.legend()
# plt.show()

# K = 100
# r = 0.05
# T = np.arange(0, 2, 0.01)
# sigma = 0.3
# S = 100

# calls = [BS_CALL(S, K, t, r, sigma) for t in T]
# puts = [BS_PUT(S, K, t, r, sigma) for t in T]
# plt.plot(T, calls, label='Call Value')
# plt.plot(T, puts, label='Put Value')
# plt.xlabel('$T$ in years')
# plt.ylabel(' Value')
# plt.legend()
# plt.show()

# start = dt.datetime(2010,1,1)    
# end =dt.datetime(2020,10,1) 
# symbol = 'AAPL' ###using Apple as an example
# source = 'yahoo'
# data = web.DataReader(symbol, source, start, end)
# data['change'] = data['Adj Close'].pct_change()
# data['rolling_sigma'] = data['change'].rolling(20).std() * np.sqrt(255)


# data.rolling_sigma.plot()
# plt.ylabel('$\sigma$')
# plt.title('AAPL Rolling Volatility')

# std = data.change.std()
# WT = np.random.normal(data.change.mean() ,std, size=Ndraws)
# plt.hist(np.exp(WT)-1,bins=300,color='red',alpha=0.4);
# plt.hist(data.change,bins=200,color='black', alpha=0.4);
# plt.xlim([-0.03,0.03])
# plt.show()


# fig, ax = plt.subplots()
# ax = sns.kdeplot(data=data['change'].dropna(), label='Empirical', ax=ax,shade=True)
# ax = sns.kdeplot(data=WT, label='Log Normal', ax=ax,shade=True)
# plt.xlim([-0.15,0.15])
# plt.ylim([-1,40])
# plt.xlabel('return')
# plt.ylabel('Density')
# plt.show()

# def BS_CALLDIV(S, K, T, r, q, sigma):
#     d1 = (np.log(S/K) + (r - q + sigma**2/2)*T) / (sigma*np.sqrt(T))
#     d2 = d1 - sigma* np.sqrt(T)
#     return S*np.exp(-q*T) * N(d1) - K * np.exp(-r*T)* N(d2)

# def BS_PUTDIV(S, K, T, r, q, sigma):
#     d1 = (np.log(S/K) + (r - q + sigma**2/2)*T) / (sigma*np.sqrt(T))
#     d2 = d1 - sigma* np.sqrt(T)
#     return K*np.exp(-r*T)*N(-d2) - S*np.exp(-q*T)*N(-d1)