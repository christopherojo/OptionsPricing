import numpy as np
import math

############params################
N = 4
S0  = 100
T = 0.5
sigma = 0.4
dt = T/N
K =105
r = 0.05
u = np.exp( sigma * np.sqrt(dt) )
d =  np.exp( -sigma * np.sqrt(dt) )
p = ( np.exp(r*dt) - d) / (u -d)


######showing terminal stock prices for 4 step model################

for k in reversed(range(N+1)):
    ST = S0 * u**k * d ** (N-k)
    print(round(ST,2), round(max(ST-K,0),2))


#176.07 71.07
#132.69 27.69
#100.0 0
#75.36 0
#56.8 0


############showing node probabilities
def combos(n, i):
    return math.factorial(n) / (math.factorial(n-i)*math.factorial(i))
    
for k in reversed(range(N+1)):
    p_star = combos(N, k)*p**k *(1-p)**(N-k)
    print(round(p_star,2))
    
#0.06
#0.24
#0.37
#0.26
#0.07


######valuing the call from example#######################

C=0   
for k in reversed(range(N+1)):
    p_star = combos(N, k)*p**k *(1-p)**(N-k)
    ST = S0 * u**k * d ** (N-k)
    C += max(ST-K,0)*p_star
    
print(np.exp(-r*T)*C)

#10.60594883990603

N =4
S0  = 100
T = 0.5
sigma = 0.4
K = 105
r = 0.05

def binom_EU1(S0, K , T, r, sigma, N, type_ = 'call'):
    dt = T/N
    u = np.exp(sigma * np.sqrt(dt))
    d = np.exp(-sigma * np.sqrt(dt))
    p = (  np.exp(r*dt) - d )  /  (  u - d )
    value = 0 
    for i in range(N+1):
        node_prob = combos(N, i)*p**i*(1-p)**(N-i)
        ST = S0*(u)**i*(d)**(N-i)
        if type_ == 'call':
            value += max(ST-K,0) * node_prob
        elif type_ == 'put':
            value += max(K-ST, 0)*node_prob
        else:
            raise ValueError("type_ must be 'call' or 'put'" )
    
    return value*np.exp(-r*T)


binom_EU1(S0, K, T, r,sigma, N)

Ns = [2, 4, 6, 8, 10, 20, 50, 100, 200, 300, 400,500, 600]
    

for n in Ns:
    c = binom_EU1(S0, K, T, r,sigma, n)
    print(f'Price is {n} steps is {round(c,2)}')

import pandas as pd
import pandas_datareader.data as web
import numpy as np
import datetime as dt
import math

import matplotlib.pyplot as plt


def combos(n, i):
    return math.factorial(n) / (math.factorial(n-i)*math.factorial(i))



def binom_EU1(S0, K , T, r, sigma, N, type_ = 'call'):
    dt = T/N
    u = np.exp(sigma * np.sqrt(dt))
    d = np.exp(-sigma * np.sqrt(dt))
    p = (  np.exp(r*dt) - d )  /  (  u - d )
    value = 0 
    for i in range(N+1):
        node_prob = combos(N, i)*p**i*(1-p)**(N-i)
        ST = S0*(u)**i*(d)**(N-i)
        if type_ == 'call':
            value += max(ST-K,0) * node_prob
        elif type_ == 'put':
            value += max(K-ST, 0)*node_prob
        else:
            raise ValueError("type_ must be 'call' or 'put'" )
    
    return value*np.exp(-r*T)


def get_data(symbol):
    obj = web.YahooOptions(f'{symbol}')
    obj.headers = {'User-Agent': 'Firefox'}
    
    df = obj.get_all_data()

    df.reset_index(inplace=True)

    df['mid_price'] = (df.Ask+df.Bid) / 2
    df['Time'] = (df.Expiry - dt.datetime.now()).dt.days / 255
    
    return df[(df.Bid>0) & (df.Ask >0)]

# Main Method
# Roughwork for Binomial option pricing model

df = get_data('TSLA')

prices = [] 


for row in df.itertuples():
    price = binom_EU1(row.Underlying_Price, row.Strike, row.Time, 0.01, 0.5, 20, row.Type)
    prices.append(price)
    
    
df['Price'] = prices
    
df['error'] = df.mid_price - df.Price 
    
    
exp1 = df[(df.Expiry == df.Expiry.unique()[2]) & (df.Type=='call')]


plt.plot(exp1.Strike, exp1.mid_price,label= 'Mid Price')
plt.plot(exp1.Strike, exp1.Price, label = 'Calculated Price')
plt.xlabel('Strike')
plt.ylabel('Call Value')
plt.legend()

plt.show()