import pandas as pd
import pandas_datareader.data as web
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt

plt.style.use('ggplot')

TSLA = web.YahooOptions('TSLA')
TSLA.headers = {'User-Agent': 'Firefox'}
alloptions = TSLA.get_all_data()

alloptions.reset_index(inplace=True)
alloptions.to_csv('options.csv', index=False)
exp1 = alloptions[alloptions.Expiry == pd.Timestamp(TSLA.expiry_dates[1])]

calls = exp1[exp1.Type=='call']
calls['C'] = (calls.Bid+calls.Ask)/2
puts = exp1[exp1.Type=='put']
puts['P'] = (puts.Bid+puts.Ask)/2

df = pd.merge(calls, puts, how='inner', on ='Strike')

print(df)

df['S'] = df.Underlying_Price_x
df['Parity_P'] = df.C + df.Strike - df.S

plt.plot(df.Strike, df.P, label='Observed')
plt.plot(df.Strike,df.Parity_P, label=r'$C+ K -S_0$')
plt.xlabel('Strike')
plt.ylabel('Price')
plt.legend()
plt.show()

df['Time'] = (df.Expiry_x - dt.datetime.now()).dt.days / 255
df['r'] = -np.log( (df.S+df.P-df.C)/ df.Strike) / df.Time 

plt.figure(figsize=(8,4))
plt.plot(df.Strike, df.r, label=r' $-\frac {ln \left[ \frac{S_0 + P - C}{K} \right]}{T}$')
plt.xlabel('Strike')
plt.ylabel('Interest Rate')
plt.vlines(df.S, df.r.min(), df.r.max(), linestyle='--', label=r'$S_0$')
plt.legend(fontsize=20)
plt.show()

df['Parity_P2'] = df.C+df.Strike*np.exp(-df.r*df.Time) - df.S
plt.plot(df.Strike, df.P, label='Observed')
plt.plot(df.Strike,df.Parity_P2, label=r'$C+Ke^{-rT} - S_0$')
plt.legend()
plt.show()