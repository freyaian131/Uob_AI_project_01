#!/usr/bin/env python
# coding: utf-8

# In[19]:


from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import yfinance
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

import seaborn as sns
sns.set_style('whitegrid')
plt.style.use("fivethirtyeight")
#plt.style.use('Solarize_Light2')
get_ipython().run_line_magic('matplotlib', 'inline')
# For reading stock data from yahoo
from pandas_datareader.data import DataReader



# In[21]:


# To Obtain CSV
df=pd.read_csv("TSLA_2018_2021.csv")
# Show teh data
df


# In[23]:



plt.figure(figsize=(15, 6))
plt.subplots_adjust(top=1.25, bottom=1.2)
# # To change the format of Date in Python
dates = mdates.date2num(df['Date'])

adjClose = df['Adj Close']
plt.ylabel('Adj Close',fontsize=22)
plt.xlabel(None)
plt.title(f"Closing Price of TSLA")
plt.plot_date(dates, adjClose)
plt.plot(dates, adjClose)
plt.tight_layout()


# In[ ]:





# In[63]:


import pandas as pd
import numpy as np
#import googlefinance 
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
plt.style.use("fivethirtyeight")
get_ipython().run_line_magic('matplotlib', 'inline')



# In[65]:


ma_day = [10, 20, 50]
for ma in ma_day:
    column_name = f"MA for {ma} days"
    df[column_name] = df['Adj Close'].rolling(ma).mean()
    


# In[69]:


df['Adj Close'].rolling(ma).mean()


# In[71]:



df_ = df.set_index(['Date'])

df_.head(10)


# In[74]:


df_[['Adj Close', 'MA for 10 days', 'MA for 20 days', 'MA for 50 days']].plot(ax=None)
plt.title('Moving Average of TSLA')

plt.show()

