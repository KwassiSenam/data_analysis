#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

other_path = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/auto.csv"

df = pd.read_csv(other_path, header = None)


# In[2]:


print("The first 5 rows of the dataframe") 
df.head(5)


# In[3]:


print("The last 5 rows of the dataframe") 
df.tail(5)


# In[4]:


headers = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
         "peak-rpm","city-mpg","highway-mpg","price"]
df.columns = headers
df.head(10)


# In[6]:


df1 = df.replace('?', np.NaN)
df = df1.dropna(subset=['price'], axis=0)
df.head(20)


# In[7]:


print(df.columns) #show the name of the columns
df.to_csv("automobile.csv", index = False)


# In[10]:


print(df.dtypes) #objet equivalent au string, int64->int etc
df.describe(include = "all") #statistical summary of each column including object colums
df.describe() #statistical summary of each column without object colums


# In[11]:


df[['length', 'compression-ratio']].describe()


# In[12]:


df.info()


# In[ ]:




