#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

path = "C:/Users/kwass/automobile.csv"
df = pd.read_csv(path)
df.head(5)


# In[3]:


#import modules for linear regression
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm


# In[32]:


X = df[['highway-mpg']]
Y = df['price']

lm.fit(X,Y)
#MSE
Yhat0=lm.predict(X)
print('The output of the first four predicted value is: ', Yhat[0:4])

print('The R-square is: ', lm.score(X, Y))


# lm.intercept_

# In[7]:


#𝑌ℎ𝑎𝑡=𝑎+𝑏𝑋 avec a=lm.intercept_ et b=lm.coef_
print(lm.intercept_)
print(lm.coef_)


# In[11]:


#Multiple Linear Regression
#Yℎ𝑎𝑡=𝑎+𝑏1𝑋1+𝑏2𝑋2+𝑏3𝑋3+𝑏4𝑋4

Z = df[['curb-weight','engine-size', 'highway-mpg']]
lm.fit(Z, df['price'])
print(lm.intercept_)
print(lm.coef_)


# In[12]:


# import the visualization package: seaborn
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

width = 12
height = 10
plt.figure(figsize=(width, height))
sns.regplot(x="highway-mpg", y="price", data=df)
plt.ylim(0,)


# In[13]:


plt.figure(figsize=(width, height))
sns.regplot(x="peak-rpm", y="price", data=df)
plt.ylim(0,)


# In[30]:


Y_hat = lm.predict(Z)

plt.figure(figsize=(width, height))


ax1 = sns.distplot(df['price'], hist=False, color="r", label="Actual Value")
sns.distplot(Y_hat, hist=False, color="b", label="Fitted Values" , ax=ax1)


plt.title('Actual vs Fitted Values for Price')
plt.xlabel('Price (in dollars)')
plt.ylabel('Proportion of Cars')

plt.show()
plt.close()


# In[17]:


#Polynomial Regression and Pipelines

def PlotPolly(model, independent_variable, dependent_variabble, Name):
    x_new = np.linspace(15, 55, 100)
    y_new = model(x_new)

    plt.plot(independent_variable, dependent_variabble, '.', x_new, y_new, '-')
    plt.title('Polynomial Fit with Matplotlib for Price ~ Length')
    ax = plt.gca()
    ax.set_facecolor((0.898, 0.898, 0.898))
    fig = plt.gcf()
    plt.xlabel(Name)
    plt.ylabel('Price of Cars')

    plt.show()
    plt.close()

x = df['highway-mpg']
y = df['price']

# Here we use a polynomial of the 3rd order (cubic) 
f = np.polyfit(x, y, 3)
p = np.poly1d(f)
print(p)


# In[18]:


PlotPolly(p, x, y, 'highway-mpg')
#np.polyfit(x, y, 3)


# In[19]:


from sklearn.preprocessing import PolynomialFeatures
pr=PolynomialFeatures(degree=2)
pr


# In[20]:


Z_pr=pr.fit_transform(Z)
Z.shape


# In[22]:


#Data Pipelines simplify the steps of processing the data
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

Input=[('scale',StandardScaler()), ('polynomial', PolynomialFeatures(include_bias=False)), ('model',LinearRegression())]
pipe=Pipeline(Input)
pipe


# In[23]:


pipe.fit(Z,y)
ypipe=pipe.predict(Z)
ypipe[0:4]


# In[33]:


from sklearn.metrics import mean_squared_error
mse = mean_squared_error(df['price'], Yhat0)
print('The mean square error of price and predicted value is: ', mse)


# In[ ]:


#For MULTIPLE Linear Regression
Y_predict_multifit = lm.predict(Z)
print('The mean square error of price and predicted value using multifit is: ',       mean_squared_error(df['price'], Y_predict_multifit))

