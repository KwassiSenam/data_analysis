#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

path='https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/automobileEDA.csv'
df = pd.read_csv(path)
df.head()


# In[2]:


get_ipython().run_cell_magic('capture', '', '! pip install seaborn\n\nimport matplotlib.pyplot as plt\nimport seaborn as sns\n%matplotlib inline ')


# In[3]:


print(df.dtypes)


# In[4]:


#Let's begin with Continuous numerical variables(int and float) : we will use regplot() to visualize the correlation
df.corr() #the correlation between the elements


# In[5]:


df[['bore', 'stroke', 'compression-ratio', 'horsepower']].corr()


# In[6]:


# Engine size as potential predictor variable of price
sns.regplot(x="engine-size", y="price", data=df)
plt.ylim(0,)


# In[7]:


df[["engine-size", "price"]].corr()


# In[9]:


sns.regplot(x="highway-mpg", y="price" , data = df)
plt.ylim(0,)


# In[10]:


df[["highway-mpg", "price"]].corr()


# In[11]:


#now let's see Categorical variables(object) : we will use boxplot() to visualize the correlation
sns.boxplot(x="body-style", y="price", data=df)
#nb : body-style is not a good indication


# In[12]:


sns.boxplot(x="engine-location", y="price", data=df)
#nb : engine-location is a potential good predictor of price


# In[18]:


drive_wheels_counts = df['drive-wheels'].value_counts().to_frame()
drive_wheels_counts.rename(columns={'drive-wheels': 'value_counts'}, inplace=True)
drive_wheels_counts.index.name = "drive-wheels"
drive_wheels_counts


# In[16]:


engine_loc_counts = df['engine-location'].value_counts().to_frame()
engine_loc_counts.rename(columns={'engine-location': 'value_counts'}, inplace=True)
engine_loc_counts.index.name = 'engine-location'
engine_loc_counts.head(10)


# In[19]:


df['drive-wheels'].unique() #show the differents values in this column


# In[20]:


# grouping results
df_group_one = df[['drive-wheels','body-style','price']]
df_group_one = df_group_one.groupby(['drive-wheels'], as_index=False).mean()
df_group_one


# In[22]:


df_gp_test = df[['drive-wheels','body-style','price']]
df_group_one1 = df_gp_test.groupby(['drive-wheels','body-style'], as_index=False).mean()
df_group_one1


# In[24]:


grouped_pivot = df_group_one1.pivot(index ='drive-wheels' , columns= 'body-style')
grouped_pivot = grouped_pivot.fillna(0) #fill missing values with 0
grouped_pivot


# In[25]:


#we use a heat map to visualize the relationship
plt.pcolor(grouped_pivot, cmap='RdBu')
plt.colorbar()
plt.show()


# In[26]:


#for better visualization
fig, ax = plt.subplots()
im = ax.pcolor(grouped_pivot, cmap='RdBu')

#label names
row_labels = grouped_pivot.columns.levels[1]
col_labels = grouped_pivot.index

#move ticks and labels to the center
ax.set_xticks(np.arange(grouped_pivot.shape[1]) + 0.5, minor=False)
ax.set_yticks(np.arange(grouped_pivot.shape[0]) + 0.5, minor=False)

#insert labels
ax.set_xticklabels(row_labels, minor=False)
ax.set_yticklabels(col_labels, minor=False)

#rotate label if too long
plt.xticks(rotation=90)

fig.colorbar(im)
plt.show()


# In[27]:


from scipy import stats

pearson_coef, p_value = stats.pearsonr(df['wheel-base'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)


# In[31]:


grouped_test2=df_gp_test[['drive-wheels', 'price']].groupby(['drive-wheels'])

grouped_test2.get_group('4wd')['price']
grouped_test2.head(2)


# In[32]:


# ANOVA
f_val, p_val = stats.f_oneway(grouped_test2.get_group('fwd')['price'], grouped_test2.get_group('rwd')['price'], grouped_test2.get_group('4wd')['price'])  
 
print( "ANOVA results: F=", f_val, ", P =", p_val)   


# In[33]:


f_val, p_val = stats.f_oneway(grouped_test2.get_group('fwd')['price'], grouped_test2.get_group('rwd')['price'])  
 
print( "ANOVA results: F=", f_val, ", P =", p_val )


# In[ ]:




