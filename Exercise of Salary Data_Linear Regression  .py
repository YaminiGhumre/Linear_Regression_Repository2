#!/usr/bin/env python
# coding: utf-8

# In[1]:


### importing of libraries -
import pandas as pd
import seaborn as sns ### for sytlist Visualisation
import matplotlib.pyplot as plt


# In[2]:


data =pd.read_csv("E:/Python docs/Salary_Data.csv")
print(data.head(5))


# In[3]:


print(data.columns)


# In[6]:


sns.scatterplot(x='YearsExperience',y='Salary',data=data)


# In[4]:


X =data['YearsExperience']
print(X)


# In[5]:


y =data['Salary']
print(y)


# In[6]:


from sklearn.model_selection import train_test_split


# In[7]:


X =data['YearsExperience'].values.reshape(-1,1)


# In[8]:


y =data['Salary'].values.reshape(-1,1)


# In[9]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)


# In[16]:


from sklearn import linear_model
from sklearn.linear_model import LinearRegression


# In[18]:


model = LinearRegression()


# In[19]:


model_train = model.fit(X_train,y_train)
print(model_train)


# In[20]:


pred = model_train.predict(X_test)
pred


# In[21]:


y_test


# In[50]:


Y_test


# In[24]:


from sklearn.metrics import r2_score


# In[25]:


r2_score(y_test,pred)


# In[ ]:




