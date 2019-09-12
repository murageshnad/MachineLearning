
# coding: utf-8

# In[26]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


# In[27]:


#Importing dataset into the datafrmae
df = pd.read_csv("D:\\MCA\\MCA 5 SEM\\ml\\pro\\logistic\\insurance_data.csv")
df.head()


# In[28]:


plt.scatter(df.age,df.bought_insurance,marker='+',color='red')


# In[29]:


from sklearn.model_selection import train_test_split


# In[32]:


X_train,X_test,y_train,y_test = train_test_split(df[['age']],df.bought_insurance,train_size = 0.9,random_state = 0)


# In[34]:


from sklearn.linear_model import LogisticRegression


# In[35]:


model = LogisticRegression()


# In[36]:


model.fit(X_train,y_train)


# In[22]:


X_test


# In[38]:


y_test


# In[37]:


y_predict = model.predict(X_test)
y_predict


# In[39]:


model.score(X_test,y_test)


# In[40]:



model.predict_proba(X_test)


# In[41]:


X_test


# In[42]:


df.shape

