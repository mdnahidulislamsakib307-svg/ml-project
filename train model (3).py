#!/usr/bin/env python
# coding: utf-8

# In[112]:


import numpy as np 
import pandas as pd 
import joblib as jb 
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor


# In[113]:


df = pd.read_csv("YouTube_Shorts_Engagement_and_Growth_Velocity.csv")


# In[114]:


df.head()


# In[115]:


df.dtypes


# In[116]:


df.isnull().sum()


# In[117]:


df.duplicated().sum()


# In[118]:


x = df.drop(['Video_ID','Engagement_Rate_%'],axis=1)
y = df['Engagement_Rate_%']


# In[119]:


numerical_cols = x.select_dtypes(include=['int64','float64']).columns.tolist()


# In[120]:


categorical_cols = x.select_dtypes(include=['object']).columns.tolist()


# In[121]:


numerical_transformer = Pipeline(steps=[
    ('imputer',SimpleImputer(strategy='mean')),
    ('scaler',StandardScaler())
])


# In[122]:


categorical_transformer = Pipeline(steps=[
    ('imputer',SimpleImputer(strategy='most_frequent')),
    ('onehot',OneHotEncoder(handle_unknown='ignore'))
])


# In[123]:


preprocessor = ColumnTransformer(transformers=[
    ('num',numerical_transformer,numerical_cols),
    ('cat',categorical_transformer,categorical_cols)
])


# In[124]:


X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state = 42)


# In[125]:


model = Pipeline(steps=[
    ('pre',preprocessor),('reg',RandomForestRegressor(n_estimators=100,random_state=42))
])


# In[126]:


model.fit(X_train,y_train)


# In[127]:


y_pred = model.predict(X_test)
print(f'Accuracy:{r2_score(y_test,y_pred)*100}')


# In[128]:


jb.dump(model,'RandomForestRegresor.pkl')


# In[ ]:





# In[ ]:




