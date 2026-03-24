#!/usr/bin/env python
# coding: utf-8

# In[15]:


import numpy as np 
import pandas as pd 
import joblib as jb


# In[16]:


df = pd.read_csv("YouTube_Shorts_Engagement_and_Growth_Velocity.csv")


# In[17]:


df.dtypes


# In[18]:


from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib as jb

model = jb.load("RandomForestRegresor.pkl")

app = FastAPI(title="Engagement_Rate_Percent Prediction")


class YoutubeData(BaseModel):

    Title:str
    Channel_Name:str
    Views:int
    Likes:int
    Comments:int
    Age_In_Days:int
    Views_Per_Day:float
    Video_URL:str
    Description_Length:int


@app.get("/")
def home():

    return {"message":"API Running"}


@app.post("/predict")
def predict(data:YoutubeData):

    df = pd.DataFrame({

        'Title':[data.Title],
        'Channel_Name':[data. Channel_Name],
        'Views':[data.Views],
        'Likes':[data.Likes],
        'Comments':[data.Comments],
        'Age_In_Days':[data.Age_In_Days],
        'Views_Per_Day':[data.Views_Per_Day],
        'Video_URL':[data.Video_URL],
        'Description_Length':[data.Description_Length],
        
    })

    prediction = model.predict(df)

    return {"prediction":prediction[0]}


# In[ ]:




