#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
plt.style.use('dark_background')


# In[2]:


rain=pd.read_csv(r"C:\Users\user\Downloads\archive (5)\weatherAUS.csv")


# In[3]:


rain.head()


# In[4]:


rain.info()


# In[5]:


rain.isnull().sum()


# In[6]:


rain=rain.drop(['Date','Location','Evaporation','Sunshine','Cloud9am','Cloud3pm'],axis=1)


# In[7]:


rain=rain.dropna(axis=0)
rain


# In[8]:


rain.shape


# In[9]:


rain.columns


# In[10]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
rain['WindGustDir']=le.fit_transform(rain['WindGustDir'])
rain['WindDir9am']=le.fit_transform(rain['WindDir9am'])
rain['WindDir3pm']=le.fit_transform(rain['WindDir3pm'])
rain['RainToday']=le.fit_transform(rain['RainToday'])
rain['RainTomorrow']=le.fit_transform(rain['RainTomorrow'])


# In[11]:


x=rain.drop(['RainTomorrow'],axis=1)
y=rain['RainTomorrow']


# In[12]:


x.shape


# In[13]:


#plt.figure(figsize = (8,8))
sns.scatterplot(x = 'MaxTemp', y = 'MinTemp', hue = 'RainTomorrow' , palette = 'inferno',data = rain)


# In[14]:


sns.scatterplot(x = 'Humidity9am', y = 'Temp9am', hue = 'RainTomorrow' , palette = 'inferno',data = rain)


# In[15]:


plt.figure(figsize=(8,8))
sns.heatmap(rain.corr())


# In[16]:


rain.to_csv('Cleaned rain.csv')


# In[17]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)


# In[18]:


from sklearn.metrics import accuracy_score


# # Logistic Regression

# In[19]:


from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(x_train,y_train)
y_predict=lr.predict(x_test)
print(accuracy_score(y_test,y_predict))


# # Decision tree

# In[20]:


from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()
dt.fit(x_train,y_train)
y_predict=dt.predict(x_test)
print(accuracy_score(y_test,y_predict))


# # Random Forest Classifier

# In[27]:


from sklearn.ensemble import RandomForestClassifier
rdf=RandomForestClassifier()
rdf.fit(x_train,y_train)
y_predict=rdf.predict(x_test)
print(accuracy_score(y_test,y_predict))


# # XGBoost Classifier

# In[28]:


get_ipython().system('pip install xgboost')


# In[29]:


import xgboost as xgb
xgb=xgb.XGBClassifier()
xgb.fit(x_train,y_train)
pred=xgb.predict(x_test)
print(accuracy_score(y_test,pred))


# In[ ]:


# Accuracy score is highest for RandomForestClassifier and XGBoost Classifier 
# Here we are using RandomForestClassifier to build a prediction model


# In[31]:


Input_data = (9.7,31.9,0.0,6,80.0,9,7,7.0,28.0,42.0,9.0,1008.9,1003.6,18.3,30.2,0)

Input_data_as_numpy_array = np.asarray(Input_data)

# reshaping the data as we are pridicting for one instance

input_reshaping = Input_data_as_numpy_array.reshape(1,-1)

m= rdf.predict(input_reshaping)

print(m)
if(m==1):
    print("Yes,There will be rain tomorrow")
if(m==0):
    print("No,There will be no rain tomorrow")

