#!/usr/bin/env python
# coding: utf-8

# In[29]:


# load libraries
from sklearn import datasets
from sklearn import metrics
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt    
import seaborn as sns
    
iris = datasets.load_iris()


# In[30]:


X = iris.data
# Iris.data contains the features or independent variables.
y = iris.target
# Iris.target contains the labels or the dependent variables.


# In[32]:



X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.50)


# In[33]:


# fit a CART model to the data
model = MLPClassifier()
model.fit(X_train, y_train)
print(); print(model)


# In[34]:


# make predictions
expected_y  = y_test
predicted_y = model.predict(X_test)


# In[24]:


model.fit(X_train, y_train)


# In[35]:


# summarize the fit of the model
print(); print(metrics.classification_report(expected_y, predicted_y))
print(); print(metrics.confusion_matrix(expected_y, predicted_y))


# In[ ]:





# In[ ]:




