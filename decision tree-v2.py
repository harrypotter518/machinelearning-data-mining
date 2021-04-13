#!/usr/bin/env python
# coding: utf-8

# In[24]:


from sklearn import datasets
iris = datasets.load_iris()


# In[25]:


X = iris.data
# Iris.data contains the features or independent variables.
y = iris.target
# Iris.target contains the labels or the dependent variables.


# In[26]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.50)


# In[27]:


from sklearn import tree

model = tree.DecisionTreeClassifier()


# In[28]:


model


# In[29]:


model.fit(X_train, y_train)


# In[30]:


y_predict = model.predict(X_test)


# In[31]:


from sklearn.metrics import accuracy_score

accuracy_score(y_test, y_predict)


# In[32]:


# summarize the fit of the model
from sklearn import metrics
print(); print(metrics.classification_report(y_test, y_predict))
print(); print(metrics.confusion_matrix(y_test, y_predict))


# In[ ]:




