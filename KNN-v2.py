#!/usr/bin/env python
# coding: utf-8

# In[31]:


from sklearn import datasets
iris = datasets.load_iris()


# In[32]:


X = iris.data
# Iris.data contains the features or independent variables.
y = iris.target
# Iris.target contains the labels or the dependent variables.


# In[33]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.50)


# In[34]:


from scipy.spatial import distance  # Built in function called distance.

                                    #Defining the n dimensional distance as euc.
def euc(a,b):                       # Lists of numeric features. 
    return distance.euclidean(a,b)  # Measure and return the distance between 2 points 
                                    # i.e. the training point and a test point.


# In[35]:



# First we implement a class. Classes let you structure code in a specific way.(source --> https://learnpythonthehardway.org/book/ex40.html)

class OneNeighborClassifier():                # This 'class' has 2 Methods : Fit and Predict
    
    #Each step is followed by a comment which explains how the classifier is working 
    
    def fit(self, X_train, y_train):          # Takes features and labels as input
        self.X_train = X_train                # Storing the X_train in self.X_train
        self.y_train = y_train                # Storing the y_train in self.y_train
                                              # This is like the ML classifier will memorize the values 
        
    def predict (self, X_test):               # Receives features from the testing data and returns predictions
        predictions = []                      # List of predictions, since X_test is a 2D array or a list of lists.
        for row in X_test:                    # Each row contains the features for one testing example
            label = self.closest(row)         # We are calling the function that we are creating in the next block
                                              # to find the closest training point from the test point
            predictions.append(label)         # Add the labels to the predictions list to fill it.
        return predictions                    # Return predictions as the output
    
    def closest(self, row):                   # Create the function closest such that -->
        best_dist = euc(row, self.X_train[0]) # Measure the shortest distance a test point and the first train point
        best_index = 0                        # Keep track of the index of the train point that is closest
        for i in range (1, len(self.X_train)):# Iterate over the different training points
            dist = euc(row, self.X_train[i])
            if dist < best_dist:              # The moment we find a closer one, we update our variables.
                best_dist = dist              # If dist is shorter than best_dist, then its the new best_dist
                best_index = i                # Using the index of best_dist to return label of the closest training pt.
        return self.y_train[best_index]       # Return that label


# In[36]:


my_classifier = OneNeighborClassifier()
my_classifier.fit(X_train, y_train)


# In[37]:


pred = my_classifier.predict(X_test)


# In[38]:


from sklearn.metrics import accuracy_score
print ('Accuracy of the classifier is', accuracy_score(y_test, pred)*100, '%')


# In[39]:


# summarize the fit of the model
from sklearn import metrics
print(); print(metrics.classification_report(y_test, pred))
print(); print(metrics.confusion_matrix(y_test, pred))


# In[ ]:





# In[ ]:




