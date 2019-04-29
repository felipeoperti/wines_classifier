#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import libraries
# Common libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Machine Learning libraries
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
import tensorflow
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from keras.utils.np_utils import to_categorical
from sklearn.metrics import classification_report
from sklearn.externals import joblib
from imblearn.over_sampling import RandomOverSampler


# In[2]:


######################################################
########## List of functions #########################
######################################################

# Function to check if a variable could be cast to float
def check_var(var,i):
    try:
        float(var)
    except:
        return var,i


# In[3]:


# Read the Data
df = pd.read_csv('winequality.csv',sep=';')
df[:5]


# In[4]:


# Some information about the dataframe
print("Lenght of the dataframe: "+str(len(df))+". Number of target: "+str(len(df.columns)-1)+".")
# Show types of column
print("Initial types of column:\n"+str(df.dtypes))
# Create a Label encoder of 'type'
le = preprocessing.LabelEncoder()
df["type sc"] = le.fit_transform(df["type"])
# Cast 'alcohol' to float 
# Find values in 'alcohol' that could not be casted to float
not_cast = []
for i in range(len(df['alcohol'].values)):
    try_ = check_var(df['alcohol'].values[i],i)
    if try_!=None:
        var,j = check_var(df['alcohol'].values[i],i)
        not_cast.append([var,j])
# Drop lines from Dataframe that could not be cast to float
df = df.drop(np.array(not_cast)[:,1].astype(int),axis='index')
df['alcohol'] = df['alcohol'].astype(float)
# Show final types of column
print("\n \n Final types of column:\n"+str(df.dtypes))


# In[5]:


# A little bit of statistics
print("Number of null values for column:\n"+str(df.isnull().sum(axis = 0)))
# Print distribution for each 
for col_name in np.delete(df.columns.values,0):
    plt.figure()
    sns.distplot(df[col_name])


# In[6]:


##############################################################
# Reflexions
# The only really strange distribution is the density. In my
# opinion ir oculd be a really important feature, or it could 
# a problem for the analysis.
# I will train the algorithm with and without that variable.
# Furthermore I will train the algorithm with and without the onehotencoding in whine


# In[7]:


# Analysis of the target 
print("There are the following unique values in 'quality':\n"+str(df['quality'].unique()))
# A the only classes present in the data are: 3,4,5,6,7,8,9. I can only classify the wines in these 7 classes
# I will not have wines with 'quality' 0,1,2, and 10
print("Count number of lines for each different target:\n"+str(df['quality'].value_counts()))


# In[8]:


#####################################
# Classification problem
# I need a multiclass classifier.
# For that reason I try two methods: RandomForest and neuronal network
# First I define the training set and the test set first with the variable 'density'
features_col = ['fixed acidity', 'volatile acidity', 'citric acid',
       'residual sugar', 'chlorides', 'free sulfur dioxide',
       'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol',
        'type sc']
X = df[features_col].values
y = df['quality'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
print("Count classes in y_train:\n "+str(pd.Series(y_train).value_counts()))
print("Count classes in y_test:\n "+str(pd.Series(y_test).value_counts()))
# The data are to unbalanced. It is necessary an oversampling
ros = RandomOverSampler(random_state=0)
X_train, y_train = ros.fit_resample(X_train, y_train)
print("Count classes in y_train:\n "+str(pd.Series(y_train).value_counts()))


# In[9]:


#################################
# Random Forest Classifier 
parameters = {
    'n_estimators':[300],
    'criterion':['gini'],
    'max_depth':[5,10,15,20,25,30],
    'min_samples_leaf': [1,2,3,5,8,10,12,15,20],
    'max_features':['auto','log2']       
}
rf = RandomForestClassifier()
CV_rd = GridSearchCV(rf, parameters, cv=5,n_jobs=-1)
CV_rd.fit(X_train,y_train)


# In[10]:


# Analysis the result of the GridSearchCV of randomforestClassifier
print("The best parameters found by:\n "+str(CV_rd.best_params_))
classifier = CV_rd.best_estimator_
## Training prediction
y_pred = classifier.predict(X_train)
result = {"Accurancy": accuracy_score(y_train, y_pred),
          "Precision": precision_score(y_train, y_pred,average=None),
          "Recal": recall_score(y_train, y_pred,average=None),
          "Confusion Matrix":confusion_matrix(y_train, y_pred)
}
# Test prediction
y_pred = classifier.predict(X_test)
result = {"Accurancy": accuracy_score(y_test, y_pred),
          "Precision": precision_score(y_test, y_pred,average=None),
          "Recal": recall_score(y_test, y_pred,average=None),
          "Precision_MW": precision_score(y_test, y_pred,average='weighted'),
          "Recal_MW": recall_score(y_test, y_pred,average='weighted'),
          "Confusion Matrix":confusion_matrix(y_test, y_pred)
}
joblib.dump(classifier, 'random_forest.pkl') 
print(classifier.feature_importances_)
print(result)


# In[11]:


#################################
# Gradient Boosting Classifier
parameters = {
    'n_estimators':[10],
    'max_depth':[5,10,15,20,25],
    'min_samples_leaf': [2,3,5,8,10,12,15],
    'max_features':['auto','log2']    
}
gb = GradientBoostingClassifier()
CV_rd = GridSearchCV(gb, parameters, cv=5,n_jobs=-1)
CV_rd.fit(X_train,y_train)


# In[12]:


# Analysis the result of the GridSearchCV of gradient boosting
print("The best parameters found by:\n "+str(CV_rd.best_params_))
classifier = CV_rd.best_estimator_
## Training prediction
y_pred = classifier.predict(X_train)
result = {"Accurancy": accuracy_score(y_train, y_pred),
          "Precision": precision_score(y_train, y_pred,average=None),
          "Recal": recall_score(y_train, y_pred,average=None),
          "Confusion Matrix":confusion_matrix(y_train, y_pred)
}
# Test prediction
y_pred = classifier.predict(X_test)
result = {"Accurancy": accuracy_score(y_test, y_pred),
          "Precision": precision_score(y_test, y_pred,average=None),
          "Recal": recall_score(y_test, y_pred,average=None),
          "Precision_MW": precision_score(y_test, y_pred,average='weighted'),
          "Recal_MW": recall_score(y_test, y_pred,average='weighted'),
          "Confusion Matrix":confusion_matrix(y_test, y_pred)
}
joblib.dump(classifier, 'gradient_boosting.pkl') 
print(classifier.feature_importances_)
print(result)


# In[ ]:




