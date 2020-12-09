#!/usr/bin/env python
# coding: utf-8

# In[21]:


import math, time 
import numpy as np 
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix

import warnings
warnings.filterwarnings('ignore')


# In[22]:


start = time.time() 
MNIST_train_small_df = pd.read_csv('dataset for SVM/mnist_subset.csv', sep=',', index_col=0)
print (MNIST_train_small_df.head(3))
print (MNIST_train_small_df.shape)


# In[23]:


#Visualizing the number of classes (digits) and corresponding data rows in each class
plt.plot(figure = (25,25))
fig = sns.countplot(MNIST_train_small_df.index)
fig.set(xlabel='digits',ylabel='count')
plt.title('Number of digit classes in training dataset')

#for p, label in zip(fig.patches, MNIST_train_small_df.index.value_counts()):
#    fig.annotate(label, (p.get_x(), p.get_height()+15))

plt.show()


# In[24]:


X_tr = MNIST_train_small_df.iloc[:,:] # iloc ensures X_tr will be a dataframe
y_tr = MNIST_train_small_df.index

print (y_tr.value_counts())
print(X_tr.shape)


# In[27]:


train_sizes = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
kernels = ['linear', 'rbf', 'poly']

for k in kernels:
    train_acc = []
    test_acc = []
    train_time = []

    for t_size in train_sizes:
        ts = time.time()
        X_train, X_test, y_train, y_test = train_test_split(X_tr,y_tr,test_size=0.2, train_size=t_size, random_state=30, stratify=y_tr)

        X_train.iloc[X_train>0]=1
        X_test.iloc[X_test>0]=1

        clf = svm.SVC(kernel = k, gamma=0.001)
        clf.fit(X_train, y_train)
        train_acc.append(clf.score(X_train,y_train))
        y_predict = clf.predict(X_test)
        test_acc.append(metrics.accuracy_score(y_true=y_test, y_pred=y_predict))

        te = time.time() - ts
        train_time.append(te)
        
    result = pd.DataFrame({"Train Size": train_sizes, "Training Accuracy Score": train_acc, "Test Accuracy Score": test_acc, "Time Taken": train_time})
    print("Results of Model: ", k)
    print(result)
    
    fig, ax1 = plt.subplots()
    ln1 = ax1.plot(train_sizes, train_acc, 'ro-', label='Training Accuracy')
    ln2 = ax1.plot(train_sizes, test_acc, 'bx-', label='Test Accuracy')
    ax1.set_xlabel('Train Size')
    ax1.set_ylabel('Accuracy Score')

    ax2 = ax1.twinx()
    ln3 = ax2.plot(train_sizes, train_time, 'g^-', label='Time to Train the Model')
    ax2.set_ylabel('Time (s)')

    ln = ln1 + ln2 + ln3
    lb = [l.get_label() for l in ln]
    ax1.legend(ln, lb, loc=0)
    
    fig.tight_layout()
    plt.title("Correlation of Model Accuracy Score")
    plt.show()


# In[28]:


#From the analysis above it is clear that a train size of 0.7 has highest accuracy score. So we choose that for future analysis
#Also we use rbf kernel function in our SVM model
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X_tr,y_tr,test_size=0.2, train_size=0.7, random_state=30, stratify=y_tr)

Xtrain.iloc[Xtrain>0]=1
Xtest.iloc[Xtest>0]=1

folds = KFold(n_splits = 5, shuffle = True, random_state = 10)
hyper_params = [ {'gamma': [1e-2, 1e-3, 1e-4],
                     'C': [5,10]}]

model = svm.SVC(kernel = 'rbf')
model_gridcv = GridSearchCV(estimator = model, 
                        param_grid = hyper_params, 
                        scoring= 'accuracy', 
                        cv = folds, 
                        verbose = 1,
                        return_train_score=True)

model_gridcv.fit(Xtrain, Ytrain)


# In[30]:


#Getting the results of Grid Search CV onto dataframe
df_gridsearch = pd.DataFrame(model_gridcv.cv_results_)
df_gridsearch

# converting C to numeric type for plotting on x-axis
df_gridsearch['param_C'] = df_gridsearch['param_C'].astype('int')

plotA = df_gridsearch[df_gridsearch['param_gamma']==0.01]
plotB = df_gridsearch[df_gridsearch['param_gamma']==0.001]
plotC = df_gridsearch[df_gridsearch['param_gamma']==0.0001]

# # plotting
plt.figure(figsize=(16,8))

# subplot 1 -> for GridSearchCV model with gamma=0.01
plt.subplot(131)
plt.plot(plotA["param_C"], plotA["mean_test_score"])
plt.plot(plotA["param_C"], plotA["mean_train_score"])
plt.xlabel('Parameter C')
plt.ylabel('Accuracy')
plt.title("Gamma=0.01")
plt.ylim([0.60, 1])
plt.legend(['Test Accuracy', 'Train Accuracy'], loc='upper left')
plt.xscale('log')

# subplot 2 -> for GridSearchCV model with gamma=0.001
plt.subplot(132)
plt.plot(plotB["param_C"], plotB["mean_test_score"])
plt.plot(plotB["param_C"], plotB["mean_train_score"])
plt.xlabel('Parameter C')
plt.ylabel('Accuracy')
plt.title("Gamma=0.001")
plt.ylim([0.60, 1])
plt.legend(['Test Accuracy', 'Train Accuracy'], loc='upper left')
plt.xscale('log')

# subplot 3 -> for GridSearchCV model with gamma=0.0001
plt.subplot(133)
plt.plot(plotC["param_C"], plotC["mean_test_score"])
plt.plot(plotC["param_C"], plotC["mean_train_score"])
plt.xlabel('Parameter C')
plt.ylabel('Accuracy')
plt.title("Gamma=0.0001")
plt.ylim([0.60, 1])
plt.legend(['Test Accuracy', 'Train Accuracy'], loc='upper left')
plt.xscale('log')


# In[ ]:




