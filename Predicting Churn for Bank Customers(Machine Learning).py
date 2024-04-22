#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")


# ## Read the data

# In[2]:


data = pd.read_csv("Churn_Modelling (1).csv")


# In[3]:


data.shape


# ## Data Exploration

# In[4]:


data.head()


# In[5]:


data.shape


# In[6]:


data.info()


# In[7]:


# Finding the the seperate object coloumn
data.select_dtypes(include='object').columns


# In[8]:


# finging the length
len(data.select_dtypes(include='object').columns)


# In[9]:


data.select_dtypes(include=['float64','int64']).columns


# In[10]:


len(data.select_dtypes(include=['float64','int64']).columns)


# ## Statistical summary

# In[11]:


data.describe()


# In[12]:


# Dealing with missing value
data.isnull().values.any()


# In[13]:


data.isnull().values.sum()


# ## Encoding the categorical data

# In[14]:


data.select_dtypes(include='object').columns


# In[15]:


data.head()


# In[16]:


data = data.drop(columns=['RowNumber','Surname'])


# In[17]:


data.head()


# In[18]:


# again checking object coloumn
data.select_dtypes(include='object').columns


# In[19]:


# check uniqueness of 'Geography'
data['Geography'].unique()


# In[20]:


# check uniqueness of 'Gender'
data['Gender'].unique()


# In[21]:


data.groupby('Geography').mean()


# In[22]:


data.groupby("Gender").mean()


# In[23]:


# one hot encoding
data = pd.get_dummies(data=data,drop_first=True)


# In[24]:


data.head()


# In[25]:


import warnings
warnings.filterwarnings("ignore")


# In[26]:


# countplot
sns.countplot(data['Exited'])
plt.plot


# In[27]:


# if we take the numerical data
# staying customer with bank
(data.Exited==0).sum()


# In[28]:


# Not staying with bank
(data.Exited==1).sum()


# In[29]:


data_1 = data.drop(columns='Exited')


# In[30]:


data.corrwith(data['Exited']).plot.bar(
figsize=(16,9),title='Correlated with exited',rot=45,grid=True
)


# In[31]:


corr = data.corr()
corr


# In[32]:


plt.figure(figsize=(16,9))
sns.heatmap(corr,annot=True)
           


# In[33]:


## bSplittig the dataset


# In[34]:


data.head()


# In[35]:


#  Independent variable
x = data.drop(columns='Exited')


# In[36]:


# dependent variable
y = data['Exited']


# In[37]:


from sklearn.model_selection import train_test_split


# In[38]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)


# In[39]:


x_train.shape


# In[40]:


x_test.shape


# In[41]:


y_train.shape


# In[42]:


y_test.shape


# In[43]:


from sklearn.preprocessing import StandardScaler


# In[44]:


sc = StandardScaler()


# In[45]:


x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


# In[46]:


x_train


# In[47]:


x_test


# ## Building the model

# ### logestic regression

# In[48]:


from sklearn.linear_model import LogisticRegression
classifier_lr = LogisticRegression(random_state=0)
classifier_lr.fit(x_train, y_train)


# In[49]:


y_pred = classifier_lr.predict(x_test)


# In[50]:


from sklearn.metrics import accuracy_score,confusion_matrix,f1_score,precision_score,recall_score


# In[51]:


acc = accuracy_score(y_test,y_pred)
f1 = f1_score(y_test,y_pred)
prec = precision_score(y_test,y_pred)
rec = recall_score(y_test,y_pred)


# In[52]:


result = pd.DataFrame([['Logistic regression',acc,f1,prec,rec]],
                      columns=['Model','Accuracy','F1','precision','Recall'])


# In[53]:


result


# 
#     1)Accuracy: This metric tells us the proportion of correctly predicted instances out of all the instances in the dataset. An accuracy of 0.8115 means that the model correctly predicted around 81.15% of the instances in the dataset.
# 
#     2)F-score: The F-score is a measure of a model's accuracy on a dataset. It's a harmonic mean of precision and recall. It ranges from 0 to 1, where 1 is the best possible F-score. In this case, the F-score is 0.339755, indicating a moderate balance between precision and recall.
# 
#     3)Precision: Precision measures the proportion of true positive instances out of all the instances that the model predicted as positive. In other words, it tells us how many of the predicted positive instances were actually positive. A precision of 0.584337 means that around 58.43% of the instances predicted as positive were indeed positive.
# 
#     4)Recall: Recall, also known as sensitivity, measures the proportion of true positive instances that were correctly identified by the model out of all actual positive instances. It tells us how many of the actual positive instances were captured by the model. A recall of 0.239506 means that around 23.95% of the actual positive instances were correctly identified by the model.
# 
#  These metrics collectively provide insights into the performance of the logistic regression model. While accuracy is high, precision and recall are relatively lower, suggesting that the model may struggle with correctly identifying positive instances, although it performs well overall in terms of accuracy

# In[54]:


cm = confusion_matrix(y_test,y_pred)
print(cm)


# The confusion matrix provides a detailed breakdown of the performance of a classification model. Each cell in the matrix represents the count of instances according to their actual and predicted class labels. Here's what each part of the confusion matrix indicates:
# 
#     True Positives (TP): These are the instances that were predicted as positive by the model and are actually positive in the dataset.
# 
#     True Negatives (TN): These are the instances that were predicted as negative by the model and are actually negative in the dataset.
# 
#     False Positives (FP): These are the instances that were predicted as positive by the model but are actually negative in the dataset. Also known as Type I errors.
# 
#     False Negatives (FN): These are the instances that were predicted as negative by the model but are actually positive in the dataset. Also known as Type II errors.
# 
# In the context, confusion matrix:
# 
#     1526 instances were correctly predicted as negative (True Negatives).
#     69 instances were incorrectly predicted as positive when they were actually negative (False Positives).
#     308 instances were incorrectly predicted as negative when they were actually positive (False Negatives).
#     97 instances were correctly predicted as positive (True Positives).
# 
# The confusion matrix provides a detailed view of the model's performance, allowing  to assess its ability to correctly classify instances into their respective classes.

# ## Cross Validation

# In[55]:


from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator= classifier_lr,X=x_train,y=y_train,cv=10)

print("Accuracy is {:.2f} %".format(accuracies.mean()*100))
print("Accuracy is {:.2f} %".format(accuracies.std()*100))


# ## Random Forest

# In[56]:


from sklearn.ensemble import RandomForestClassifier
classifier_rf = RandomForestClassifier(random_state = 0)
classifier_rf.fit(x_train,y_train)


# In[57]:


y_pred = classifier_rf.predict(x_test)


# In[58]:


from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


# In[59]:


acc = accuracy_score(y_test,y_pred)
f1 = f1_score(y_test,y_pred)
prec = precision_score(y_test,y_pred)
rec = recall_score(y_test,y_pred)


# In[60]:


model_results = pd.DataFrame([['Random_forest',acc,f1,prec,rec]],
                            columns=['Model','Accuracy','F1','precision','Recall'])


# In[61]:


result = result.append(model_results,ignore_index = True)


# In[62]:


result


#     The numbers in the table represent performance metrics that evaluate the effectiveness of a machine learning model, specifically a Random Forest classifier in this case. Here's what each number signifies:
# 
#     Accuracy (0.8675): This number indicates that the Random Forest classifier correctly predicted the class label for approximately 86.75% of the instances in the dataset. In other words, it made the correct prediction for about 86.75% of the data points.
# 
#     F1 Score (0.610866): The F1 score is the harmonic mean of precision and recall. It balances both precision and recall, providing a single score that reflects the model's ability to make accurate positive predictions while minimizing false positives and false negatives. A higher F1 score indicates better performance.
# 
#     Precision (0.753623): Precision measures the accuracy of positive predictions made by the model. This number indicates that out of all the instances predicted as positive by the model, approximately 75.36% were actually positive.
# 
#     Recall (0.513580): Recall, also known as sensitivity or true positive rate, measures the model's ability to correctly identify positive instances out of all actual positive instances. This number indicates that the model identified approximately 51.36% of all actual positive instances.

# In[63]:


cm = confusion_matrix(y_test,y_pred)
print(cm)


# ## Cross Validation

# In[64]:


from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=classifier_rf,X=x_train,y=y_train,cv=10)
print('Accuracy is {:.2f} %'.format(accuracies.mean()*100))
print('Standard Deviation is {:.2f} %'.format(accuracies.std()*100))


# ## XGBoost

# In[65]:


import warnings

# To ignore all warnings
warnings.filterwarnings("ignore")


# In[66]:


get_ipython().system('pip install xgboost')


# In[67]:



from xgboost import XGBClassifier
classifier_xgb = XGBClassifier()
warnings.filterwarnings('ignore', message='.*missing.*')
classifier_xgb.fit(x_train, y_train)


# In[68]:


y_pred = classifier_xgb.predict(x_test)


# In[69]:


acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)


# In[70]:


model_results = pd.DataFrame([['XGBoost classifier',acc,f1,prec,rec]],
                            columns=['Model','Accuracy','F1','precision','Recall'])


# In[71]:


result = result.append(model_results,ignore_index=True)
result


# In[72]:


cm = confusion_matrix(y_test,y_pred)
print(cm)


# In[73]:


from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=classifier_xgb,X=x_train,y=y_train,cv=10)
print('Accuracy is {:.2f} %'.format(accuracies.mean()*100))
print('Standard Deviation is {:.2f} %'.format(accuracies.std()*100))


# In[74]:


from sklearn.model_selection import RandomizedSearchCV


# In[75]:


Parameters = {
    'learning_rate':[0.05, 0.1, 0.15, 0.20, 0.25, 0.30],
    'max_depth':[3, 4, 5 , 6, 7, 8 , 10, 12, 15],
    'min_child_weight':[1, 3, 5, 7],
    'gamma':[0.0, 0.1, 0.2, 0.3, 0.4],
    'colsample_bytree':[0.3, 0.4, 0.5, 0.7]
}


# In[76]:


Parameters


# In[77]:


randomized_search = RandomizedSearchCV(estimator=classifier_xgb,param_distributions=Parameters,n_iter=5,n_jobs=-1,scoring='r2',
                                      cv=5,verbose=3)


# In[78]:


randomized_search.fit(x_train,y_train)


# In[79]:


randomized_search.best_estimator_


# In[80]:


randomized_search.best_params_


# In[81]:


randomized_search.best_score_


# ## Final Model (XGBoost)

# In[82]:


from xgboost import XGBClassifier
classifier = XGBClassifier(base_score=None, booster=None, callbacks=None,
              colsample_bylevel=None, colsample_bynode=None,
              colsample_bytree=0.7, device=None, early_stopping_rounds=None,
              enable_categorical=False, eval_metric=None, feature_types=None,
              gamma=0.4, grow_policy=None, importance_type=None,
              interaction_constraints=None, learning_rate=0.1, max_bin=None,
              max_cat_threshold=None, max_cat_to_onehot=None,
              max_delta_step=None, max_depth=7, max_leaves=None,
              min_child_weight=7, missing=np.nan, monotone_constraints=None,
              multi_strategy=None, n_estimators=None, n_jobs=None,
              num_parallel_tree=None, random_state=None)
classifier.fit(x_train, y_train)


# In[83]:


y_pred = classifier.predict(x_test)
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)

model_results = pd.DataFrame([['Final XGBoost',acc,f1,prec,rec]],
                            columns=['Model','Accuracy','F1','precision','Recall'])

result = result.append(model_results,ignore_index=True)
result


# In[84]:


cm = confusion_matrix(y_test,y_pred)
print(cm)


# ### Cross Validation

# In[85]:


# cross validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=classifier, X=x_train, y=y_train, cv=10)

print("Accuracy is {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation is {:.2f} %".format(accuracies.std()*100))


# ## Predicting a single observation

# In[86]:


data.head()


# In[87]:


data.shape


# In[88]:


single_obs = [[12345678,650,48,3,160000.52,1,0,0,50000,0,0,1]]


# In[89]:


single_obs


# In[90]:


classifier.predict(sc.transform(single_obs))

