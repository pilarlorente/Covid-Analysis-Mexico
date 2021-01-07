#!/usr/bin/env python
# coding: utf-8

# In[1]:


##### Import packages
# Basic packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Modelling packages

from sklearn.impute import SimpleImputer
from imblearn.under_sampling import RandomUnderSampler
#metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report
#models
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Other packages
from datetime import datetime
from collections import Counter

# To avoid warnings
import warnings
warnings.filterwarnings("ignore")


# In[2]:


##### Import data

df_cov = pd.read_csv("201114COVID19MEXICO.csv", encoding = "ISO-8859-1") # Covid-19 data


# ## PREPROCESSING DATA
# 

# In[4]:


##### Create a new column with the time difference between been positive in COVID-19 and die
# Value 0 if the person doesn't die

df_cov['FECHA_SINTOMAS'] = df_cov['FECHA_SINTOMAS'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
df_cov['FECHA_DEF'] = df_cov['FECHA_DEF'].replace('9999-99-99', '2001-01-01')

df_cov['FECHA_DEF'] = df_cov['FECHA_DEF'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
df_cov['DIFERENCIA'] = df_cov['FECHA_DEF'].sub(df_cov['FECHA_SINTOMAS'], axis=0)

df_cov['DIFERENCIA'] = df_cov['DIFERENCIA'] / np.timedelta64(1, 'D')
df_cov.loc[df_cov['DIFERENCIA']<0,'DIFERENCIA'] = 0


# In[5]:


# Value 2 or 98 on diseases columns means that the person doesn't have the disease
#Create a new column with 0= NO DISEASE and  1= disease

ill_name = ['DIABETES','EPOC','ASMA','INMUSUPR','HIPERTENSION','OTRA_COM','CARDIOVASCULAR','OBESIDAD','RENAL_CRONICA']
df_cov[ill_name] = df_cov[ill_name].replace([2,98],0)

df_cov['n_ENFERMEDADES'] = df_cov[ill_name].sum(axis = 1)


# In[6]:


##### Create a new column (target) with boolean value: 0 if the person doesn't die, 1 otherwise

df_cov.loc[df_cov['DIFERENCIA']==0,'MORTALIDAD'] = 0
df_cov.loc[df_cov['DIFERENCIA']!=0,'MORTALIDAD'] = 1


# In[7]:


##### Replacing missing values with NaNs

df_cov.replace([97,98,99],np.nan,inplace=True)
df_cov.isnull().sum()/len(df_cov)*100


# In[8]:


##### Deleting columns with NaNs > 0.8
df_cov.drop(columns=['INTUBADO', 'MIGRANTE', 'UCI'], inplace = True)


# In[ ]:


#SimpleImputer for missing values
imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
df_cov_imputer = imputer.fit_transform(df_cov)


# In[ ]:


df_cov = pd.DataFrame(df_cov_imputer, columns = df_cov.columns)


# In[ ]:


features = list(df_cov)
remove = ['FECHA_ACTUALIZACION', 'ID_REGISTRO', 'FECHA_INGRESO','FECHA_SINTOMAS','FECHA_DEF','HABLA_LENGUA_INDIG','CLASIFICACION_FINAL',
          'PAIS_NACIONALIDAD','RESULTADO_LAB','PAIS_ORIGEN','NACIONALIDAD','EMBARAZO','DIFERENCIA','MORTALIDAD']

for col in remove:
    features.remove(col)


# ## MODELS

# In[ ]:


X = df_cov[features].values.astype('int')
y = df_cov['MORTALIDAD'].values.astype('int')


# In[ ]:


#Hold out validation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, stratify=y)


# ## Random Forest

# In[ ]:


##### Random Forest

clf = RandomForestClassifier()
clf.fit(X_train, y_train)

yhat = clf.predict(X_test)
print('Accuracy score: \n',accuracy_score(y_test,yhat))

#classification report
print(classification_report(y_test, yhat, digits=3))

#confusion matrix
plot_confusion_matrix(clf, X_test, y_test,
                             display_labels=[0,1],
                             cmap=plt.cm.Blues,
                             normalize="true")
plt.title('Confussion matrix on how the \n model can predict mortality')
plt.show()


# In[ ]:


#Feature Importance

features_importance = clf.feature_importances_
features_array = np.array(features)
features_array_ordered = features_array[(features_importance).argsort()[::-1]]
features_array_ordered

plt.figure(figsize=(16,10))
sns.barplot(y = features_array, x = features_importance, orient='h', order=features_array_ordered[:50])

plt.show()


# ## Logistic Regression

# In[ ]:


clf = LogisticRegression(solver='liblinear')
clf.fit(X_train, y_train)

yhat = clf.predict(X_test)
print('Accuracy score: \n',accuracy_score(y_test,yhat))

#classification report
print(classification_report(y_test, yhat, digits=3))

#confusion matrix
plot_confusion_matrix(clf, X_test, y_test,
                             display_labels=[0,1],
                             cmap=plt.cm.Blues,
                             normalize="true")
plt.title('Confussion matrix on how the \n model can predict mortality')
plt.show()


# In[ ]:


#Feature Importance

features_importance = clf.feature_importances_
features_array = np.array(features)
features_array_ordered = features_array[(features_importance).argsort()[::-1]]
features_array_ordered

plt.figure(figsize=(16,10))
sns.barplot(y = features_array, x = features_importance, orient='h', order=features_array_ordered[:50])

plt.show()


# ## BALANCING CLASSES
# Undersampling method (80% of major class)

# In[ ]:


undersampling = RandomUnderSampler(sampling_strategy=0.8) 
X_balance, y_balance = undersampling.fit_resample(X, y)
Counter(y_balance)


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X_balance, y_balance, test_size=0.4)


# In[ ]:


##### Random Forest with undersampling

clf = RandomForestClassifier()
clf.fit(X_train, y_train)

yhat = clf.predict(X_test)
print('Accuracy score: \n',accuracy_score(y_test,yhat))

#classification report
print(classification_report(y_test, yhat, digits=3))

#confusion matrix
plot_confusion_matrix(clf, X_test, y_test,
                             display_labels=[0,1],
                             cmap=plt.cm.Blues,
                             normalize="true")
plt.title('Confussion matrix on how the \n model can predict mortality')
plt.show()


# In[ ]:


##### Logistic Regression with undersampling
clf = LogisticRegression(solver='liblinear')
clf.fit(X_train, y_train)

yhat = clf.predict(X_test)
print('Accuracy score: \n',accuracy_score(y_test,yhat))

#classification report
print(classification_report(y_test, yhat, digits=3))

#confusion matrix
plot_confusion_matrix(clf, X_test, y_test,
                             display_labels=[0,1],
                             cmap=plt.cm.Blues,
                             normalize="true")
plt.title('Confussion matrix on how the \n model can predict mortality')
plt.show()

