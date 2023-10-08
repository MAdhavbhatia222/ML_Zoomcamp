#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

import seaborn as sns
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score


# In[3]:


df = pd.read_csv('data.csv')

df = df[['Make', 'Model', 'Year', 'Engine HP',
       'Engine Cylinders', 'Transmission Type','Vehicle Style',
       'highway MPG','city mpg','MSRP']]
df = df.rename(columns={'MSRP':'price'})
df = df.fillna(0)
df.columns = df.columns.str.lower().str.replace(' ', '_')

string_columns = list(df.dtypes[df.dtypes == 'object'].index)

for col in string_columns:
    df[col] = df[col].str.lower().str.replace(' ', '_')
    
df['above_average'] = df['price'] > df['price'].mean()
df['above_average'] = df['above_average'].astype(int)


# In[4]:


df_train_full, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_train_full, test_size=0.25, random_state=1)

y_train = df_train.above_average.values
y_val = df_val.above_average.values

del df_train['above_average']
del df_val['above_average']


# In[5]:


categorical = ['make', 'model', 'transmission_type','vehicle_style',]
numerical = ['year', 'engine_hp', 'engine_cylinders', 'highway_mpg', 'city_mpg']


# In[6]:


def tpr_fpr_dataframe(y_val, y_pred):
    scores = []

    thresholds = np.linspace(0, 1, 101)

    for t in thresholds:
        tp = ((y_pred >= t) & (y_val == 1)).sum()
        fp = ((y_pred >= t) & (y_val == 0)).sum()
        fn = ((y_pred < t) & (y_val == 1)).sum()
        tn = ((y_pred < t) & (y_val == 0)).sum()

        scores.append((t, tp, fp, fn, tn))

    df_scores = pd.DataFrame(scores)
    df_scores.columns = ['threshold', 'tp', 'fp', 'fn', 'tn']

    df_scores['tpr'] = df_scores.tp / (df_scores.tp + df_scores.fn)
    df_scores['fpr'] = df_scores.fp / (df_scores.fp + df_scores.tn)

    return df_scores


# ### For each numerical variable, use it as score and compute AUC with the above_average variable
# ### Use the training dataset for that

# In[7]:


for each_numeric in numerical:
    print(each_numeric)
    train_dict = df_train[[each_numeric]].to_dict(orient='records')
    
    dv = DictVectorizer(sparse=False)
    dv.fit(train_dict)
    
    X_train = dv.transform(train_dict)
    model = LogisticRegression(solver='liblinear', random_state=1)
    model.fit(X_train, y_train)
    
    y_pred = model.predict_proba(X_train)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_train, y_pred)
    print(auc(fpr, tpr))
# year
# 0.31244850719531714
# engine_hp
# 0.9171031265539011
# engine_cylinders
# 0.766116490165669
# highway_mpg
# 0.6330587871772013
# city_mpg
# 0.6734244643245233

# ### Apply one-hot-encoding using DictVectorizer and train the logistic regression with these parameters:
# ### LogisticRegression(solver='liblinear', C=1.0, max_iter=1000)
# ### What's the AUC of this model on the validation dataset? (round to 3 digits)

# In[8]:


train_dict = df_train[categorical + numerical].to_dict(orient='records')

dv = DictVectorizer(sparse=False)
dv.fit(train_dict)

X_train = dv.transform(train_dict)
model = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000, random_state=1)
model.fit(X_train, y_train)


val_dict = df_val[categorical + numerical].to_dict(orient='records')
X_val = dv.transform(val_dict)
y_pred = model.predict_proba(X_val)[:, 1]
fpr, tpr, thresholds = roc_curve(y_val, y_pred)
print(round(auc(fpr, tpr),3))
# 0.976

# ## Precision and recall

# In[9]:


scores = []

thresholds = np.linspace(0, 1, 101)

for t in thresholds: #B
    tp = ((y_pred >= t) & (y_val == 1)).sum()
    fp = ((y_pred >= t) & (y_val == 0)).sum()
    fn = ((y_pred < t) & (y_val == 1)).sum()
    tn = ((y_pred < t) & (y_val == 0)).sum()
    scores.append((t, tp, fp, fn, tn))

df_scores = pd.DataFrame(scores)
df_scores.columns = ['threshold', 'tp', 'fp', 'fn', 'tn']


# In[10]:


df_scores['precision'] = df_scores.tp / (df_scores.tp + df_scores.fp)
df_scores['recall'] = df_scores.tp / (df_scores.tp + df_scores.fn)


# In[11]:


plt.figure(figsize=(6, 4))

plt.plot(df_scores.threshold, df_scores.precision, color='black', linestyle='solid', label='TPR')
plt.plot(df_scores.threshold, df_scores.recall, color='black', linestyle='dashed', label='FPR')
plt.legend()

plt.xticks(np.linspace(0, 1, 11))
plt.yticks(np.linspace(0, 1, 11))

plt.xlabel('Thresholds')
plt.title('precision and recall')


plt.show()




# In[12]:


df_scores['f1_score'] = 2 * (df_scores.precision * df_scores.recall  / (df_scores.precision + df_scores.recall))


# ### At which threshold F1 is maximal?

# In[13]:


df_scores[df_scores['f1_score'] == max(df_scores['f1_score'])]
# 	threshold	tp	fp	fn	tn	precision	recall	f1_score
# 51	0.51	568	69	86	1660	0.89168	0.868502	0.879938

# ## K-fold cross-validation

# In[16]:


def train(df, y):
    cat = df[categorical + numerical].to_dict(orient='records')
    
    dv = DictVectorizer(sparse=False)
    dv.fit(cat)

    X = dv.transform(cat)

    model = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000)
    model.fit(X, y)

    return dv, model


def predict(df, dv, model):
    cat = df[categorical + numerical].to_dict(orient='records')
    
    X = dv.transform(cat)

    y_pred = model.predict_proba(X)[:, 1]

    return y_pred


# In[17]:


kfold = KFold(n_splits=5, shuffle=True, random_state=1)


# In[20]:


aucs = []

for train_idx, val_idx in kfold.split(df_train_full):
    df_train = df_train_full.iloc[train_idx]
    y_train = df_train.above_average.values

    df_val = df_train_full.iloc[val_idx]
    y_val = df_val.above_average.values

    dv, model = train(df_train, y_train)
    y_pred = predict(df_val, dv, model)

    rocauc = roc_auc_score(y_val, y_pred)
    aucs.append(rocauc)


# In[21]:


np.array(aucs).round(3)

# array([0.978, 0.981, 0.978, 0.98 , 0.985])

# In[22]:


print('auc = %0.3f ± %0.3f' % (np.mean(aucs), np.std(aucs)))
# auc = 0.980 ± 0.003

# Tuning the parameter `C`

# In[23]:


def train(df, y, C=1.0):
    cat = df[categorical + numerical].to_dict(orient='records')
    
    dv = DictVectorizer(sparse=False)
    dv.fit(cat)

    X = dv.transform(cat)

    model = LogisticRegression(solver='liblinear', C=C, max_iter=1000)
    model.fit(X, y)

    return dv, model


# In[25]:


nfolds = 5
kfold = KFold(n_splits=nfolds, shuffle=True, random_state=1)

for C in [0.01, 0.1, 0.5, 10]:
    aucs = []

    for train_idx, val_idx in kfold.split(df_train_full):
        df_train = df_train_full.iloc[train_idx]
        df_val = df_train_full.iloc[val_idx]

        y_train = df_train.above_average.values
        y_val = df_val.above_average.values

        dv, model = train(df_train, y_train, C=C)
        y_pred = predict(df_val, dv, model)
        
        auc = roc_auc_score(y_val, y_pred)
        aucs.append(auc)

    print('C=%s, auc = %0.3f ± %0.3f' % (C, np.mean(aucs), np.std(aucs)))


# C=0.01, auc = 0.952 ± 0.002
# C=0.1, auc = 0.972 ± 0.002
# C=0.5, auc = 0.980 ± 0.003
# C=10, auc = 0.982 ± 0.003




