#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

import seaborn as sns
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


df = pd.read_csv('housing_alexy.csv')
len(df)


# In[4]:


df.head()


# In[ ]:


median_house_value 


# In[5]:


df.columns = df.columns.str.lower().str.replace(' ', '_')

string_columns = list(df.dtypes[df.dtypes == 'object'].index)

print(string_columns)

for col in string_columns:
    df[col] = df[col].str.lower().str.replace(' ', '_')


# In[6]:


df.head()


# In[7]:


for col in df.columns:
    print(col)
    print(df[col].unique()[:5])
    print(df[col].nunique())
    print()


# ## Exploratory data analysis

# In[8]:


plt.figure(figsize=(6, 4))

sns.histplot(df.median_house_value, bins=40, color='black', alpha=1)
plt.ylabel('Frequency')
plt.xlabel('median_house_value')
plt.title('Distribution of median_house_value')

plt.show()


# In[9]:


# plt.figure(figsize=(6, 4))

# sns.histplot(df.msrp[df.msrp < 100000], bins=40, color='black', alpha=1)
# plt.ylabel('Frequency')
# plt.xlabel('Price')
# plt.title('Distribution of prices')

# plt.show()


# In[10]:


# log_price = np.log1p(df.msrp)

# plt.figure(figsize=(6, 4))

# sns.histplot(log_price, bins=40, color='black', alpha=1)
# plt.ylabel('Frequency')
# plt.xlabel('Log(Price + 1)')
# plt.title('Distribution of prices after log tranformation')

# plt.show()


# In[18]:


df.ocean_proximity.unique()


# In[19]:


df_subset = df[(df.ocean_proximity == '<1h_ocean')
              |
               (df.ocean_proximity == 'inland')
              ]


# In[20]:


df_subset


# In[22]:


df_subset.isnull().sum()


# In[25]:


np.median(df_subset.population)


# ## Validation framework

# In[27]:


np.random.seed(42)

n = len(df_subset)

n_val = int(0.2 * n)
n_test = int(0.2 * n)
n_train = n - (n_val + n_test)

idx = np.arange(n)
np.random.shuffle(idx)

df_shuffled = df_subset.iloc[idx]

df_train = df_shuffled.iloc[:n_train].copy()
df_val = df_shuffled.iloc[n_train:n_train+n_val].copy()
df_test = df_shuffled.iloc[n_train+n_val:].copy()


# In[28]:


y_train_orig = df_train.median_house_value.values
y_val_orig = df_val.median_house_value.values
y_test_orig = df_test.median_house_value.values

y_train = np.log1p(df_train.median_house_value.values)
y_val = np.log1p(df_val.median_house_value.values)
y_test = np.log1p(df_test.median_house_value.values)

del df_train['median_house_value']
del df_val['median_house_value']
del df_test['median_house_value']


# ## Linear Regression

# In[29]:


def train_linear_regression(X, y):
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])

    XTX = X.T.dot(X)
    XTX_inv = np.linalg.inv(XTX)
    w = XTX_inv.dot(X.T).dot(y)
    
    return w[0], w[1:]


# In[31]:


df_train.head(3)


# In[32]:


df_train.columns


# ## Baseline solution

# In[47]:


base = ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'population', 'households', 'median_income']
Null_col = ['total_bedrooms']


# In[48]:


def prepare_X(df):
    df_num = df[base+Null_col]
    df_num = df_num.fillna(0)
    X = df_num.values
    return X

def prepare_X_mean(df):
    df_null = df[Null_col]
    mean_null = df_null.mean()
    df_null = df_null.fillna(mean_null)
    df_base = df[base]
    df_num = pd.concat([df_base,df_null],axis=1)
    X = df_num.values
    return X


# In[49]:


X_train = prepare_X(df_train)
w_0, w = train_linear_regression(X_train, y_train)


# In[50]:


y_pred = w_0 + X_train.dot(w)


# In[51]:


plt.figure(figsize=(6, 4))

sns.histplot(y_train, label='target', color='#222222', alpha=0.6, bins=40)
sns.histplot(y_pred, label='prediction', color='#aaaaaa', alpha=0.8, bins=40)

plt.legend()

plt.ylabel('Frequency')
plt.xlabel('Log(Price + 1)')
plt.title('Predictions vs actual distribution')

plt.show()


# In[52]:


def rmse(y, y_pred):
    error = y_pred - y
    mse = (error ** 2).mean()
    return np.sqrt(mse)


# In[53]:


rmse(y_train, y_pred)


# In[54]:


X_val = prepare_X(df_val)
y_pred = w_0 + X_val.dot(w)


# In[55]:


rmse(y_val, y_pred)


# In[56]:


X_train = prepare_X_mean(df_train)
w_0, w = train_linear_regression(X_train, y_train)
y_pred = w_0 + X_train.dot(w)
rmse(y_train, y_pred)


# In[57]:


X_val = prepare_X_mean(df_val)
y_pred = w_0 + X_val.dot(w)
rmse(y_val, y_pred)


# ## Regularization

# In[58]:


def train_linear_regression_reg(X, y, r=0.0):
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])

    XTX = X.T.dot(X)
    reg = r * np.eye(XTX.shape[0])
    XTX = XTX + reg

    XTX_inv = np.linalg.inv(XTX)
    w = XTX_inv.dot(X.T).dot(y)
    
    return w[0], w[1:]


# In[59]:


X_train = prepare_X(df_train)


# In[62]:


for r in [0, 0.000001, 0.0001, 0.001, 0.01, 0.1, 1, 5, 10]:
    w_0, w = train_linear_regression_reg(X_train, y_train, r=r)
    print('%5s, %.2f' % (r, w_0))


# In[63]:


X_train = prepare_X(df_train)
w_0, w = train_linear_regression_reg(X_train, y_train, r=0)

y_pred = w_0 + X_train.dot(w)
print('train', rmse(y_train, y_pred))

X_val = prepare_X(df_val)
y_pred = w_0 + X_val.dot(w)
print('val', round(rmse(y_val, y_pred),2))


# In[70]:


X_train = prepare_X(df_train)
X_val = prepare_X(df_val)

for r in [0, 0.000001, 0.0001, 0.001, 0.01, 0.1, 1, 5, 10]:
    w_0, w = train_linear_regression_reg(X_train, y_train, r=r)
    y_pred = w_0 + X_val.dot(w)
    print('%6s' %r, round(rmse(y_val, y_pred),8))


# In[71]:


X_train = prepare_X(df_train)
w_0, w = train_linear_regression_reg(X_train, y_train, r=0)

X_val = prepare_X(df_val)
y_pred = w_0 + X_val.dot(w)
print('validation:', rmse(y_val, y_pred))

X_test = prepare_X(df_test)
y_pred = w_0 + X_test.dot(w)
print('test:', rmse(y_test, y_pred))


# In[79]:


rmse_scores_seed = []


# In[80]:


for seed_val in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
    np.random.seed(seed_val)

    n = len(df_subset)

    n_val = int(0.2 * n)
    n_test = int(0.2 * n)
    n_train = n - (n_val + n_test)

    idx = np.arange(n)
    np.random.shuffle(idx)

    df_shuffled = df_subset.iloc[idx]

    df_train = df_shuffled.iloc[:n_train].copy()
    df_val = df_shuffled.iloc[n_train:n_train+n_val].copy()
    df_test = df_shuffled.iloc[n_train+n_val:].copy()

    y_train_orig = df_train.median_house_value.values
    y_val_orig = df_val.median_house_value.values
    y_test_orig = df_test.median_house_value.values

    y_train = np.log1p(df_train.median_house_value.values)
    y_val = np.log1p(df_val.median_house_value.values)
    y_test = np.log1p(df_test.median_house_value.values)

    del df_train['median_house_value']
    del df_val['median_house_value']
    del df_test['median_house_value']

    X_train = prepare_X(df_train)
    w_0, w = train_linear_regression_reg(X_train, y_train, r=0)

    y_pred = w_0 + X_train.dot(w)
    print('train', rmse(y_train, y_pred))

    X_val = prepare_X(df_val)
    y_pred = w_0 + X_val.dot(w)
    print('val', rmse(y_val, y_pred))
    
    rmse_scores_seed.append(rmse(y_val, y_pred))
    print(rmse_scores_seed)


# In[81]:


round(np.std(rmse_scores_seed),3)


# In[92]:





# In[95]:


np.random.seed(9)

n = len(df_subset)

n_val = int(0.2 * n)
n_test = int(0.2 * n)
n_train = n - (n_val + n_test)

idx = np.arange(n)
np.random.shuffle(idx)

df_shuffled = df_subset.iloc[idx]

df_train = df_shuffled.iloc[:n_train].copy()
df_val = df_shuffled.iloc[n_train:n_train+n_val].copy()
df_train = pd.concat([df_train,df_val])

df_test = df_shuffled.iloc[n_train+n_val:].copy()


y_train_orig = df_train.median_house_value.values
y_val_orig = df_val.median_house_value.values
y_train_orig = np.append(y_train_orig, y_val_orig)

y_test_orig = df_test.median_house_value.values


y_train = np.log1p(df_train.median_house_value.values)

y_test = np.log1p(df_test.median_house_value.values)

del df_train['median_house_value']
del df_val['median_house_value']
del df_test['median_house_value']

X_train = prepare_X(df_train)
w_0, w = train_linear_regression_reg(X_train, y_train, r=0.001)

y_pred = w_0 + X_train.dot(w)
print('train', rmse(y_train, y_pred))

X_test = prepare_X(df_test)
y_pred = w_0 + X_test.dot(w)
print('test', rmse(y_test, y_pred))


# In[ ]:




