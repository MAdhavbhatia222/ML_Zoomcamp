#!/usr/bin/env python
# coding: utf-8

# To download the dataset, run
# 
# ```
# wget https://raw.githubusercontent.com/alexeygrigorev/datasets/master/housing.csv
# ```

# In[1]:


get_ipython().system('head housing.csv')


# In[2]:


import pandas as pd
import numpy as np

import seaborn as sns
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Data preparation

# In[3]:


df = pd.read_csv('housing.csv')


# In[4]:


df.head()


# In[5]:


df.columns = df.columns.str.lower().str.replace(' ', '_')

string_columns = list(df.dtypes[df.dtypes == 'object'].index)

print(string_columns)

for col in string_columns:
    df[col] = df[col].str.lower().str.replace(' ', '_')


# In[6]:


df.columns = df.columns.str.lower()
df.head()


# In[7]:


df = df[(df.ocean_proximity == '<1h_ocean')
              |
               (df.ocean_proximity == 'inland')
              ]


# In[10]:


df = df.fillna(0)


# In[11]:


df.describe().round()


# In[12]:


df.isnull().sum()


# Now the stats are more meaninful

# Let's look at the target variable

# In[14]:


df.median_house_value.mean()


# Now we're ready to prepare the data for training:
# 
# * First, do train-validation-test split
# * Then, apply one-hot encoding to categorical features and get the feature matrix 

# In[15]:


from sklearn.model_selection import train_test_split


# In[16]:


df_train_full, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_train_full, test_size=0.25, random_state=1)


# In[17]:


y_train_orig = df_train.median_house_value.values
y_val_orig = df_val.median_house_value.values
y_test_orig = df_test.median_house_value.values

y_train = np.log1p(df_train.median_house_value.values)
y_val = np.log1p(df_val.median_house_value.values)
y_test = np.log1p(df_test.median_house_value.values)

del df_train['median_house_value']
del df_val['median_house_value']
del df_test['median_house_value']


# In[18]:


len(df_train), len(df_val), len(df_test)


# For OHE, we'll use `DictVectorizer`

# In[19]:


from sklearn.feature_extraction import DictVectorizer


# In[21]:


dict_train = df_train.fillna(0).to_dict(orient='records')
dict_val = df_val.fillna(0).to_dict(orient='records')


# In[22]:


dict_train[0]


# In[23]:


dv = DictVectorizer(sparse=True)

X_train = dv.fit_transform(dict_train)
X_val = dv.transform(dict_val)


# Now we're ready to train a model. We'll start with decision trees

# ## Decision trees
# 
# We'll use `DecisionTreeRegressor` and for evaluating the quality of our models, we'll use RMSE
# 

# In[26]:


from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error


# Let's fit the tree with default parameters

# In[27]:


dt = DecisionTreeRegressor()
dt.fit(X_train, y_train)


# To get the predictions (probabilities), we use `predict_proba`. Let's check AUC on train data:

# In[30]:


y_pred = dt.predict(X_train)
mean_squared_error(y_train, y_pred)


# And on validation:

# In[32]:


y_pred = dt.predict(X_val)
mean_squared_error(y_val, y_pred)


# That's a case of _overfitting_ - our model on the training data performs perfectly, but fails on validation
# 
# Let's change the depth parameter: restring the size of the tree to 2 levels:

# In[34]:


dt = DecisionTreeRegressor(max_depth=1)
dt.fit(X_train, y_train)

y_pred = dt.predict(X_train)
RMSE = mean_squared_error(y_train, y_pred)
print('train RMSE: %.3f' % RMSE)

y_pred = dt.predict(X_val)
RMSE = mean_squared_error(y_val, y_pred)
print('val RMSE: %.3f' % RMSE)


# In[37]:


print(dt.feature_importances_)
print(list(zip(dv.feature_names_, dt.feature_importances_)))
importances = list(zip(dv.feature_names_, dt.feature_importances_))

df_importance = pd.DataFrame(importances, columns=['feature', 'gain'])
df_importance = df_importance.sort_values(by='gain', ascending=False)
print(df_importance)
df_importance = df_importance[df_importance.gain > 0]
num = len(df_importance)
plt.barh(range(num), df_importance.gain[::-1])
plt.yticks(range(num), df_importance.feature[::-1])

plt.show()


# ## Random forest

# In[38]:


from sklearn.ensemble import RandomForestRegressor


# To fix this issue, let's set the seed

# In[39]:


rf = RandomForestRegressor(n_estimators=10, random_state=1,n_jobs=-1)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_val)
mean_squared_error(y_val, y_pred)


# Now we'll check how RMSE depends on the number of trees

# In[40]:


RMSES = []

for i in range(10, 201, 10):
    rf = RandomForestRegressor(n_estimators=i, random_state=1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_val)
    RMSE = mean_squared_error(y_val, y_pred)
    print('%s -> %.3f' % (i, RMSE))
    RMSES.append(RMSE)


# Tuninig the `max_depth` parameter:

# In[41]:


all_RMSES = {}

for depth in [10, 15, 20, 25]:
    print('depth: %s' % depth)
    RMSES = []

    for i in range(10, 201, 10):
        rf = RandomForestRegressor(n_estimators=i, max_depth=depth, random_state=1,n_jobs=-1)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_val)
        RMSE = mean_squared_error(y_val, y_pred)
        print('%s -> %.3f' % (i, RMSE))
        RMSES.append(RMSE)
    
    all_RMSES[depth] = RMSES
    print()
    


# Training the final model:

# In[42]:


rf = RandomForestRegressor(n_estimators=10, max_depth=20, random_state=1,n_jobs=-1)
rf.fit(X_train, y_train)


# In[43]:


y_pred_rf = rf.predict(X_val)
mean_squared_error(y_val, y_pred)


# In[44]:


print(rf.feature_importances_)
importances = list(zip(dv.feature_names_, rf.feature_importances_))

df_importance = pd.DataFrame(importances, columns=['feature', 'gain'])
df_importance = df_importance.sort_values(by='gain', ascending=False)
print(df_importance)
df_importance = df_importance[df_importance.gain > 0.01]
num = len(df_importance)
plt.barh(range(num), df_importance.gain[::-1])
plt.yticks(range(num), df_importance.feature[::-1])

plt.show()


# ## XGBoost

# In[45]:


import xgboost as xgb


# In[48]:


# Creating a function to clean the feature names
def clean_feature_names(feature_names):
    cleaned_feature_names = []
    for name in feature_names:
        cleaned_name = name.replace("[", "_").replace("]", "_").replace("<", "_")
        cleaned_feature_names.append(cleaned_name)
    return cleaned_feature_names

# Getting the cleaned feature names
cleaned_feature_names = clean_feature_names(dv.feature_names_)


# In[49]:


dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=cleaned_feature_names)
dval = xgb.DMatrix(X_val, label=y_val, feature_names=cleaned_feature_names)


# In[59]:


xgb_params = {
    'eta': 0.3,
    'max_depth': 6,
    'min_child_weight': 1,

    'objective': 'reg:squarederror',
    'nthread': 8,
    
    'seed': 1,
    'verbosity': 1,
}
model = xgb.train(xgb_params, dtrain, num_boost_round=100)
y_pred = model.predict(dval)
mean_squared_error(y_val, y_pred)


# In[60]:


xgb_params = {
    'eta': 0.1,
    'max_depth': 6,
    'min_child_weight': 1,

    'objective': 'reg:squarederror',
    'nthread': 8,
    
    'seed': 1,
    'verbosity': 1,
}
model = xgb.train(xgb_params, dtrain, num_boost_round=100)
y_pred = model.predict(dval)
mean_squared_error(y_val, y_pred)


# In[61]:


watchlist = [(dtrain, 'train'), (dval, 'val')]
xgb_params = {
    'eta': 0.3,
    'max_depth': 6,
    'min_child_weight': 1,

    'objective': 'reg:squarederror',
    'nthread': 8,
    'eval_metric': 'rmse',

    'seed': 1,
    'verbosity': 1,
}
model = xgb.train(xgb_params, dtrain,
                  num_boost_round=100,
                  evals=watchlist, verbose_eval=10)


# In[62]:


watchlist = [(dtrain, 'train'), (dval, 'val')]
xgb_params = {
    'eta': 0.1,
    'max_depth': 6,
    'min_child_weight': 1,

    'objective': 'reg:squarederror',
    'nthread': 8,
    'eval_metric': 'rmse',

    'seed': 1,
    'verbosity': 1,
}
model = xgb.train(xgb_params, dtrain,
                  num_boost_round=100,
                  evals=watchlist, verbose_eval=10)

