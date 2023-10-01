#!/usr/bin/env python
# coding: utf-8

# In[28]:


import pandas as pd
import numpy as np

import seaborn as sns
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[29]:


df = pd.read_csv('data.csv')


# In[30]:


len(df)


# ## Initial data preparation

# In[31]:


df.head()


# In[32]:


df = df[['Make', 'Model', 'Year', 'Engine HP',
       'Engine Cylinders', 'Transmission Type','Vehicle Style',
       'highway MPG','city mpg','MSRP']]
df = df.rename(columns={'MSRP':'price'})
df = df.fillna(0)


# In[33]:


df.dtypes


# In[34]:


df.columns = df.columns.str.lower().str.replace(' ', '_')

string_columns = list(df.dtypes[df.dtypes == 'object'].index)

for col in string_columns:
    df[col] = df[col].str.lower().str.replace(' ', '_')


# In[35]:


df.transmission_type.value_counts()
# automatic           8266
# manual              2935
# automated_manual     626
# direct_drive          68
# unknown               19
# Name: transmission_type, dtype: int64

# In[36]:


categorical = ['make', 'model', 'transmission_type','vehicle_style',]
numerical = ['year', 'engine_hp', 'engine_cylinders', 'highway_mpg', 'city_mpg']


# In[37]:


display(df[numerical].corrwith(df['year']).to_frame('correlation_year'))
display(df[numerical].corrwith(df['engine_cylinders']).to_frame('correlation_engine_cylinders'))
display(df[numerical].corrwith(df['city_mpg']).to_frame('correlation_city_mpg'))

# correlation_year
# year	1.000000
# engine_hp	0.338714
# engine_cylinders	-0.040708
# highway_mpg	0.258240
# city_mpg	0.198171
# correlation_engine_cylinders
# year	-0.040708
# engine_hp	0.774851
# engine_cylinders	1.000000
# highway_mpg	-0.614541
# city_mpg	-0.587306
# correlation_city_mpg
# year	0.198171
# engine_hp	-0.424918
# engine_cylinders	-0.587306
# highway_mpg	0.886829
# city_mpg	1.000000
# In[38]:


df['above_average'] = df['price'] > df['price'].mean()
df['above_average'] = df['above_average'].astype(int)


# In[39]:


from sklearn.model_selection import train_test_split
df_train_full, df_test = train_test_split(df, test_size=0.2, random_state=42)
df_train, df_val = train_test_split(df_train_full, test_size=0.25, random_state=42)
y_train = df_train.above_average.values
y_val = df_val.above_average.values
y_test = df_test.above_average.values


# In[43]:


del df_train['above_average']
del df_val['above_average']
del df_test['above_average']
del df_train['price']
del df_val['price']
del df_test['price']


# ## Exploratory data analysis

# In[44]:


df_train_full.isnull().sum()


# In[45]:


df_train_full.above_average.value_counts()


# In[46]:


global_mean = df_train_full.above_average.mean()
round(global_mean, 3)


# In[47]:


df_train_full[categorical].nunique()


# ## Feature importance

# In[48]:


from IPython.display import display


# In[49]:


for col in categorical:
    df_group = df_train_full.groupby(by=col).above_average.agg(['mean'])
    df_group['diff'] = df_group['mean'] - global_mean
    df_group['risk'] = df_group['mean'] / global_mean
    display(df_group)


# In[50]:


from sklearn.metrics import mutual_info_score


# In[51]:


def calculate_mi(series):
    return round(mutual_info_score(series, df_train_full.above_average),2)

df_mi = df_train_full[categorical].apply(calculate_mi)
df_mi = df_mi.sort_values(ascending=False).to_frame(name='MI')


display(df_mi.head())
display(df_mi.tail())

# MI
# model	0.46
# make	0.24
# vehicle_style	0.08
# transmission_type	0.02
# In[52]:


df_train_full[numerical].corrwith(df_train_full.above_average).to_frame('correlation')

# 	correlation
# year	0.318753
# engine_hp	0.660670
# engine_cylinders	0.453162
# highway_mpg	-0.134484
# city_mpg	-0.157912
# In[53]:


df_train_full.groupby(by='above_average')[numerical].mean()
# 	year	engine_hp	engine_cylinders	highway_mpg	city_mpg
# above_average					
# 0	2008.926882	202.652981	5.107355	27.363412	20.522124
# 1	2014.300986	365.903336	6.927976	24.734268	17.524261

# ## One-hot encoding

# In[95]:


from sklearn.feature_extraction import DictVectorizer
train_dict = df_train[categorical + numerical].to_dict(orient='records')
dv = DictVectorizer(sparse=False)
dv.fit(train_dict)
X_train = dv.transform(train_dict)


# In[97]:


train_dict[0]


# In[98]:


dv.fit(train_dict)


# In[100]:


X_train.shape


# In[101]:


dv.get_feature_names_out()


# ## Training logistic regression

# In[102]:


from sklearn.linear_model import LogisticRegression


# In[103]:


model = LogisticRegression(solver='liblinear', C=10, max_iter=1000, random_state=42)
model.fit(X_train, y_train)


# In[104]:


val_dict = df_val[categorical + numerical].to_dict(orient='records')
X_val = dv.transform(val_dict)


# In[105]:


model.predict_proba(X_val)


# In[106]:


y_pred = model.predict_proba(X_val)[:, 1]


# In[107]:


y_pred


# In[108]:


above_average = y_pred > 0.5


# In[109]:


(y_val == above_average).mean()

# 0.9433487201007134
# ## Model Subsests

# In[110]:


var_testing = ['year','engine_hp','transmission_type','city_mpg']


# In[111]:


subset_list=[]
for each in var_testing:
    subset = list(set(numerical).union(set(categorical)) - {each})
    subset_list.append(subset)


# In[114]:


for subset in subset_list:
    train_dict_small = df_train[subset].to_dict(orient='records')
    dv_small = DictVectorizer(sparse=False)
    dv_small.fit(train_dict_small)

    X_small_train = dv_small.transform(train_dict_small)

    dv_small.get_feature_names_out()
    model_small = LogisticRegression(solver='liblinear', random_state=1)
    model_small.fit(X_small_train, y_train)
#     print(model_small.intercept_[0])
#     print(dict(zip(dv_small.get_feature_names_out(), model_small.coef_[0].round(3))))
    val_dict_small = df_val[subset].to_dict(orient='records')
    X_small_val = dv_small.transform(val_dict_small)
    y_pred_small = model_small.predict_proba(X_small_val)[:, 1]
    above_avg_small = y_pred_small > 0.5
    print("Accuracy: ",(y_val == above_avg_small).mean())
    print("Difference :",0.9433487201007134 - ((y_val == above_avg_small).mean()))


# In[ ]:

# Accuracy:  0.9483843894250944
# Difference : -0.005035669324381042
# Accuracy:  0.9211078472513639
# Difference : 0.022240872849349502
# Accuracy:  0.9412505245488879
# Difference : 0.0020981955518254436
# Accuracy:  0.9467058329836341
# Difference : -0.003357112882920732



# In[ ]:





# ### Linear Regression

# In[127]:


from sklearn.model_selection import train_test_split
df['price_log'] = np.log1p(df['price'])
df_train_full, df_test = train_test_split(df, test_size=0.2, random_state=42)
df_train, df_val = train_test_split(df_train_full, test_size=0.25, random_state=42)
y_train = df_train.price_log.values
y_val = df_val.price_log.values
y_test = df_test.price_log.values

del df_train['price_log']
del df_val['price_log']
del df_test['price_log']
del df_train['above_average']
del df_val['above_average']
del df_test['above_average']
del df_train['price']
del df_val['price']
del df_test['price']


from sklearn.linear_model import Ridge
from sklearn.feature_extraction import DictVectorizer


# In[129]:


# List of alpha values to try
alphas = [0, 0.01, 0.1, 1, 10]
# Dictionary to store RMSE scores for each alpha
rmse_scores = {}

for alpha in alphas:
    # Create and fit the Ridge regression model
    model = Ridge(alpha=alpha, solver='sag', random_state=42)
    train_dict = df_train[categorical + numerical].to_dict(orient='records')
    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(train_dict)
    model.fit(X_train, y_train)
    
    
    test_dict = df_test[categorical + numerical].to_dict(orient='records')
    X_test = dv.transform(test_dict)
    # Predict on the test set
    y_pred = model.predict(X_test)
    
    # Calculate RMSE and round it to 3 decimal places
    rmse = round(np.sqrt(((y_pred - y_test) ** 2).mean()), 10)
    
    # Store RMSE in the dictionary
    rmse_scores[alpha] = rmse

# Display the RMSE scores for different alpha values
for alpha, rmse in rmse_scores.items():
    print(f"Alpha={alpha}: RMSE={rmse}")


# Alpha=0: RMSE=0.4863782356
# Alpha=0.01: RMSE=0.4863784353
# Alpha=0.1: RMSE=0.4863802316
# Alpha=1: RMSE=0.4863981913
# Alpha=10: RMSE=0.4865773808


