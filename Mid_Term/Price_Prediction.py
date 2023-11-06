#!/usr/bin/env python
# coding: utf-8

# Data from https://www.kaggle.com/datasets/airbnb/seattle?select=listings.csv

# In[1]:


get_ipython().system('head listings.csv')


# In[2]:


import pandas as pd
import numpy as np

import seaborn as sns
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import json

from IPython.display import display
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from sklearn.linear_model import Ridge
from sklearn.feature_extraction import DictVectorizer

from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')


# In[3]:


df = pd.read_csv('listings.csv')


# In[4]:


print(len(df))


# In[5]:


display(df.head().T)


# In[6]:


print(df.dtypes)


# In[7]:


print(df.columns)


# In[8]:


columns_filtered = ['host_since', 'host_response_time',
       'host_response_rate', 
'host_acceptance_rate', 'host_is_superhost', 'host_total_listings_count','host_identity_verified', 'neighbourhood', 'neighbourhood_cleansed',
       'neighbourhood_group_cleansed', 'city', 'state', 'zipcode', 'market',
      'country_code', 'country','property_type', 'room_type', 'accommodates',
       'bathrooms', 'bedrooms', 'beds', 'bed_type', 'amenities',
       'price']


# In[9]:


df = df[columns_filtered]
display(df.head().T)


# ## Data Cleaning

# In[10]:


df.columns = df.columns.str.lower().str.replace(' ', '_')

string_columns = list(df.dtypes[df.dtypes == 'object'].index)

for col in string_columns:
    df[col] = df[col].str.lower().str.replace(' ', '_')


# In[11]:


display(df.head().T)


# In[12]:


# Function to convert JSON string to list
def json_to_list(json_str):
    try:
        # Remove the curly braces and quotes, then split by comma
        # Assuming that the amenities are in a set-like structure `{amenity1,amenity2,...}`
        json_str = json_str.strip("{}")
        # Split the string by comma not within quotes
        lst = [s.strip().strip('"') for s in json_str.split(',') if s.strip()]
        return lst
    except json.JSONDecodeError:
        # In case of decoding error, return an empty list or None
        return []


# In[13]:


# Apply the function to the 'amenities' column
df['amenities_list'] = df['amenities'].apply(json_to_list)

# Convert the 'price' column to a numeric type (float)
df['price'] = df['price'].replace('[\$,]', '', regex=True).astype(float)


# In[14]:


# Convert the 'host_since' column to datetime
df['host_since'] = pd.to_datetime(df['host_since'])

# Extract the year from the datetime column
df['year'] = df['host_since'].dt.year
df['age'] = max(df['year']) - df.year


# In[15]:


df['host_response_rate'] = pd.to_numeric(df['host_response_rate'].str.replace('%', ''), errors='coerce')


# In[16]:


df['host_acceptance_rate'] = pd.to_numeric(df['host_acceptance_rate'].str.replace('%', ''), errors='coerce')


# In[17]:


# Convert all amenities to lowercase to avoid case sensitivity issues
df['amenities_list'] = df['amenities_list'].apply(lambda amenities: [amenity.lower() for amenity in amenities])

# Get the unique list of all possible amenities
all_amenities = set(amenity for sublist in df['amenities_list'] for amenity in sublist)

# Now create binary columns for each amenity
for amenity in all_amenities:
    # Create a new column for each amenity
    df[amenity] = df['amenities_list'].apply(lambda x: 1 if amenity in x else 0)


# ## Exploratory data analysis

# In[18]:


plt.figure(figsize=(6, 4))

sns.histplot(df.price.values, bins=40, color='black', alpha=1)
plt.ylabel('Frequency')
plt.xlabel('price')
plt.title('Distribution of price')

plt.show()


# In[19]:


log_price = np.log1p(df.price)

plt.figure(figsize=(6, 4))

sns.histplot(log_price, bins=40, color='black', alpha=1)
plt.ylabel('Frequency')
plt.xlabel('Log(Price + 1)')
plt.title('Distribution of prices after log tranformation')

plt.show()


# In[20]:


print(df.isnull().sum()[df.isnull().sum()>0])


# In[21]:


for eachcolumn in ['host_response_time','host_response_rate','host_acceptance_rate','age','beds',
               'bedrooms','bathrooms','host_total_listings_count']:
    df[eachcolumn] = df[eachcolumn].fillna(0)


# In[22]:


df['neighbourhood'] = df['neighbourhood'].fillna('unknown')
df['zipcode'] = df['zipcode'].fillna('unknown')


# In[23]:


print(df.isnull().sum()[df.isnull().sum()>0])


# In[24]:


df = df.dropna()


# In[25]:


print(df.isnull().sum()[df.isnull().sum()>0])


# In[26]:


print(df.dtypes)


# In[ ]:





# ## Feature Importance

# In[27]:


categorical = ['host_response_time', 'host_is_superhost', 'host_identity_verified',
       'neighbourhood', 'neighbourhood_cleansed',
       'neighbourhood_group_cleansed', 'city', 'state', 'zipcode', 'market', 'country_code', 'property_type',
       'room_type', 'bed_type','24-hour_check-in','air_conditioning','breakfast','buzzer/wireless_intercom',
       'cable_tv','carbon_monoxide_detector','cat(s)','dog(s)','doorman','dryer','elevator_in_building',
       'essentials','family/kid_friendly','fire_extinguisher','first_aid_kit','free_parking_on_premises',
       'gym','hair_dryer','hangers','heating','hot_tub','indoor_fireplace','internet','iron','kitchen',
       'laptop_friendly_workspace','lock_on_bedroom_door','other_pet(s)','pets_allowed',
       'pets_live_on_this_property','pool','safety_card','shampoo','smoke_detector','smoking_allowed',
       'suitable_for_events','tv','washer','washer_/_dryer','wheelchair_accessible','wireless_internet']
numerical = ['host_acceptance_rate','host_response_rate','age','host_total_listings_count', 'accommodates', 'bathrooms', 'bedrooms',
       'beds']


# In[28]:


# Pre-process the categorical columns
label_encoders = {}
for column in categorical:
    le = LabelEncoder()
    # Fill NaN with a placeholder string and convert to type str to ensure all data is of the same type
    df[column] = df[column].fillna('Unknown').astype(str)
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

def calculate_mi(series):
    return round(mutual_info_score(series, df.price), 2)

# Now apply the calculate_mi function
df_mi = df[categorical].apply(calculate_mi)
df_mi = df_mi.sort_values(ascending=False).to_frame(name='MI')

display(df_mi.head(10))
display(df_mi.tail())


# In[29]:


df[numerical].corrwith(df.price).to_frame('correlation')


# In[30]:


display(df[numerical].corr())


# ### Linear Regression

# In[31]:


df['price_log'] = np.log1p(df['price'])
df_train_full, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_train_full, test_size=0.25, random_state=1)
y_train = df_train.price_log.values
y_val = df_val.price_log.values
y_test = df_test.price_log.values

del df_train['price_log']
del df_val['price_log']
del df_test['price_log']

del df_train['price']
del df_val['price']
del df_test['price']



# In[32]:


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


# ## Decision Tree

# In[33]:


dict_train = df_train[categorical + numerical].to_dict(orient='records')
dict_val = df_val[categorical + numerical].to_dict(orient='records')
dv = DictVectorizer(sparse=True)

X_train = dv.fit_transform(dict_train)
X_val = dv.transform(dict_val)


# In[34]:


dt = DecisionTreeRegressor()
dt.fit(X_train, y_train)


# In[35]:


y_pred = dt.predict(X_train)
print("On Training :" , np.sqrt(mean_squared_error(y_train, y_pred)))
y_pred = dt.predict(X_val)
print("On Testing :",np.sqrt(mean_squared_error(y_val, y_pred)))


# In[36]:


dt = DecisionTreeRegressor(max_depth=1)
dt.fit(X_train, y_train)

y_pred = dt.predict(X_train)
RMSE = np.sqrt(mean_squared_error(y_train, y_pred))
print('On Training RMSE: %.3f' % RMSE)

y_pred = dt.predict(X_val)
RMSE = np.sqrt(mean_squared_error(y_val, y_pred))
print('On Testing RMSE: %.3f' % RMSE)


# ## Random forest

# In[37]:


rf = RandomForestRegressor(n_estimators=10, random_state=1,n_jobs=-1)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_val)
print(np.sqrt(mean_squared_error(y_val, y_pred)))


# In[38]:


RMSES = []

for i in range(10, 201, 10):
    rf = RandomForestRegressor(n_estimators=i, random_state=1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_val)
    RMSE = np.sqrt(mean_squared_error(y_val, y_pred))
    print('%s -> %.3f' % (i, RMSE))
    RMSES.append(RMSE)


# Tuinnig the `max_depth` parameter:

# In[39]:


all_RMSES = {}

for depth in [40, 80, 100]:
    print('depth: %s' % depth)
    RMSES = []

    for i in range(10, 201, 10):
        rf = RandomForestRegressor(n_estimators=i, max_depth=depth, random_state=1,n_jobs=-1)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_val)
        RMSE =  np.sqrt(mean_squared_error(y_val, y_pred))
        print('%s -> %.3f' % (i, RMSE))
        RMSES.append(RMSE)
    
    all_RMSES[depth] = RMSES
    print()
    


# In[40]:


rf = RandomForestRegressor(n_estimators=80 , max_depth=40, random_state=1,n_jobs=-1)
rf.fit(X_train, y_train)


# In[41]:


y_pred_rf = rf.predict(X_val)
RMSE =  np.sqrt(mean_squared_error(y_val, y_pred))
print('%.3f' % (RMSE))


# In[42]:


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

# In[43]:


cleaned_feature_names = dv.feature_names_


# In[44]:


dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=cleaned_feature_names)
dval = xgb.DMatrix(X_val, label=y_val, feature_names=cleaned_feature_names)


# In[45]:


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
print(np.sqrt(mean_squared_error(y_val, y_pred)))


# In[46]:


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

