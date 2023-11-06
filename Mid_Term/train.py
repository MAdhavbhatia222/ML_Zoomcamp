
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_squared_error
import pickle
import json
from sklearn.preprocessing import LabelEncoder
from datetime import datetime

# Define the columns to be used based on the user's code
columns_filtered = [
    'host_since', 'host_response_time', 'host_response_rate',
    'host_acceptance_rate', 'host_is_superhost', 'host_total_listings_count',
    'host_identity_verified', 'neighbourhood', 'neighbourhood_cleansed',
    'neighbourhood_group_cleansed', 'city', 'state', 'zipcode', 'market',
    'country_code', 'country', 'property_type', 'room_type', 'accommodates',
    'bathrooms', 'bedrooms', 'beds', 'bed_type', 'amenities', 'price'
]

# Data cleaning and feature engineering functions

def json_to_list(json_str):
    try:
        json_str = json_str.strip("{}").replace('"', "")
        if json_str == "":
            return []
        else:
            return json_str.split(',')
    except json.JSONDecodeError:
        return []

def preprocess_features(df):
    # Convert price to float and log-transform
    df['price'] = df['price'].str.replace('[\$,]', '', regex=True).astype(float)
    df['price_log'] = np.log1p(df['price'])

    # Convert host_since to datetime and extract year
    df['host_since'] = pd.to_datetime(df['host_since'])
    df['host_since_year'] = df['host_since'].dt.year

    # Cleaning and encoding string columns
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].str.lower().str.replace(' ', '_')

    # Convert host_response_rate and host_acceptance_rate to numeric
    df['host_response_rate'] = pd.to_numeric(df['host_response_rate'].str.replace('%', ''), errors='coerce')
    df['host_acceptance_rate'] = pd.to_numeric(df['host_acceptance_rate'].str.replace('%', ''), errors='coerce')

    # Fill missing values
    for col in ['host_response_rate', 'host_acceptance_rate', 'beds', 'bedrooms', 'bathrooms']:
        df[col] = df[col].fillna(0)

    # Convert amenities to list and create binary columns
    df['amenities_list'] = df['amenities'].apply(json_to_list)
    for amenity in set.union(*df['amenities_list'].apply(set)):
        df['amenity_' + amenity.lower()] = df['amenities_list'].apply(lambda x: 1 if amenity in x else 0)

    df = df.drop('host_since', axis=1)
    df = df.dropna()
    # Encode categorical variables
    le = LabelEncoder()
    for col in df.select_dtypes(include='object').columns:
        df[col] = le.fit_transform(df[col].astype(str))

    return df

# Load the dataset and filter columns
try:
    df = pd.read_csv('listings.csv', usecols=columns_filtered)
except ValueError as e:
    print(f"Error loading the CSV file: {e}")
    # Handle the error, perhaps exit the script or provide a message

# Preprocess the data using the previously defined functions
df = preprocess_features(df)

# Split the data into train and test sets
df_train_full, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_train_full, test_size=0.25, random_state=1)
y_train = df_train.price_log.values
y_val = df_val.price_log.values
y_test = df_test.price_log.values

# Remove the target variable from the training and validation sets
del df_train['price_log']
del df_val['price_log']
del df_test['price_log']

# DictVectorizer is used for encoding categorical features as one-hot numeric array
dv = DictVectorizer(sparse=False)

# Prepare the training and testing data
train_dict = df_train.to_dict(orient='records')
X_train = dv.fit_transform(train_dict)

test_dict = df_test.to_dict(orient='records')
X_test = dv.transform(test_dict)

# Initialize the Ridge Regression model
model = Ridge(alpha=0.01, solver='sag', random_state=1)

# Train the model
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Calculate the Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f'Root Mean Squared Error: {rmse}')

# Save the model and DictVectorizer to files
model_path = 'ridge_model.bin'

with open(model_path, 'wb') as f_out:
    pickle.dump((dv,model), f_out)
