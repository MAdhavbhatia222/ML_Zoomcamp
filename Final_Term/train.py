import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_squared_error
import pickle
from sklearn.preprocessing import LabelEncoder

# Define the columns to be used based on the user's code
columns_filtered = ['NAME', 'GENRE', 'TYPE', 'EPISODES', 'ANIME_RATING', 'MEMBERS']

def preprocess_features(df):
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    
    # Convert ratings to float and log-transform
    df['anime_rating'] = pd.to_numeric(df['anime_rating'])
    df['anime_rating_log'] = np.log1p(df['anime_rating'])

    # Cleaning and encoding string columns
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].str.lower().str.replace(' ', '_')
    df['episodes'] = df['episodes'].replace('unknown', '0', regex=True).astype(float)
    df['members'] = pd.to_numeric(df['members'])
    df['episodes'] = pd.to_numeric(df['episodes'])

    # Convert genre to list and create binary columns for important genres only
    important_genres = ['shounen', 'drama', 'action', 'romance', 'fantasy', 'dementia', 'sci-fi', 'kids']
    df['genre_list'] = df['genre'].apply(lambda x: x.split(","))
    df['genre_list'] = df['genre_list'].apply(lambda x: [genre.lower() for genre in x])
    for genre in important_genres:
        df['genre_' + genre] = df['genre_list'].apply(lambda x: 1 if genre in x else 0)

    df = df.dropna()
    df = df.drop(columns=['anime_rating', 'genre', 'genre_list'])
    
    # Encode categorical variables
    le = LabelEncoder()
    for col in df.select_dtypes(include='object').columns:
        df[col] = le.fit_transform(df[col].astype(str))

    return df

# Load the dataset and filter columns
try:
    df = pd.read_csv('Anime_Ratings_Data.csv', usecols=columns_filtered)
except ValueError as e:
    print(f"Error loading the CSV file: {e}")

# Preprocess the data using the previously defined functions
df = preprocess_features(df)

# Split the data into train and test sets
df_train_full, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_train_full, test_size=0.25, random_state=1)
y_train = df_train.anime_rating_log.values
y_val = df_val.anime_rating_log.values
y_test = df_test.anime_rating_log.values

# Remove the target variable from the training and validation sets
del df_train['anime_rating_log']
del df_val['anime_rating_log']
del df_test['anime_rating_log']

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
    pickle.dump((dv, model), f_out)