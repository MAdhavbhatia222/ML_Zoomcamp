import pickle
import numpy as np
from flask import Flask, request, jsonify

# The file where you saved your DictVectorizer and model
model_file = 'ridge_model.bin'

# Load the DictVectorizer and model
with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

app = Flask('rating_prediction')

def preprocess_input(anime):
    # Convert string columns to lowercase and replace spaces with underscores
    for col in ['name', 'type', 'genre']:
        if col in anime:
            anime[col] = anime[col].lower().replace(' ', '_')

    # Convert 'genre' to a list and create binary columns for important genres only
    important_genres = ['shounen', 'drama', 'action', 'romance', 'fantasy', 'dementia', 'sci-fi', 'kids']
    if 'genre' in anime:
        genre_list = anime['genre'].split(',')
        for genre in important_genres:
            anime['genre_' + genre] = 1 if genre in genre_list else 0

    return anime

@app.route('/predict', methods=['POST'])
def predict():
    anime = request.get_json()
    anime = preprocess_input(anime)

    X = dv.transform([anime])
    y_pred = model.predict(X)[0]

    # Apply inverse log transformation if necessary
    predicted_rating = np.expm1(y_pred)

    result = {
        'predicted_rating': float(predicted_rating)
    }

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)