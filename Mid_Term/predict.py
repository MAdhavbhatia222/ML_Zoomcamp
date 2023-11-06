import pickle

from flask import Flask
from flask import request
from flask import jsonify
import numpy as np

# The file where you saved your DictVectorizer and model
model_file = 'ridge_model.bin'

# Load the DictVectorizer and model
with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

app = Flask('price_prediction')

@app.route('/predict', methods=['POST'])
def predict():
    listing = request.get_json()

    X = dv.transform([listing])
    y_pred = model.predict(X)[0]  # Since this is a regression model, we use predict instead of predict_proba

    # You can directly return the predicted price or apply an inverse transformation if the price was log-transformed
    predicted_price = np.expm1(y_pred)  # Apply expm1 if the price was log-transformed

    result = {
        'predicted_price': float(predicted_price)
    }

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
