#!/usr/bin/env python
# coding: utf-8

import requests

# URL where the Flask app is hosted
url = 'http://localhost:9696/predict'

# Client information that you want to score
client = {"job": "retired", "duration": 445, "poutcome": "success"}

# Make a POST request to the Flask app
response = requests.post(url, json=client).json()

# Printing the whole response
print(response)

# You can customize this part to take the necessary actions based on the prediction
if response['prediction'] == 'positive':
    print('Taking action for positive prediction.')
else:
    print('Taking action for negative prediction.')
