#!/usr/bin/env python
# coding: utf-8

import requests

# URL where the Flask app is hosted
url = 'http://localhost:9696/predict'

# Client information that you want to score
client = {"name": "Example Anime", "genre": "Action, Adventure", "type": "TV", "episodes": 24, "members": 200000}

# Make a POST request to the Flask app
response = requests.post(url, json=client).json()

# Printing the whole response
print(response)


#Output {'predicted_rating': 7.778192482489309}