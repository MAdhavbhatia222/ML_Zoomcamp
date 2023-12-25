---

# Anime Rating Predictor

## Overview
This repository contains the code for a machine learning model that predicts anime ratings based on various features. The application is built in Python, using Flask for the web framework, and is containerized using Docker for easy deployment and scaling.

## Dataset
The dataset used for training the machine learning model is sourced from Kaggle: [Anime Recommendations Database](https://www.kaggle.com/datasets/CooperUnion/anime-recommendations-database/data). It includes features like anime name, genre, type, number of episodes, and member count, which are used to predict the anime rating.

## Problem Statement
The primary objective is to develop a predictive model that can accurately predict the rating of an anime based on its characteristics. This can be used by anime recommendation systems or for analytical purposes.

## Solution Approach
The solution involves preprocessing the data, training a Ridge Regression model, and deploying the model as a Flask web application. The web application accepts JSON input containing anime features and returns a predicted rating. The entire application is containerized using Docker.

## Repository Structure
- `Train.py`: Python script for training the machine learning model.
- `Predict.py`: Flask web application for serving the trained model.
- `Dockerfile`: Instructions for Docker to build the application container.
- `Pipfile` and `Pipfile.lock`: Dependency management files.
- `requirements.txt`: List of Python package dependencies.
- `ridge_model.bin`: Serialized machine learning model.
- `README.md`: Documentation of the project (this file).

## Getting Started
To get the application running, follow these steps:

1. **Clone the Repository**:
   ```
   git clone [repository URL]
   cd [repository directory]
   ```

2. **Build the Docker Image**:
   ```
   docker build -t anime-rating-predictor .
   ```

3. **Run the Docker Container**:
   ```
   docker run -p 9696:9696 anime-rating-predictor
   ```

   This will start the Flask application inside the Docker container and expose it on port 9696 of your localhost.

4. **Making Predictions**:
   - Use `curl` or Postman to send POST requests to `http://localhost:9696/predict` with the anime data in JSON format.
   - Alternatively, use the provided `prediction_test.py` script to test the predictions.

5. **Example Request**:
   ```json
   {
     "name": "Example Anime",
     "genre": "Action, Adventure",
     "type": "TV",
     "episodes": 24,
     "members": 200000
   }
   ```

---
