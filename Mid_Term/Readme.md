# Airbnb Price Prediction in Seattle

## Overview
This repository is dedicated to the machine learning project that tackles the challenge of predicting Airbnb listing prices in Seattle. Utilizing a comprehensive dataset from Kaggle, the project aims to build a predictive model that can estimate the cost of renting a property based on its characteristics.

## Dataset
The core of this project is a dataset available on [Kaggle](https://www.kaggle.com/datasets/airbnb/seattle?select=listings.csv), which details a wide array of features pertaining to Airbnb listings in Seattle. These features encompass listing identifiers, names, descriptions, host information, geographical coordinates, property details, and price points, among others.

## Problem Statement
Setting the right price for an Airbnb listing is a complex task that affects both the host's earnings and the guest's decision-making. This project seeks to develop a model that can predict the optimal listing price based on property characteristics and market data, thereby benefiting both hosts and guests alike.

## Solution Approach
To address the price prediction challenge, we employ a Linear Regression model. This statistical method is selected for its simplicity and effectiveness in understanding the linear relationships between independent variables (listing features) and the dependent variable (price).

## Repository Structure
- `train.py`: The Python script that encapsulates the training process of the Linear Regression model.
- `requirements.txt`: A file listing all the necessary Python libraries required to execute the project's code.
- `listings.csv`: The dataset file that provides the input data for model training.

## Getting Started
To get started with this project:
1. Clone the repository to your local machine.
2. Install the required dependencies by running `pip install -r requirements.txt`.
3. Execute the `train.py` script to train the model on the dataset.

## License
This project is released under the MIT License. For more information, see the `LICENSE` file in the repository.

## Contributors
This project welcomes contributions from the community. If you have suggestions or improvements, feel free to open an issue or a pull request.

## Acknowledgements
Special thanks to Kaggle and the dataset authors for providing the data that made this project possible.

---
