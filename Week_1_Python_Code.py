##############
import pandas as pd
print(pd.__version__)
####### '1.5.3'
housing = pd.read_csv("Path/to_housing/housing.csv")
print(len(housing))
####### 20640

print(housing.isna().sum())

# longitude               0
# latitude                0
# housing_median_age      0
# total_rooms             0
# total_bedrooms        207
# population              0
# households              0
# median_income           0
# median_house_value      0
# ocean_proximity         0

print(housing.ocean_proximity.unique())
# array(['NEAR BAY', '<1H OCEAN', 'INLAND', 'NEAR OCEAN', 'ISLAND'],
#       dtype=object)

print(housing[housing.ocean_proximity=='NEAR BAY'].median_house_value.mean())
# 259212.31179039303

print(housing.total_bedrooms.mean())
# 537.8705525375618
housing.total_bedrooms = housing.total_bedrooms.fillna(housing.total_bedrooms.mean())
print(housing.total_bedrooms.mean())
# 537.8705525375617

X = housing[housing.ocean_proximity=='ISLAND'][['housing_median_age','total_rooms','total_bedrooms']]
import numpy as np
X_Arr = np.array(X)
XTX=(X_Arr.T).dot(X_Arr)
X_Arr_inv = np.linalg.inv(XTX)
print(X_Arr_inv.dot(X_Arr.T).dot(y))
# array([23.12330961, -1.48124183,  5.69922946])






