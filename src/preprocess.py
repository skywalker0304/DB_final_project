import pandas as pd
import numpy as np

from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler, StandardScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_absolute_error
from sklearn.base import BaseEstimator

from typing import List, Tuple

import matplotlib.pyplot as plt
import seaborn as sn

################# impute missing values #####################3

# replace all NaN to 0
def data_replace_NaN_to_0(data: pd.DataFrame):
    missing_cols = [col for col in data.columns if data[col].isnull().any()]
    data[missing_cols] = data[missing_cols].fillna(0)
    return data

# replace all NaN to "type" ('mean' / 'median' / 'most_frequent' / 'constant')
def data_replace_NaN_with_type(data: pd.DataFrame, type: str):
    missing_cols = [col for col in data.columns if data[col].isnull().any()]
    imputer = SimpleImputer(strategy=type)
    data[missing_cols] = imputer.fit_transform(data[missing_cols])
    return data

# replace all NaN with k-Nearest Neighbors
def data_replace_NaN_with_KNN(data: pd.DataFrame, k: int):
    missing_cols = [col for col in data.columns if data[col].isnull().any()]
    imputer = KNNImputer(n_neighbors=k)
    data[missing_cols] = imputer.fit_transform(data[missing_cols])
    return data

####################      Data Scaling       ####################
def min_max_scaling(data: pd.DataFrame, col: List, min: int, max: int):
    scaler = MinMaxScaler(feature_range=(min, max))
    scaled_col = scaler.fit_transform(data[col])
    data[col] = scaled_col
    return data

def standard_scaling(data: pd.DataFrame, col: List):
    scaler = StandardScaler()
    scaled_col = scaler.fit_transform(data[col])
    data[col] = scaled_col
    return data
