import pandas as pd
import numpy as np

from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler, StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_absolute_error
from sklearn.base import BaseEstimator

from typing import List, Tuple

import matplotlib.pyplot as plt
import seaborn as sn

def encode(data: pd.DataFrame):
    string_cols = data.select_dtypes(include=['object']).columns.tolist()
    encoder = OrdinalEncoder()
    data[string_cols] = encoder.fit_transform(data[string_cols])


def preprocess_data(data : pd.DataFrame, preprocessing_methods:list, features:list):
    # Apply data preprocessing methods
    preprocessed_data = data.copy()

    #encode string columns
    encode(preprocessed_data)

    if 'remove_duplicates' in preprocessing_methods:
        preprocessed_data.drop_duplicates(inplace=True)

    if 'standard_scaling' in preprocessing_methods:
        preprocessed_data = standard_scaling(preprocessed_data, features)
    elif 'min_max_scaling' in preprocessing_methods:
        preprocessed_data = min_max_scaling(preprocessed_data, features, 0, 1)


    if 'impute_mean' in preprocessing_methods:
        preprocessed_data = data_replace_NaN_with_type(preprocessed_data, 'mean')
    elif 'impute_median' in preprocessing_methods:
        preprocessed_data = data_replace_NaN_with_type(preprocessed_data, 'median')
    elif 'impute_most_frequent' in preprocessing_methods:
        preprocessed_data = data_replace_NaN_with_type(preprocessed_data, 'most_frequent')

    return preprocessed_data

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
