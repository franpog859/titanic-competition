# %%
import os
import pandas as pd
import numpy as np
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Tuple


class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names]


# Inspired from stackoverflow.com/questions/25239958
class MostFrequentImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.most_frequent_ = pd.Series([X[c].value_counts().index[0] for c in X],
                                        index=X.columns)
        return self
    def transform(self, X, y=None):
        return X.fillna(self.most_frequent_)


script_directory = os.getcwd()
DATA_DIRECTORY = "data/"
full_data_path = os.path.join(script_directory, DATA_DIRECTORY)
DATA_PATH = full_data_path
FILE_NAME = "train.csv"

def load_data(data_path=DATA_PATH, file_name=FILE_NAME) -> pd.DataFrame:
    csv_path = os.path.join(data_path, file_name)
    return pd.read_csv(csv_path)


def get_independent_variable(data: pd.DataFrame) -> np.ndarray:
    numeric_pipeline = Pipeline([
        ("select_numeric", DataFrameSelector(["Age", "SibSp", "Parch", "Fare"])),
        ("imputer", SimpleImputer(strategy="median")),
    ])
    numeric_pipeline.fit_transform(data)

    category_pipeline = Pipeline([
        ("select_cat", DataFrameSelector(["Pclass", "Sex", "Embarked"])),
        ("imputer", MostFrequentImputer()),
        ("cat_encoder", OneHotEncoder(sparse=False)),
    ])
    category_pipeline.fit_transform(data)

    preprocess_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", numeric_pipeline),
        ("cat_pipeline", category_pipeline),
    ])
    
    processed_data = preprocess_pipeline.fit_transform(data)
    return processed_data


def get_train_data(data_path=DATA_PATH, file_name=FILE_NAME) -> Tuple[np.ndarray, np.ndarray]:
    data = load_data(data_path, file_name)
    X = get_independent_variable(data)
    y = data.values[:,1]
    return (X, y)
    

# %%
