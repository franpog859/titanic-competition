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


class TitleAdder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
        for row in [X]:
            row['Title'] = row.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

            row['Title'] = row['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
 	            'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
            row['Title'] = row['Title'].replace('Mlle', 'Miss')
            row['Title'] = row['Title'].replace('Ms', 'Miss')
            row['Title'] = row['Title'].replace('Mme', 'Mrs')
            
            row['Title'] = row['Title'].map(title_mapping)
            row['Title'] = row['Title'].fillna(0)
        return X


class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names]


# Inspired by stackoverflow.com/questions/25239958
class MostFrequentImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.most_frequent_ = pd.Series(
            [X[c].value_counts().index[0] for c in X],
            index=X.columns
        )
        return self
    def transform(self, X, y=None):
        return X.fillna(self.most_frequent_)


DATA_PATH = "data/"

def load_data(file_name: str, data_path: str=DATA_PATH) -> pd.DataFrame:
    csv_path = os.path.join(data_path, file_name)
    return pd.read_csv(csv_path)


# Change this function to clean the data
def get_independent_variable(data: pd.DataFrame) -> np.ndarray:
    numeric_pipeline = Pipeline([
        ("add_title", TitleAdder()),
        ("select_numeric", DataFrameSelector(["Age", "SibSp", "Parch", "Fare", "Title"])),
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
    return preprocess_pipeline.fit_transform(data)


def get_train_data(file_name: str, data_path: str=DATA_PATH) -> Tuple[np.ndarray, np.ndarray]:
    data = load_data(file_name, data_path)
    X = get_independent_variable(data)
    y = data["Survived"] # This changes with different datasets
    return (X, y)
    

# %%
