# %%
import os
import pandas as pd
import numpy as np
from typing import Tuple
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin


class NameToTitleConverter(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        for row in [X]:
            row['Title'] = row.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

            row['Title'] = row['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
 	            'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
            row['Title'] = row['Title'].replace('Mlle', 'Miss')
            row['Title'] = row['Title'].replace('Ms', 'Miss')
            row['Title'] = row['Title'].replace('Mme', 'Mrs')
            
            row['Title'] = row['Title'].fillna(0)
        X = X.drop(["Name"], axis=1)
        return X


class AgeBinner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        X['AgeBand'] = pd.cut(X['Age'], bins=[0, 5, 18, 30, 38, 50, 65, 74.3, 90])
        X = X.drop(["Age"], axis=1)
        return X


class FareBinner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        X['FareBand'] = pd.qcut(X['Fare'], 4)
        X = X.drop(["Fare"], axis=1)
        return X


class IsAloneAdder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        X['IsAlone'] = X['SibSp'] + X['Parch'] > 0
        X = X.drop(["SibSp"], axis=1)
        X = X.drop(["Parch"], axis=1)
        return X


class FamilySizeAdder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        X['FamilySize'] = X['SibSp'] + X['Parch'] + 1
        X = X.drop(["SibSp"], axis=1)
        X = X.drop(["Parch"], axis=1)
        return X


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
    cat_attribs = ["Pclass", "Sex", "Embarked", "Name", "Age", "Parch", "SibSp", "Fare"]
    category_pipeline = Pipeline([
        ("name_to_title", NameToTitleConverter()),
        ("age_binner", AgeBinner()),
        ("fare_binner", FareBinner()),
        ("is_alone_adder", IsAloneAdder()),
        ("imputer", MostFrequentImputer()),
        ("cat_encoder", OneHotEncoder(sparse=False)),
    ])

    num_attribs = ["SibSp", "Parch", "Fare", "Age"]
    numeric_pipeline = Pipeline([
        ("family_size_adder", FamilySizeAdder()),
        ("imputer", SimpleImputer(strategy="median")),
        ("std_scaler", StandardScaler()),
    ])

    full_pipeline = ColumnTransformer([
        ("cat", category_pipeline, cat_attribs),
        ("num", numeric_pipeline, num_attribs),
    ])

    return full_pipeline.fit_transform(data)


def get_train_data(file_name: str, data_path: str=DATA_PATH) -> Tuple[np.ndarray, np.ndarray]:
    data = load_data(file_name, data_path)
    X = get_independent_variable(data)
    y = data["Survived"] # This changes with different datasets
    return (X, y)
    

# %%
