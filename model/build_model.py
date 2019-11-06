# %%
from clean.prepare_data import get_train_data
from sklearn.ensemble import RandomForestClassifier
import pickle
import os
import numpy as np

MODEL_PATH = "model/bin/"

def save_model(model, model_file_name: str, model_file_path=MODEL_PATH):
    full_file_name = os.path.join(model_file_path, model_file_name)
    with open(full_file_name, 'wb') as file:  
        pickle.dump(model, file)

def build_model():
    X, y = get_train_data()
    forest_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    forest_clf.fit(X, list(y))
    save_model(forest_clf, "RandomForestClf_nEs100_rS42.pkl")

    


# %%
