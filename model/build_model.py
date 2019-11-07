# %%
from clean.prepare_data import get_train_data
import pickle
import os
import numpy as np
from model.models import Models


MODEL_PATH = "model/bin/"

def save_model(model, model_file_name: str, model_file_path: str=MODEL_PATH):
    full_file_name = os.path.join(model_file_path, model_file_name)
    with open(full_file_name, 'wb') as file:  
        pickle.dump(model, file)


def load_model(model_file_name: str, model_file_path: str=MODEL_PATH):
    full_file_name = os.path.join(model_file_path, model_file_name)
    with open(full_file_name, 'rb') as file:
            model = pickle.load(file)
            return model


def build_model(model, model_file_name: str, data_file_name: str):
    X, y = get_train_data(file_name=data_file_name)
    model.fit(X, list(y))
    save_model(model, model_file_name)
    return model


# Change this function to tune the model
def build_tuned_model():
    model_name = "RandomForestClf_nEs100_rS42"

    model, file_name = Models().dict[model_name]

    return build_model(
        model=model,
        model_file_name=file_name, 
        data_file_name="train.csv"
    )


# %%
