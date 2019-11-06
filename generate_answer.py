# %%
import pickle

with open("RandomForestClf_nEs100_rS42.pkl", 'rb') as file:
        model = pickle.load(file)