# %%
from sklearn.ensemble import RandomForestClassifier

class Models:
    def __init__(self):
        self.dict = {
            "RandomForestClf_nEs100_rS42": (
                RandomForestClassifier(n_estimators=100, random_state=42), 
                "RandomForestClf_nEs100_rS42.pkl"
            )
        }


# %%
