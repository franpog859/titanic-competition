# %%
import pandas as pd
from model.build_model import load_model
from clean.prepare_data import load_data
from clean.prepare_data import get_independent_variable

ANSWERS_FILE = "answers.csv"

def save_answer(data: dict, answers_file_name: str=ANSWERS_FILE):
    output = pd.DataFrame(data=data)
    output.to_csv(answers_file_name, index=False)


# Change this function to prepare answers file
def main():
    print("Loading data...")
    data = load_data("test.csv")
    X = get_independent_variable(data)

    print("Predicting...")
    model = load_model("RandomForestClf_nEs100_rS42.pkl")
    y_predicted = model.predict(X)

    print("Saving answers...")
    output_data = {
        "PassengerId": data["PassengerId"],
        "Survived": y_predicted
    }
    save_answer(output_data, ANSWERS_FILE)
    print("Succesfully saved answers in {} file!".format(ANSWERS_FILE))


if __name__ == "__main__":
    main()
    

# %%
