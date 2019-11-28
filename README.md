# Titanic Competition

[![Kaggle Score](https://img.shields.io/badge/score-78.47%25-blue)](https://www.kaggle.com/franpogoda)

Titanic is a well known Kaggle competition you can find [here](https://www.kaggle.com/c/titanic/overview). Its description says "Start here! Predict survival on the Titanic and get familiar with ML basics" so I started here.

While messing around with the data, books and some kernels I tried to create an automatic process or some sort of framework for future machine learning projects. 

## Machine Learning project structure

- `data/` - a directory with all the provided data and generated data in the process of dividing and enhancing the dataset
- `clean/` - a directory with a Jupyter Notebook `process_data.ipynb` where I divide the data into training and validation sets, visualize it, clean and modify it to match the model needs and a Python file `prepare_data.py` which contains all functions needed for the automatic process of preparing data for models
- `model/` - a directory with a Jupyter Notebook `compare_models.ipynb` where I compare different models using prepared data and tune them to achieve best possible scores, Python class in `models.py` file which stores all models I used in the comparisons just to have them in one place and the `build_model.py` file which contains all functions needed to build and fit the best tuned model I found
- `generate_answer.py` - a Python script used to generate answers using the provided test data and the built model

All of these files visualize some sort of my workflow of dealing with Machine Learning projects. If you can see some bug or bad habit or you have some better practice than me, share it! I'd love to fix it and get better for the upcoming projects.

The structure of this project and the presented workflow is the main thing I wanted to share with you. In case you want to dive a little bit deeper into the implementation and my problem analysis feel free to read the notebooks and look at the scripts! 
