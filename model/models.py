# %%
from sklearn import ensemble, linear_model, neighbors, svm, tree, neural_network, discriminant_analysis, naive_bayes, gaussian_process
from typing import Tuple
from sklearn import model_selection

class Models:
    def __init__(self, _dict: dict={
            #Ensemble Methods
            "AdaBoostClassifier": (ensemble.AdaBoostClassifier()),
            "BaggingClassifier": (ensemble.BaggingClassifier()),
            "ExtraTreesClassifier": (ensemble.ExtraTreesClassifier()),
            "GradientBoostingClassifier": (ensemble.GradientBoostingClassifier()),
            "RandomForrestClassifier": (ensemble.RandomForestClassifier(random_state=42)),

            #Gaussian Processes
            "GaussianProcessClassifier": (gaussian_process.GaussianProcessClassifier()),

            #Linear Model
            "LogisticRegression": (linear_model.LogisticRegression(solver='liblinear', multi_class='ovr')),
            "LogisticRegressionCV": (linear_model.LogisticRegressionCV()),
            "PassiveAggressiveClassifier": (linear_model.PassiveAggressiveClassifier()),
            "RidgeClassifierCV": (linear_model.RidgeClassifierCV()),
            "SGDClassifier": (linear_model.SGDClassifier()),
            "Perceptron": (linear_model.Perceptron()),

            #Discriminant Analysis
            "LinearDiscriminantAnalysis": (discriminant_analysis.LinearDiscriminantAnalysis()),

            #Naive Bayes
            "BernoulliNB": (naive_bayes.BernoulliNB()),
            "GaussianNB": (naive_bayes.GaussianNB()),

            #Nearest Neighbor
            "KNeighborsClassifier": (neighbors.KNeighborsClassifier()),

            #SVM
            "SVC": (svm.SVC(gamma='auto', probability=True)),
            "NuSVC": (svm.NuSVC(probability=True)),
            "LinearSVC": (svm.LinearSVC()),

            #Trees
            "DecisionTreeClassifier": (tree.DecisionTreeClassifier()),
            "ExtraTreeClassifier": (tree.ExtraTreeClassifier()),

            #Tuned
            "SVC_tuned": (svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
                    decision_function_shape='ovr', degree=3, gamma='auto_deprecated',
                    kernel='rbf', max_iter=-1, probability=False, random_state=42,
                    shrinking=True, tol=0.001, verbose=False))        
        }):
        self.dict = _dict
    def get_list(self) -> list:
        dictlist = []
        for key in self.dict:
            dictlist.append((key, self.dict[key]))
        return dictlist
    def get_results(self, X, Y, scoring: str='accuracy', n_splits: int=10) -> Tuple[list, list]:
        results = []
        names = []
        for name, model in self.get_list():
            kfold = model_selection.KFold(n_splits=n_splits)
            cv_results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
            results.append(cv_results)
            names.append(name)
        return (names, results)


# %%
