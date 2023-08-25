import os
import sys
import pandas as pd
import numpy as np
import pickle
from src.exception import CustomException
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from src.logger import logging


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
            
    except Exception as e:
        raise(e, sys)


def load_object(file_path):
    with open(file_path, 'rb') as f:
        loaded_object = pickle.load(f)

    return loaded_object


def evaluate_model(X_train, y_train, X_test, y_test, models, params):
    """
    this function takes X train and test and models and returns a report
    """

    try:
        
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            model_param = params[list(models.keys())[i]]

            logging.info("Started {} {}".format(model, model_param))

            gs = GridSearchCV(model, model_param, cv=3)
            gs.fit(X_train, y_train)
            
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            y_test_pred = model.predict(X_test)

            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        
        return report

    except Exception as e:
        raise CustomException(e, sys)