import os
import sys
from dataclasses import dataclass
from catboost import CatBoostRegressor
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model


@dataclass
class ModelTrainerConfig:
    model_file_path: str = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self) -> None:
        self.model_trainer_config = ModelTrainerConfig
    
    def initiate_model_training(self, train_array, test_array):
        try:
            logging.info("Split training and test data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                "Random Forest": RandomForestRegressor(),
                "Linear Regression": LinearRegression(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "KNN": KNeighborsRegressor(),
                "XGBRegressor": XGBRegressor(),
                "Catboosting Regressor": CatBoostRegressor(allow_writing_files=False, verbose=0),
                "AdaBoostRegressor": AdaBoostRegressor()
            }

            model_report = evaluate_model(X_train, y_train, X_test, y_test, models)
            logging.info("Created report")

            # getting the best model and score name
            best_model_name = max(model_report, key= lambda x: model_report[x])
            best_model_score = max(model_report.values())

            best_model = models[best_model_name]
            
            if best_model_score < 0.6:
                raise CustomException("Bad models with less than 60% with R2 Score", sys)
            
            logging.info("Best model found on both training and test data")

            save_object(self.model_trainer_config.model_file_path, best_model)

            logging.info("Model has been saved")
            
            # return r2 score for best model
            return model_report[best_model_name]

        except Exception as e:
            raise CustomException