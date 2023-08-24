import sys
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler 
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_object_file_path = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    def __init__(self) -> None:
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self) -> ColumnTransformer:
        """
        this function is to make data transformations
        """

        try:
            logging.info("Data transformation started")

            numerical_columns = ["reading_score" , "writing_score"]
            categorical_columns = ["gender", "race_ethnicity", "parental_level_of_education", "lunch", "test_preparation_course"]

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            logging.info("Numerical columns scaling completed")

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder()), 
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            logging.info("Categorical columns encoding completed")

            preprocessor = ColumnTransformer([
                ("numerical_pipeline", num_pipeline, numerical_columns),
                ("categorical_pipeline", cat_pipeline, categorical_columns)
            ])

            logging.info("Full pipeline has been completed")

            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Train and test data have been read")

            logging.info("Obtaining preprocessor object")

            preprocessor = self.get_data_transformer_object()

            target_column_name = "math_score"

            # we can check the types of data here

            input_features_train_df = train_df.drop(target_column_name, axis=1)
            target_features_train_df = train_df[target_column_name]

            input_features_test_df = test_df.drop(target_column_name, axis=1)
            target_features_test_df = test_df[target_column_name]

            input_feature_train_arr = preprocessor.fit_transform(input_features_train_df)
            input_feature_test_arr = preprocessor.transform(input_features_test_df)
            
            train_arr = np.c_[
                input_feature_train_arr, np.array(target_features_train_df)
            ]

            test_arr = np.c_[
                input_feature_test_arr, np.array(target_features_test_df)
            ]

            save_object(
                file_path = self.data_transformation_config.preprocessor_object_file_path, 
                obj = preprocessor

            )

            logging.info("Saved preprocessing Object")

            return (
                train_arr, 
                test_arr, 
                self.data_transformation_config.preprocessor_object_file_path
            )
        except Exception as e:
            raise CustomException(e, sys)