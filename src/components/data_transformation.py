import sys
import os
import numpy as np
import pandas as pd
from src.logger import logging
from dataclasses import dataclass
from src.utils import save_object
from sklearn.pipeline import Pipeline
from src.exception import CustomException
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer #For missing values
from sklearn.preprocessing import OneHotEncoder,StandardScaler


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.preprocessor_config=DataTransformationConfig()
    
    def get_data_transformer_object(self):
        try:
            #Create numerical and categorical columns
            numerical_columns=['writing_score','reading_score']
            categorical_columns=['gender','race_ethnicity','parental_level_of_education','lunch','test_preparation_course']

            #Numerical pipeline
            num_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler())
                ]
            )

            #Categorical pipeline
            cat_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encoder',OneHotEncoder()),
                    ('scaler',StandardScaler(with_mean=False))
                ]
            )
            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            #Combine numerical and categorical pipeline
            preprocessor=ColumnTransformer(
                [
                    ('num_pipeline',num_pipeline,numerical_columns),
                    ('cat_pipeline',cat_pipeline,categorical_columns)
                ]
            )
            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_data=pd.read_csv(train_path)
            test_data=pd.read_csv(test_path)
            logging.info("Read train and test data completed")

            preprocessor_obj=self.get_data_transformer_object()

            #Separate input features and target feature
            target_col_name='math_score'

            input_feature_train_df=train_data.drop(columns=[target_col_name],axis=1)
            target_feature_train_df=train_data[target_col_name]

            input_feature_test_df=test_data.drop(columns=[target_col_name],axis=1)
            target_feature_test_df=test_data[target_col_name]
            logging.info("Separated input features and target feature from train and test data")

            #Transforming using preprocessor object
            input_feature_train_arr=preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessor_obj.transform(input_feature_test_df)
            logging.info("Applied preprocessing object on training and testing data")

            #combine input and target feature arrays
            train_arr=np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr=np.c_[input_feature_test_arr,np.array(target_feature_test_df)]
            logging.info("Combined input and target feature arrays")

            #Save the preprocessor object
            save_object(
                file_path=self.preprocessor_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )
            logging.info("Saved preprocessing object")

            return(
                train_arr,
                test_arr,
                self.preprocessor_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e,sys)