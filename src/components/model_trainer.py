import os
import sys
import numpy as np
import pandas as pd
from src.logger import logging
from src.utils import save_object,evaluate_models
from src.exception import CustomException
from dataclasses import dataclass
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import(AdaBoostRegressor,RandomForestRegressor,GradientBoostingRegressor)
from sklearn.metrics import r2_score,mean_squared_error

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info('Splitting training and testing data inputs seperately')
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models={
                "XGBRegressor":XGBRegressor(),
                "K-Neighbours":KNeighborsRegressor(),
                "Linear Regression":LinearRegression(),
                "Random Forest":RandomForestRegressor(),
                "Decision Tree":DecisionTreeRegressor(),
                "AdaBoost Regressor":AdaBoostRegressor(),
                "Gradient Boosting":GradientBoostingRegressor(),
                "CatBoosting Regressor":CatBoostRegressor(verbose=False)
            }
            params={
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "K-Neighbors": {
                    'n_neighbors': [3, 5, 7, 9],
                    'weights': ['uniform', 'distance'],
                    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
                },
                "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoosting Regressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }

            model_report:dict=evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                params=params
            )
            
            #To get the best model score from dict
            best_model_score=max(sorted(model_report.values()))
            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
                ]

            if best_model_score<0.65:
                raise CustomException("There is NO best model found")
            logging.info(f"Best model found:{best_model_name} with r2 score: {best_model_score}")

            best_model=models[best_model_name]

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            # Evaluate the model with train data
            predicted_train=best_model.predict(X_train)
            r2_score_train=r2_score(y_train,predicted_train)
            mean_square_error_train=mean_squared_error(y_train,predicted_train)

            # Evaluate the model with test data
            predicted_test=best_model.predict(X_test)
            r2_score_test=r2_score(y_test,predicted_test)
            mean_square_error_test=mean_squared_error(y_test,predicted_test)

            logging.info(f"r2 score for training data is {r2_score_train} and for testing data is {r2_score_test}")
            logging.info(f"mean square error for training data is {mean_square_error_train} and for testing data is {mean_square_error_test}")

            return r2_score_test
        
        except Exception as e:
            raise CustomException(e,sys)
        