import os
import sys
from src.logger import logging
from src.exception import CustomException
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

class TrainPipeline:
    def __init__(self):
        pass
    def train_pipeline(self):
        try:
            logging.info("Training pipeline is starting")
            obj=DataIngestion()
            train_data,test_data=obj.initiate_data_ingestion()
            logging.info("Data ingestion is completed and train and test data are saved in artifacts folder")
            logging.info("Data transformation is starting")
            data_transformation=DataTransformation()
            train_array,test_array,_=data_transformation.initiate_data_transformation(train_data,test_data)
            logging.info("Data transformation is completed and transformed data is saved in artifacts folder")
            logging.info("Model training is starting")
            modeltrainer=ModelTrainer()
            modeltrainer.initiate_model_trainer(train_array,test_array)
            logging.info("Model training is completed")

            return
        except Exception as e:
            raise CustomException(e,sys)

if __name__=="__main__":
    pipeline=TrainPipeline()
    pipeline.train_pipeline()    