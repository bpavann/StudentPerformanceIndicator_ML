import os
import sys
import pandas as pd
from src.logger import logging
from dataclasses import dataclass
from src.exception import CustomException
from sklearn.model_selection import train_test_split
from src.components.data_transformation import DataTransformation,DataTransformationConfig

@dataclass
class DataIngestionConfig:
    train_path :str=os.path.join('artifacts','train.csv')
    test_path: str=os.path.join('artifacts','test.csv')
    raw_path: str=os.path.join('artifacts','raw.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Data Ingestion method starting")
        try:
            df=pd.read_csv("notebook/data/stud.csv")
            logging.info("Dataset read as dataframe")
            os.makedirs(os.path.dirname(self.ingestion_config.train_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_path,index=False,header=True)
            logging.info("Raw data is saved")

            train_df,test_df=train_test_split(df,test_size=0.2,random_state=42)
            train_df.to_csv(self.ingestion_config.train_path,index=False,header=True)
            test_df.to_csv(self.ingestion_config.test_path,index=False,header=True)
            logging.info("Ingestion of the data is completed")
            return(
                  self.ingestion_config.train_path,
                  self.ingestion_config.test_path
                )
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__=="__main__":
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()

    data_transformation=DataTransformation()
    data_transformation.initiate_data_transformation(train_data,test_data)