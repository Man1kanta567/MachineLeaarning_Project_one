import os
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
import pandas as pd 
from sklearn.model_selection import train_test_split 
import sys
from src.components.dataTransformation import DataTransformation
from src.components.model_training import ModelTrainer




@dataclass
class Data_injestion_config:
    raw_data_path: str = os.path.join('artifacts','data.csv')
    training_data_path: str = os.path.join('artifacts','training_data.csv')
    test_data_path: str = os.path.join('artifacts','test_data.csv')
    
    
class DataInjestion:
    def __init__(self):
        self.injestion_config = Data_injestion_config()
        
        
    def initiate_data_injestion(self):
        logging.info('started initiating the data injestion')
        try:
            df = pd.read_csv(r'/Users/manikantamukkapati/Downloads/Personal Coding Programs/python projects/python_machine_learning_projects/MachineLearningProject_one/data/stud.csv')
            logging.info('data is readed from file and stored as a data frame')
            
            os.makedirs(os.path.dirname(self.injestion_config.raw_data_path),exist_ok=True)
            logging.info('directory is created if already not present')
            
            df.to_csv(self.injestion_config.raw_data_path,header=True,index=False)
            
            training_data,test_data = train_test_split(df,test_size=0.2,random_state=42)
            
            training_data.to_csv(self.injestion_config.training_data_path,header=True,index=False)
            test_data.to_csv(self.injestion_config.test_data_path,header=True,index=False)

            logging.info('training and test data are loaded in the files ...')
            
            return(
                self.injestion_config.training_data_path,
                self.injestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)
        
        
if __name__ =='__main__':
    datainjestion = DataInjestion()
    train_path, test_path =datainjestion.initiate_data_injestion()
    data_transformation = DataTransformation()
    train_arr,test_arr,path =data_transformation.initiate_data_transformation(train_path=train_path,test_path=test_path)
    modeltrainer = ModelTrainer()
    r2ScoreValue =modeltrainer.initiate_model_training(train_arr=train_arr,test_arr=test_arr)
    print('accuracy is :',r2ScoreValue)