from sklearn.ensemble import AdaBoostRegressor,GradientBoostingRegressor,RandomForestRegressor
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.tree import DecisionTreeRegressor
from catboost import CatBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
import sys

import os
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from src.utils import evaluate_model_training,save_object
from sklearn.metrics import r2_score
import dill

@dataclass
class ModelTrainingConfig:
    model_training_config = os.path.join('artifacts','model.pkl')
    
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config_value = ModelTrainingConfig()
        
    
    def initiate_model_training(self,train_arr,test_arr):
        try:
            
            models: dict = {
                "AdaBoostRegressor" : AdaBoostRegressor(),
                'GradientBoostRegressor': GradientBoostingRegressor(),
                'RandomForestRegressor' : RandomForestRegressor(),
                'LinearRegressor' : LinearRegression(),
                'Ridge' : Ridge(),
                'Lasso': Lasso(),
                'DecisionTreeRegressor' : DecisionTreeRegressor(),
                'CatBoostRegressor' : CatBoostRegressor(),
                'KNeighborRegressor' : KNeighborsRegressor()
            }
            
            
            X_train,X_test,Y_train,Y_test = (
                train_arr[:,:-1],
                test_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,-1]
            )
            
            
            report :dict= evaluate_model_training(models,X_train,X_test,Y_train,Y_test)
            
            max_accuracy = max(sorted(report.values()))
            
            if max_accuracy < 0.6:
                raise CustomException('this is a broken model which has accuracy less than 60% percentage, please check with other',sys)
            
            best_model_name:str =''
            if max_accuracy > 0.6 :
              for model_name , accuracy_score in report.items():
                       print(f'model_name : {model_name} wit accuracy : {accuracy_score}')
                       if max_accuracy == accuracy_score:
                         best_model_name = model_name
                         
                         
            
            if best_model_name is not None:
                print(f'best model is {best_model_name} with an accuracy of {report.get(best_model_name)}')   
            
            best_model_obj = models.get(best_model_name)
            
            
            best_model_obj.fit(X_train,Y_train)
            save_object(
                filepath=self.model_trainer_config_value.model_training_config,
                obj=best_model_obj
            )
            predicted_values = best_model_obj.predict(X_test) 
            r2ScoreValue = r2_score(Y_test,predicted_values)
            return r2ScoreValue
            
        except Exception as e:
            raise CustomException(e,sys)
        
        