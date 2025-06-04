from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import  train_test_split
import pandas as pd 
import numpy as np
from  sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
import dataclasses
import os 
from src.utils import save_object
from src.exception import CustomException
import sys

@dataclasses.dataclass
class DataTransformerConfig:
    data_transformer_config: str = os.path.join('artifacts', 'preprocessing.pkl')
    

class DataTransformation:
    
    def __init__(self):
        self.data_transformation_config = DataTransformerConfig()
    
    def get_data_transformer_object(self):
      try:   
        numerical_features = ['reading_score',
                              'writing_score']
        
        categorical_features = ['gender', 
                                'race_ethnicity',
                                'parental_level_of_education', 
                                'lunch', 
                                'test_preparation_course']
        
        numerical_features_pipeline = Pipeline(
            steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler',StandardScaler())
            ]
        )
        
        categorical_features_pipeline = Pipeline(
            steps=[
                ('imputer',SimpleImputer(strategy='most_frequent')),
                ('onehotencoder',OneHotEncoder()),
                ('scaler',StandardScaler(with_mean=False))
            ]
        )
        
        
        preprocessor = ColumnTransformer(
            [
                ('numerical_pipeline',numerical_features_pipeline,numerical_features),
                ('categorical_pipeline',categorical_features_pipeline,categorical_features)
            ]
        ) 
        
        return preprocessor   
      except Exception as e:
          raise CustomException(e,sys)  
        
        
    def initiate_data_transformation(self,train_path,test_path):
      try:    
        train_data = pd.read_csv(train_path)
        test_data  = pd.read_csv(test_path)
        
        output_feature = 'math_score'
        
        input_features_train_data_df = train_data.drop(columns=output_feature,axis=1)
        target_feature_train_df = train_data[output_feature]
        
        input_features_test_data_df = test_data.drop(columns=output_feature,axis=1)
        target_feature_test_df = test_data[output_feature]
        
        preprocessor_obj = self.get_data_transformer_object()
        
        input_feature_train_df = preprocessor_obj.fit_transform(input_features_train_data_df)
        input_feature_test_df = preprocessor_obj.transform(input_features_test_data_df)
        
        
        train_arr = np.c_[
                input_feature_train_df, np.array(target_feature_train_df)
            ]
        test_arr = np.c_[input_feature_test_df, np.array(target_feature_test_df)]
        save_object(self.data_transformation_config.data_transformer_config,preprocessor_obj)
        
        
        return (
            train_arr,
            test_arr,
            self.data_transformation_config.data_transformer_config
        )
        
      except Exception as e:
          raise CustomException(e,sys)