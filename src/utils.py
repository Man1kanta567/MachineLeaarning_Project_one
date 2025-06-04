import os 
import dill
from src.exception import CustomException
import sys
from sklearn.metrics import r2_score

def save_object(filepath,obj):
    try:
        dir_path = os.path.dirname(filepath)
        os.makedirs(dir_path, exist_ok=True)
        with open(filepath, mode='wb') as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)
    
    
def evaluate_model_training(models:dict,X_train,X_test,Y_train,Y_test)->dict:
    try:
       report = {}
       for model_name, model_obj in models.items():
          model_obj.fit(X_train, Y_train)
          predicted_data_test = model_obj.predict(X_test)
          accuracy = r2_score(y_true=Y_test, y_pred=predicted_data_test)
          report[model_name] = accuracy
       return report
    except Exception as e:
        raise CustomException(e,sys)
        