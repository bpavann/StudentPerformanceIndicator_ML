import os
import sys
import pickle
from sklearn.metrics import r2_score
from src.exception import CustomException

def save_object(file_path,obj):
    try:
        dirc_name=os.path.dirname(file_path)
        os.makedirs(dirc_name,exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj,file_obj)
    except Exception as e:
        raise CustomException(e,sys)
    

def evaluate_models(X_train,y_train,X_test,y_test,models):
    try:
        report={}
        for i in range(len(models)):
            model=list(models.values())[i]
            model.fit(X_train,y_train)

            y_train_pred=model.predict(X_train)
            model_train_score=r2_score(y_train,y_train_pred)

            y_test_pred=model.predict(X_test)
            model_test_score=r2_score(y_test,y_test_pred)

            report[list(models.keys())[i]] = model_test_score
        
        return report
    
    except Exception as e:
        raise CustomException(e,sys)