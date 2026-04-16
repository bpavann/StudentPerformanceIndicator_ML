import os
import sys
import pickle
from sklearn.metrics import r2_score
from src.exception import CustomException
from sklearn.model_selection import GridSearchCV

def save_object(file_path,obj):
    try:
        dirc_name=os.path.dirname(file_path)
        os.makedirs(dirc_name,exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj,file_obj)
    except Exception as e:
        raise CustomException(e,sys)
    

def evaluate_models(X_train,y_train,X_test,y_test,models,params):
    try:
        report={}
        for model_name,model in models.items():
            params=params.get(model_name,{})

            gs=GridSearchCV(model,params,cv=5)
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

            y_train_pred=model.predict(X_train)
            model_train_score=r2_score(y_train,y_train_pred)

            y_test_pred=model.predict(X_test)
            model_test_score=r2_score(y_test,y_test_pred)

            report[model_name] = model_test_score
        
        return report
    
    except Exception as e:
        raise CustomException(e,sys)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e,sys)
        
    