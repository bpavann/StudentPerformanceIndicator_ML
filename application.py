import sys
import pickle
import numpy as np
from src.logger import logging
from src.exception import CustomException
from flask import Flask,request,render_template
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application=Flask(__name__)
app=application

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['GET', 'POST'])
def predict():
    try:
        if request.method == 'GET':
            return render_template('home.html')
        else:
            data=CustomData(
                gender=request.form.get('gender'),
                race_ethnicity=request.form.get('race_ethnicity'),
                parental_level_of_education=request.form.get('parental_level_of_education'),
                lunch=request.form.get('lunch'),
                test_preparation_course=request.form.get('test_preparation_course'),
                reading_score=int(request.form.get('reading_score')),
                writing_score=int(request.form.get('writing_score'))
            )
            final_new_data=data.get_data_as_dataframe()
            print(final_new_data)

            predict_pipeline=PredictPipeline()
            result=predict_pipeline.predict(final_new_data)
            print(f"Prediction result is {result}")
            return render_template('home.html',result=result[0])


    except Exception as e:
        raise CustomException(e,sys)

if __name__ == "__main__":
    app.run(host="0.0.0.0")

