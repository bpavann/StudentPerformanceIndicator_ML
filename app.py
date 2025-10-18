import sys
import pickle
import numpy as np
from src.logger import logging
from src.exception import CustomException
from flask import Flask,request,render_template
from sklearn.preprocessing import StandardScaler

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
            pass

    except Exception as e:
        raise CustomException(e,sys)