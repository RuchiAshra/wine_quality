import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import Ridge
from flask import Flask, request, jsonify, render_template

app=Flask(__name__)
# upload pickle models from the folder
ann_model = pickle.load(open('wine_quality_scalar.pkl', 'rb'))

# open home page

@app.route('/')
def home_page():
    return render_template('index.html')
#prediction page
@app.route('/predict', methods=['POST', 'GET'])
def predict_datapoint():
    if request.method == 'POST':
        Type = float(request.form.get('type'))
        Fixed_acidity = float(request.form.get('fixed acidity	'))
        Volatile_acidity= float(request.form.get('volatile acidity	'))
        Citric_acid=float(request.form.get('citric acid'))
        Residual_suger=float(request.form.get('residual sugar'))
        Chlorides=float(request.form.get('chlorides'))
        Free_sulfur_dioxide=float(request.form.get('free sulfur dioxide'))
        Total_sulfur_dioxide=float(request.form.get('total sulfur dioxide'))
        Density=float(request.form.get('density'))
        PH=float(request.form.get('pH'))
        Sulphates=float(request.form.get('sulphates'))
        Alcohol=float(request.form.get('alcohol'))

        new_data=scaler_model.transform([[Type,Fixed_acidity,Volatile_acidity,Citric_acid,Residual_suger,Chlorides,Free_sulfur_dioxide,Total_sulfur_dioxide,Density,PH,Sulphates,Alcohol]])
        result=ann_model_model.predict(new_data)

        return render_type('index.html', Result=result[0])
    else:
        return render_type('index.html')
    if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)






