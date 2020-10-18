from flask import Flask
import pickle
import pandas as pd
import numpy as np
import joblib
import json

app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))

'''['<:age>', '<:sex>', '<:chest_pain_type>', 'resting_blood_pressure', 
              '<:cholesterol>', 'fasting_blood_sugar',
              '<:max_heart_rate_achieved>', 'exercise_induced_angina', 
              '<:st_depression>', '<:st_slope_type>', 'num_major_vessels', 
              '<:thalassemia_type>', 'target']'''

@app.route("/api/<int:age>/<int:sex>/<int:chest_pain_type>/<int:resting_blood_pressure>/<int:cholesterol>/<int:fasting_blood_sugar>/<int:max_heart_rate_achieved>/<int:exercise_induced_angina>/<int:st_depression>/<int:st_slope_type>/<int:num_major_vessels>/<int:thalassemia_type>" , methods = ['GET'])
def api (age, sex, chest_pain_type, resting_blood_pressure, cholesterol, fasting_blood_sugar, max_heart_rate_achieved, exercise_induced_angina, st_depression, st_slope_type, num_major_vessels, thalassemia_type):
    df = np.array([[age, sex, chest_pain_type, resting_blood_pressure, cholesterol, fasting_blood_sugar,  max_heart_rate_achieved, exercise_induced_angina, st_depression, st_slope_type, num_major_vessels, thalassemia_type]]) 

    df=(df-np.min(df))/(np.max(df)-np.min(df))

    print(age, sex)

    y = model.predict(df)

    if(y == 0):
        result = 'You have a Healthy Heart!'
    else:
        result = 'Seek Medical Care ASAP'

    return json.dumps(result)


if __name__ == '__main__':
    app.run(debug=True)


