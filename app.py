import json
import pickle

from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd

app=Flask(__name__)
## Load the model
regmodel=pickle.load(open('regmodel.pkl','rb'))
scalar=pickle.load(open('scaling.pkl','rb'))
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    input_json = request.get_json()
    data = input_json.get('data', input_json)

    input_array = np.array(list(data.values())).reshape(1, -1)
    new_data = scalar.transform(input_array)
    output = regmodel.predict(new_data)[0]
    output = max(0, output)

    return f"The predicted house price is: ${output:.2f}"



@app.route('/predict', methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    final_input = scalar.transform([data])  # scale the input
    output = regmodel.predict(final_input)[0]
    # Clamp to avoid negative price
    output = max(0, output)
    return render_template("home.html", prediction_text=f"The House price prediction is ${output:.2f}")


if __name__=="__main__":
    app.run(debug=True)
   
     
