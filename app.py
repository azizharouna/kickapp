from flask import Flask, render_template, request
import jsonify
import requests
import pickle
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
Kidict = pickle.load(open('kidict.pkl', 'rb'))
@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')


standard_to = StandardScaler()
@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        MMRAcquisitionAuctionAveragePrice = float(request.form["MMRAcquisitionAuctionAveragePrice"])
        VehOdo = float(request.form["VehOdo"])
        Make = request.form["Make"]
        Make2 = numpy.nan_to_num(Kidict.get(Make))
        Model = request.form["Model"]
        Model2 = numpy.nan_to_num(Kidict.get(Model))
        Transmission = request.form["Transmission"]
        Transmission2 = numpy.nan_to_num(Kidict.get(Transmission))
        output = model.predict([[MMRAcquisitionAuctionAveragePrice,VehOdo,Make,Model,Transmission]])

        if output < 1:
            return render_template('index.html',prediction_texts="This is car is a good oportunity")
        else:
            return render_template('index.html',prediction_text="Becareful this is not a good buy")
    else:
        return render_template('index.html')

if __name__=="__main__":
    app.run(debug=True)