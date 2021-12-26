from flask import Flask, render_template, request
import jsonify
import requests
import pickle
import numpy as np

#import awsgi

application = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))
Kidict = pickle.load(open('kidict.pkl', 'rb'))

    
@application.route('/',methods=['GET'])
def index():
    return render_template('index.html')

@application.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        MMRAcquisitionAuctionAveragePrice = float(request.form["MMRAcquisitionAuctionAveragePrice"])
        VehOdo = float(request.form["VehOdo"])
        Make = request.form["Make"]
        Make2 = np.nan_to_num(Kidict.get(Make))
        Model = request.form["Model"]
        Model2 = np.nan_to_num(Kidict.get(Model))
        Transmission = request.form["Transmission"]
        Transmission2 = np.nan_to_num(Kidict.get(Transmission))
        output = model.predict([[MMRAcquisitionAuctionAveragePrice,VehOdo,Make2,Model2,Transmission2]])

        if output == 0:
            return render_template('result2.html',prediction_text = "This is car is a good oportunity")
        else:
            return render_template('result.html',prediction_text = "Becareful this is not a good buy")
    else:
        return render_template('index.html')

if __name__=="__main__":
    application.run(host='0.0.0.0', port=8080)
    
    