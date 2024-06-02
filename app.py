import pickle
from flask import Flask,request,jsonify,render_template
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

app=Flask(__name__)

scaler=pickle.load(open("scalar.pkl","rb"))
model=pickle.load(open("ridge.pkl",'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['GET',"POST"])
def predict():
    if request.method=="POST":
        Temp=float(request.form.get("Temp"))
        RH=float(request.form.get("RH"))
        WS=float(request.form.get("WS"))
        Rain=float(request.form.get("Rain"))
        FFMC=float(request.form.get("FFMC"))
        DMC=float(request.form.get("DMC"))
        ISI=float(request.form.get("ISI"))
        Classes=float(request.form.get("Classes"))
        Region=float(request.form.get("Region"))
        new_data=scaler.transform([[Temp,RH,WS,Rain,FFMC,DMC,ISI,Classes,Region]])
        result=model.predict(new_data)
        return render_template('index.html',result=result[0])
    else:
        return render_template('index.html')






if __name__=="__main__":
    app.run()