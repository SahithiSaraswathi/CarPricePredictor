from flask import Flask,render_template,request
import numpy as np
import pandas as pd
import pickle
app = Flask(__name__)
model = pickle.load(open('LinearRegModel.pkl','rb'))
car=pd.read_csv('clean_car.csv')

@app.route('/')
def index():
    companies=sorted(car['company'].unique())
    carmodels=sorted(car['name'].unique())
    years=sorted(car['year'].unique(),reverse=True)
    fueltype=car['fuel_type'].unique()
    
    return render_template('index.html',company=companies,year=years,car_models=carmodels,fuel_type=fueltype);
@app.route('/predict',methods=['POST'])
def predict():
    company=request.form.get('company')
    carmodels=request.form.get('car_models')
    year=int(request.form.get('year'))
    fuel_type=request.form.get('fuel_type')
    kms_driven=int(request.form.get('kilo_driven'))
    #print(company,carmodels,fuel_type,year,kms_driven)
    #print('hi')
    prediction = model.predict(pd.DataFrame([[carmodels,company,year,kms_driven,fuel_type]],columns=[
        'name','company','year','kms_driven','fuel_type']))
    return str(np.round(prediction[0],2))
    

if __name__ == "__main__":
    app.run(debug=True)
 