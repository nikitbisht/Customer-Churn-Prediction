from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np


model_path = 'model\churn_pred_model.pkl'
with open(model_path,'rb') as f:
    model = pickle.load(f)



app = Flask(__name__)

@app.route('/')
def front_end():
    return render_template("index.html")  


@app.route('/predict', methods=['POST'])
def predict():
    age = int(request.form['age'])
    gender = request.form['gender']
    tenure = int(request.form['tenure'])
    support_calls = int(request.form['support_calls'])
    payment_delay = int(request.form['payment_delay'])
    subscription_type = request.form['subscription_type']
    contact_length = request.form['contract_length']
    expense = int(request.form['total_spend'])
    interaction = int(request.form['last_interaction'])
# Age	Gender	Tenure	Support Calls	Payment Delay	Subscription Type	Contract Length	Total Spend	Last Interaction	Churn

    data = {
            'Age': [age],
            'Gender':[gender],
            'Tenure': [tenure],
            'Support Calls': [support_calls],
            'Payment Delay': [payment_delay],
            'Subscription Type': [subscription_type],
            'Contract Length' : [contact_length],
            'Total Spend' : [expense],
            'Last Interaction' : [interaction]
        }
    df = pd.DataFrame(data)

    df['Gender'] = np.where(df['Gender']=='Male',1,np.where(df['Gender']=='Female',0,df['Gender']))
    df['Subscription Type'] = np.where(df['Subscription Type']=='Premium',2,np.where(df['Subscription Type']=='Standard',1,0))
    df['Contract Length'] = np.where(df['Contract Length']=='Annual',2,np.where(df['Contract Length']=='Quarterly',1,0))
    prediction = model.predict(df)
    if prediction == 1:
        prediction = "Churn"
    else:
        prediction = "No Churn"

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
