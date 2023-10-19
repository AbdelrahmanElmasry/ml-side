from flask import Flask
from flask import request
from flask import jsonify 

import pickle

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

app = Flask('churn-model-service')

model_file = 'model_C=1.0.bin'


with open(model_file, 'rb') as f_in:
    dv, model =  pickle.load(f_in) 


test_customer = {
 'gender': 'male',
 'seniorcitizen': 0,
 'partner': 'yes',
 'dependents': 'no',
 'tenure': 67,
 'phoneservice': 'yes',
 'multiplelines': 'yes',
 'internetservice': 'fiber_optic',
 'onlinesecurity': 'no',
 'onlinebackup': 'yes',
 'deviceprotection': 'no',
 'techsupport': 'no',
 'streamingtv': 'yes',
 'streamingmovies': 'no',
 'contract': 'one_year',
 'paperlessbilling': 'yes',
 'paymentmethod': 'bank_transfer_(automatic)',
 'monthlycharges': 88.4,
 'totalcharges': 5798.3 
}

@app.route('/predict', methods=['POST'])
def predict():
    customer = request.get_json()

    X = dv.transform([customer]) 
    y_pred = model.predict_proba(X)[0,1]
    churn_decision = bool (y_pred >= 0.5)

    result = jsonify({
        'churn_probability': y_pred,
        'churn': churn_decision
    })

    return result


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)

# print('Customer', test_customer)
# print('Churn probability', predict(test_customer))