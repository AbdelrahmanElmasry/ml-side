from flask import Flask
from flask import request
from flask import jsonify 

import pickle

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

app = Flask('client-credit-model-service')

model_file = 'model2.bin'
dv_file = 'dv.bin'


with open(model_file, 'rb') as f_in:
    model =  pickle.load(f_in) 
with open(dv_file, 'rb') as f_in:
    dv =  pickle.load(f_in) 


test_client = {
 'job': 'retired',
 'duration': 445,
 'poutcome': 'success',
}

@app.route('/predict', methods=['POST'])
def predict():
    customer = request.get_json()

    X = dv.transform([customer]) 
    y_pred = model.predict_proba(X)[0,1]
    churn_decision = bool(y_pred >= 0.5)

    result = jsonify({
        'credit_probability': y_pred,
        'credit': churn_decision
    })

    return result


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)