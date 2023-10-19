#!/usr/bin/env python
# coding: utf-8

import requests


url = 'http://localhost:8000/predict'

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

response = requests.post(url, json=test_customer).json()
print('Churn prediction is', response)