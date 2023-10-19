#!/usr/bin/env python
# coding: utf-8

import requests

url = 'http://localhost:8000/predict'

client = {"job": "retired", "duration": 445, "poutcome": "success"}


response = requests.post(url, json=client).json()
print('Credit prediction is', response)