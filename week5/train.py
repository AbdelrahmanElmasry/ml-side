#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import pickle


from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction import DictVectorizer

## Parameters ##

C = 1.0
n_splits = 5

df = pd.read_csv('data.csv')


# Data preparation


df.columns = df.columns.str.lower().str.replace(' ','_')

categorial_cols = list(df.dtypes[df.dtypes == 'object'].index)

for col in categorial_cols:
    df[col] = df[col].str.lower().str.replace(' ', '_')
# Convert total_charges to numerical values, ignore errors as NA
total_charges = pd.to_numeric(df.totalcharges, errors='coerce')
df.totalcharges = total_charges
df.totalcharges = df.totalcharges.fillna(0)
# Filter dataframe by specific column, pick specific columns
df[df['totalcharges'].isnull()][['customerid', 'totalcharges']]
df.churn = (df.churn == 'yes').astype('int')


# Data split


df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
y_test = df_test.churn.values
# del df_full_train['churn']


numerical = ['tenure', 'monthlycharges', 'totalcharges']
categorial = ['gender', 'seniorcitizen', 'partner', 'dependents',
     'phoneservice', 'multiplelines', 'internetservice',
       'onlinesecurity', 'onlinebackup', 'deviceprotection', 'techsupport',
       'streamingtv', 'streamingmovies', 'contract', 'paperlessbilling',
       'paymentmethod']



def train(df, y_train, C=1.0):
    dicts = df[categorial + numerical].to_dict(orient='records')
    dv = DictVectorizer(sparse = False)
    X_train = dv.fit_transform(dicts)
    
    model = LogisticRegression(C=C, max_iter=1000)
    model.fit(X_train, y_train)
    
    return dv, model

def predict(df_train, dv, model):
    dicts = df_train[categorial + numerical].to_dict(orient='records')
    X = dv.transform(dicts)
    
    y_pred = model.predict_proba(X)[:,1]
    
    return y_pred

#  Validation
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1) 
print(f'model-validation::validate with C = {C}')
scores = []
fold_number = 0
for train_idx, val_idx in kfold.split(df_full_train) :
    df_train = df_full_train.iloc[train_idx]
    df_val = df_full_train.iloc[val_idx]

    y_train = df_train.churn.values
    y_val = df_val.churn.values

    ## Train the model on the training dataset
    dv, model = train(df_train, y_train, C=C)
    ## Validate the model on the validation dataset
    y_pred = predict(df_val, dv, model)
    
    ## Evaluate the model accuracy
    auc = roc_auc_score(y_val, y_pred)
    scores.append(auc)

    print(f'model-validation::auc on fold {fold_number} ==> {auc}')
    fold_number += 1

print(f'model-validation::validation result is {(np.mean(scores), np.std(scores))}')


print(f'model-training:: training started.... ')

## Train the final model on the full training dataset
dv, model = train(df_full_train, df_full_train.churn.values, C=1)
## Validate the model on the test dataset
y_test_pred = predict(df_test, dv, model )

auc = roc_auc_score(y_test, y_test_pred)
print(f'model-training:: final AUC = {auc}')


# ***Save the model***
output_file = f'model_C={C}.bin'
output_file

with open(output_file, 'wb') as f_out:
    pickle.dump((dv, model), f_out)
    f_out.close()

print(f'model-saving:: the model has been saved to {output_file}')

