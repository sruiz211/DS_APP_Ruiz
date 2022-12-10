# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 19:42:35 2022

@author: sruiz2
"""

import numpy as np 
import pickle 
import os
import streamlit as st

import sklearn
from sklearn.neural_network import MLPClassifier


#%% Load ML model
filename = 'CS_app.pkl'
model = pickle.load( open( filename , 'rb' ) )

#%% Load test data

# x_path = os.path.join( 'test_data' , 'X_test.txt'  )
# y_path = os.path.join( 'test_data' , 'y_test.txt'  )

X = {} 
Z = {}

# prepare test data

for i in range(1,5):
    X[f'{i:.0f}'] = np.loadtxt( os.path.join( 'test_data' ,  f'X_test{i:.0f}.txt' ) )
    Z[f'{i:.0f}'] = np.loadtxt( os.path.join( 'test_data' ,  f'Z_test{i:.0f}.txt' ) )    
    
def predict_function( model , X , class_type , class_opt , class_type_options , class_opt_options ):
    
    class_type_idx = class_type_options.index( class_type ) + 1
    class_opt_idx = class_opt_options.index( class_opt ) + 1
    
    class_type_str = '{}'.format(class_type_idx) 
    
    X_to_pred = X[class_type_str][class_opt_idx-1:class_opt_idx]
    
    y_pred = model.predict( X_to_pred ) 
    y_pred_prob = model.predict_proba( X_to_pred ) 
    
    return y_pred.astype(int)[0] , y_pred_prob.max()*100

st.write("""
# Human Activity Recognition using sensor data 

This app performs classification of human activities based on a series of feature vectors.

The classification is made for the following categories: walking, walking upstairs, walking downstairs, sitting, standing, and laying.
""")

# st.sidebar.header('User Input Parameters') #user input parameter collection with streamlit side bar
st.subheader('User Input parameters')

st.image('iCONS.jpg')

class_type_options = ('Walking','Walking downstairs','Walking upstairs','Sitting','Standing','Laying')
class_type = st.selectbox('Select human activity to predict', class_type_options )

class_opt_options = ('Human 1', 'Human 2', 'Human 3', 'Human 4', 'Human 5') 
class_opt = st.radio('Select one of the ', class_opt_options )

y_pred, y_pred_prob = predict_function( model , X , class_type , class_opt , class_type_options , class_opt_options )

st.text( f'Prediction is: {y_pred}, {class_type_options[y_pred-1]}, with a probability of {y_pred_prob:.2f}%')
