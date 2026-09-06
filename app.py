

import streamlit as st
import pandas as pd
import joblib

# Load the saved linear regression model
loaded_lr_model = joblib.load('linear_regression_model.sav')

st.title('Sales Prediction App')

st.write("Enter the advertising budget for TV, Radio, and Newspaper to predict sales.")

# Create input fields for features
tv = st.slider('TV Advertising Budget', 0.0, 300.0, 150.0)
radio = st.slider('Radio Advertising Budget', 0.0, 50.0, 25.0)
newspaper = st.slider('Newspaper Advertising Budget', 0.0, 100.0, 50.0)

# Create a DataFrame for the input
input_data = pd.DataFrame([[tv, radio, newspaper]], columns=['TV', 'Radio', 'Newspaper'])

# Make prediction
if st.button('Predict Sales'):
    prediction = loaded_lr_model.predict(input_data)[0]
    st.success(f'Predicted Sales: {prediction:.2f} units')
