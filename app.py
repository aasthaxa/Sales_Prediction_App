

import streamlit as st
import pandas as pd
import joblib

# Load the saved linear regression model
loaded_lr_model = joblib.load('linear_regression_model.sav')

st.title('Sales Prediction App')

st.write("Enter the advertising budget for TV, Radio, and Newspaper to predict sales.")

# Display model coefficients and intercept
st.subheader('Model Insights')
st.write(f"**Intercept:** {loaded_lr_model.intercept_:.2f}")
st.write("**Coefficients:**")
coef_df = pd.DataFrame({'Feature': ['TV', 'Radio', 'Newspaper'], 'Coefficient': loaded_lr_model.coef_})
st.table(coef_df)

st.subheader('Input Advertising Budgets')

# Create input fields for features
tv = st.number_input('TV Advertising Budget', min_value=0.0, max_value=300.0, value=150.0, step=0.1)
radio = st.slider('Radio Advertising Budget', 0.0, 50.0, 25.0)
newspaper = st.slider('Newspaper Advertising Budget', 0.0, 100.0, 50.0)

# Create a DataFrame for the input
input_data = pd.DataFrame([[tv, radio, newspaper]], columns=['TV', 'Radio', 'Newspaper'])

# Make prediction
if st.button('Predict Sales'):
    prediction = loaded_lr_model.predict(input_data)[0]
    st.success(f'Predicted Sales: {prediction:.2f} units')


