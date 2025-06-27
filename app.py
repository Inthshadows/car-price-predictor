#!/usr/bin/env python
# coding: utf-8

# In[3]:


import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Load all saved models and encoders
model = pickle.load(open("XGBRegressor_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
label_encoders = pickle.load(open("label_encoders.pkl", "rb"))
ohe = pickle.load(open("One_Hot_Encoder.pkl", "rb"))

# Streamlit App Title
st.title("🚗 Car Price Prediction App")

st.sidebar.header("Enter Car Features:")

levy = st.sidebar.number_input("Levy", min_value=0.0, value=900.0)
engine_volume = st.sidebar.number_input("Engine Volume (L)", min_value=0.0, value=2.0)
mileage = st.sidebar.number_input("Mileage (km)", min_value=0, value=100000)
age_of_car = st.sidebar.slider("Age of Car (Years)", min_value=0, max_value=30, value=10)

manufacturer = st.sidebar.selectbox("Manufacturer", label_encoders['Manufacturer'].classes_)
model_name = st.sidebar.selectbox("Model", label_encoders['Model'].classes_)
category = st.sidebar.selectbox("Category", label_encoders['Category'].classes_)
fuel_type = st.sidebar.selectbox("Fuel Type", label_encoders['Fuel_type'].classes_)
color = st.sidebar.selectbox("Color", label_encoders['Color'].classes_)
leather = st.sidebar.selectbox("Leather Interior", ['Yes', 'No'])
wheel = st.sidebar.selectbox("Wheel", ['Left wheel', 'Right-hand drive'])

gear_type = st.sidebar.selectbox("Gear Box Type", ['Automatic', 'Manual', 'Tiptronic', 'Variator'])
drive_wheel = st.sidebar.selectbox("Drive Wheel", ['4x4', 'Front', 'Rear'])

cylinders = st.sidebar.slider("Cylinders", min_value=1, max_value=16, value=4)
airbags = st.sidebar.slider("Airbags", min_value=0, max_value=16, value=6)

# Encode the selected categorical features
encoded_inputs = {
    'Manufacturer': label_encoders['Manufacturer'].transform([manufacturer])[0],
    'Model': label_encoders['Model'].transform([model_name])[0],
    'Category': label_encoders['Category'].transform([category])[0],
    'Fuel_type': label_encoders['Fuel_type'].transform([fuel_type])[0],
    'Color': label_encoders['Color'].transform([color])[0],
    'Leather_interior': label_encoders['Leather_interior'].transform([leather])[0],
    'Wheel': label_encoders['Wheel'].transform([wheel])[0]
}

# One-hot encoding for gear box and drive wheels
ohe_input = pd.DataFrame([[gear_type, drive_wheel]], columns=['Gear_box_type', 'Drive_wheels'])
ohe_transformed = ohe.transform(ohe_input)
ohe_df = pd.DataFrame(ohe_transformed, columns=ohe.get_feature_names_out(['Gear_box_type', 'Drive_wheels']))

# Final input dataframe
input_df = pd.DataFrame([[
    levy,
    2025 - age_of_car,
    engine_volume,
    mileage,
    cylinders,
    airbags,
    age_of_car,
    encoded_inputs['Manufacturer'],
    encoded_inputs['Model'],
    encoded_inputs['Category'],
    encoded_inputs['Fuel_type'],
    encoded_inputs['Color'],
    encoded_inputs['Leather_interior'],
    encoded_inputs['Wheel']
]], columns=[
    'Levy', 'Prod_year', 'Engine_volume', 'Mileage', 'Cylinders', 'Airbags',
    'Age_of_Car', 'Manufacturer', 'Model', 'Category', 'Fuel_type', 'Color',
    'Leather_interior', 'Wheel'
])

# Combine with one-hot encoded
final_input = pd.concat([input_df, ohe_df], axis=1)

# Apply scaler
final_input[['Levy', 'Engine_volume', 'Mileage', 'Age_of_Car']] = scaler.transform(
    final_input[['Levy', 'Engine_volume', 'Mileage', 'Age_of_Car']]
)

# Predict
if st.button("Predict Price"):
    price = model.predict(final_input)[0]
    st.success(f"💰 Predicted Car Price: ${price:,.2f}")


# In[ ]:




