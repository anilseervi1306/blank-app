import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the trained model
with open('rf_regression_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Function to preprocess input data
def preprocess_input(data):
    # Create a DataFrame from the input data
    input_df = pd.DataFrame(data, index=[0])

    # Convert categorical variables to numeric using one-hot encoding
    input_df = pd.get_dummies(input_df, drop_first=True)

    # Ensure all expected columns are present (including dummy variables)
    expected_cols = ['Present_Price', 'Kms_Driven', 'No_of_Years',
                     'Fuel_Type_Diesel', 'Fuel_Type_Petrol',
                     'Seller_Type_Individual', 'Transmission_Manual']
    for col in expected_cols:
        if col not in input_df:
            input_df[col] = 0

    # Return input DataFrame with the correct columns
    return input_df[expected_cols]

# Streamlit UI
st.title("Car Price Prediction App")

# Input fields for the model
present_price = st.number_input("Present Price (in currency)", min_value=0.0)
kms_driven = st.number_input("Kilometers Driven", min_value=0)
fuel_type = st.selectbox("Fuel Type", options=["Petrol", "Diesel"])
seller_type = st.selectbox("Seller Type", options=["Individual", "Dealer"])
transmission = st.selectbox("Transmission", options=["Manual", "Automatic"])
owner = st.number_input("Owner", min_value=0, max_value=3)  # Assuming 0: First Owner, 3: Third Owner

# Calculate the number of years
current_year = 2020
year_of_manufacture = current_year - st.number_input("Car Year (Year of Manufacture)", min_value=1900, max_value=current_year)

no_of_years = current_year - year_of_manufacture

# Button for prediction
if st.button("Predict Selling Price"):
    # Prepare input data
    input_data = {
        'Present_Price': present_price,
        'Kms_Driven': kms_driven,
        'Fuel_Type': fuel_type,
        'Seller_Type': seller_type,
        'Transmission': transmission,
        'Owner': owner,
        'No_of_Years': no_of_years
    }
    
    # Preprocess the input data
    processed_input = preprocess_input(input_data)
    
    # Make prediction
    predicted_price = model.predict(processed_input)
    
    # Display result
    st.write(f"Predicted Selling Price: {predicted_price[0]:.2f} currency")
