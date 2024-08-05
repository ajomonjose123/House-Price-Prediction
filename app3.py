import numpy as np
import pandas as pd
import pickle
import streamlit as st
import os
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()

# Load the model
file_path = r'D:\Big_data_analytics\sem_3\Ameer\rf_reg.pkl'

if not os.path.exists(file_path):
    st.error(f"File not found: {file_path}")
else:
    with open(file_path, 'rb') as f:
        model = pickle.load(f)

# Function to get predicted price
def get_predicted_price(input_data):
    features = np.array([input_data])
    prediction = model.predict(features)[0]
    return round(prediction, 2)

# Streamlit app
def main():
    st.title("House Price Predictor")
    html_temp = """
    <div>
    <h2>House Price Prediction ML App</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    
    # Input fields
    bedrooms = st.number_input("Bedrooms", min_value=0, max_value=10, value=3)
    bathrooms = st.number_input("Bathrooms", min_value=0.0,value=5.0)
    floors = st.number_input("Floors", min_value=0, max_value=5, value=1)
    waterfront = st.selectbox("Waterfront", [0, 1])
    view = st.number_input("View", min_value=0, max_value=4, value=0)
    sqft_above = st.number_input("Square Feet Above", min_value=0, value=1500)
    sqft_basement = st.number_input("Square Feet Basement", min_value=0, value=2000)
    lat = st.number_input("Latitude", min_value=0.0, value=90.0)
    long = st.number_input("Longitude", min_value=-180.0, value=180.0)
    living_lot = st.number_input("Living Lot", min_value=0.0, value=1.0)
    living15_lot15 = st.number_input("Living15 Lot15", min_value=0.0, value=1.0)
    old = st.number_input("Old (Years Since Built)", min_value=0, value=200)
    con_grade = st.number_input("Condition/Grade", min_value=0.0, value=1.0)
    


    if st.button("Predict"):
        input_data = [
            bedrooms, bathrooms, floors, waterfront, view, 
            sqft_above, sqft_basement, lat, long, living_lot, 
            living15_lot15, old, con_grade
        ]
        # input_data = pd.DataFrame(input_data)
        # input_data_r=scaler.fit_transform(input_data)
        # print(input_data)
        # print(input_data)
        print(np.array(input_data).ravel())
        result = get_predicted_price(np.array(input_data).ravel())
        st.success(f"Predicted Price: {result} USD")

if __name__ == "__main__":
    main()
