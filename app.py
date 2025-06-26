import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sklearn


st.write(f"✅ scikit-learn version: {sklearn.__version__}")
st.write(f"✅ joblib version: {joblib.__version__}")
st.write(f"✅ numpy version: {np.__version__}")
st.write(f"✅ pandas version: {pd.__version__}")


# ✅ Load the compressed model pipeline
model = joblib.load("model.joblib")

st.title("California Housing Price Prediction")
st.write("Input the features to predict median house value.")

# User inputs
longitude = st.slider("Longitude", -125, -113, -120)
latitude = st.slider("Latitude", 32, 42, 36)
housing_median_age = st.slider("Housing Median Age", 1, 52, 25)
total_rooms = st.slider("Total Rooms", 2, 40000, 2000)
total_bedrooms = st.slider("Total Bedrooms", 1, 7000, 400)
population = st.slider("Population", 3, 35682, 1000)
households = st.slider("Households", 1, 6082, 400)
median_income = st.slider("Median Income", 0, 15, 3)
ocean_proximity = st.selectbox("Ocean Proximity", ['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN'])

# Build input DataFrame
input_df = pd.DataFrame([{
    "longitude": longitude,
    "latitude": latitude,
    "housing_median_age": housing_median_age,
    "total_rooms": total_rooms,
    "total_bedrooms": total_bedrooms,
    "population": population,
    "households": households,
    "median_income": median_income,
    "ocean_proximity": ocean_proximity
}])

# Predict
prediction = model.predict(input_df)
st.subheader(f"🏡 Predicted Median House Value: ${prediction[0]:,.2f}")
