import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Custom transformer used in the pipeline
from sklearn.base import BaseEstimator, TransformerMixin

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rooms_per_household = X[:, 3] / X[:, 6]
        population_per_household = X[:, 5] / X[:, 6]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, 4] / X[:, 3]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]


# ✅ Load the full pipeline (preprocessing + model)
model = joblib.load("full_model_pipeline.pkl")

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

# Build input as DataFrame
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

# Make prediction
prediction = model.predict(input_df)
st.subheader(f"🏡 Predicted Median House Value: ${prediction[0]:,.2f}")
