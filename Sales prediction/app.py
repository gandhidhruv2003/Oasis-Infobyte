import streamlit as st
from joblib import load
import numpy as np

model = load("model.joblib")

st.title("Sales Prediction Dashboard")
st.write("Enter values for TV, Newspaper, and Radio to get the predicted Sales value.")

tv = st.slider("TV", min_value=0.0, max_value=500.0, step=0.1)
newspaper = st.slider("Newspaper", min_value=0.0, max_value=500.0, step=0.1)
radio = st.slider("Radio", min_value=0.0, max_value=500.0, step=0.1)

input_features = np.array([[tv, newspaper, radio]])
predicted_sales = model.predict(input_features)[0]

st.write(f"Predicted Sales: {predicted_sales:.2f}")