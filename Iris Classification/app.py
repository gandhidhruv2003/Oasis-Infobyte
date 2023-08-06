import streamlit as st
import numpy as np
from joblib import load
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris

model = load("model.joblib")

iris = load_iris()

st.title("Flower Species Predictor")

sepal_length = st.number_input("Enter Sepal Length (cm):")
sepal_width = st.number_input("Enter Sepal Width (cm):")
petal_length = st.number_input("Enter Petal Length (cm):")
petal_width = st.number_input("Enter Petal Width (cm):")

predict_button = st.button("Predict Flower Species")
if predict_button: 
    scaler = StandardScaler()

    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    input_data_scaled = scaler.fit_transform(input_data)

    prediction = model.predict(input_data_scaled)[0]
    predicted_species = iris.target_names[prediction]

    st.write("Predicted Flower Species:", predicted_species)
    