import streamlit as st
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load trained model
model = joblib.load('diabetes_model.pkl')

# Title
st.title('Diabetes Prediction App')

# User input features in sidebar
st.sidebar.header('Input Features')
pregnancies = st.sidebar.slider('Pregnancies', 0, 17, 1)
glucose = st.sidebar.slider('Glucose', 0, 200, 120)
blood_pressure = st.sidebar.slider('Blood Pressure', 0, 122, 70)
skin_thickness = st.sidebar.slider('Skin Thickness', 0, 99, 20)
insulin = st.sidebar.slider('Insulin', 0, 846, 80)
bmi = st.sidebar.slider('BMI', 0.0, 67.1, 32.0)
dpf = st.sidebar.slider('Diabetes Pedigree Function', 0.0, 2.5, 0.5)
age = st.sidebar.slider('Age', 21, 81, 30)

# Input features array
input_features = np.array([pregnancies, glucose, blood_pressure, skin_thickness, 
                           insulin, bmi, dpf, age]).reshape(1, -1)

# Prediction
if st.sidebar.button('Predict'):
    prediction = model.predict(input_features)
    probability = model.predict_proba(input_features)
    st.write(f'### Prediction: {"Diabetic" if prediction[0] == 1 else "Not Diabetic"}')
    st.write(f'Probability of Not Diabetic: {probability[0][0]:.2f}')
    st.write(f'Probability of Diabetic: {probability[0][1]:.2f}')

# Optional: Visualize dataset
if st.checkbox('Show Data Visualization'):
    df = pd.read_csv('diabetes.csv')
    fig, ax = plt.subplots()
    sns.scatterplot(x='Glucose', y='BMI', hue='Outcome', data=df, ax=ax)
    st.pyplot(fig)