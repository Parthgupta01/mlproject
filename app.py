import streamlit as st
import pickle
import numpy as np

# Load the trained Random Forest model
model = pickle.load(open("boston_house_price_predictor.pkl", "rb"))

st.set_page_config(page_title="Boston House Price Predictor", page_icon="🏠")

st.title("🏠 Boston House Price Predictor")
st.write("Enter the feature values to predict the house price.")

# Input sliders / boxes
lstat = st.slider("LSTAT (% lower status of population)", 0.0, 40.0, 12.5)
rm = st.slider("RM (Average number of rooms per dwelling)", 3.0, 10.0, 6.5)
ptratio = st.slider("PTRATIO (Pupil-teacher ratio by town)", 10.0, 30.0, 18.0)
indus = st.slider("INDUS (Proportion of non-retail business acres per town)", 0.0, 30.0, 2.5)
tax = st.slider("TAX (Full-value property-tax rate per $10,000)", 100, 700, 300)
nox = st.slider("NOX (Nitric oxides concentration)", 0.2, 1.0, 0.55)

# Predict button
if st.button("Predict Price"):
    features = np.array([[lstat, rm, ptratio, indus, tax, nox]])
    prediction = model.predict(features)[0]
    st.success(f"Predicted House Price: ${prediction:,.2f}")
