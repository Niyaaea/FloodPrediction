# Import Libraries
import streamlit as st
import joblib
import pandas as pd

# Load Model and Scaler
try:
    model = joblib.load('random_forest_model.pkl')
    scaler = joblib.load('scaler.pkl')
except Exception as e:
    st.error(f"Error loading model or scaler: {e}")

# App Title and Description
st.title("Flood Probability Prediction App")
st.write("Adjust the environmental factors to predict flood probability.")

# Input Sliders for 20 Features
features = [
    'MonsoonIntensity', 'TopographyDrainage', 'RiverManagement', 'Deforestation', 
    'Urbanization', 'ClimateChange', 'DamsQuality', 'Siltation', 'AgriculturalPractices', 
    'Encroachments', 'IneffectiveDisasterPreparedness', 'DrainageSystems', 
    'CoastalVulnerability', 'Landslides', 'Watersheds', 'DeterioratingInfrastructure', 
    'PopulationScore', 'WetlandLoss', 'InadequatePlanning', 'PoliticalFactors'
]

# Create sliders dynamically
input_data = {}
for feature in features:
    input_data[feature] = st.slider(feature, min_value=1, max_value=10, value=5)

# Convert inputs into DataFrame
input_df = pd.DataFrame([input_data])

# Scale Inputs and Predict
if st.button('Predict Flood Probability'):
    try:
        scaled_input = scaler.transform(input_df)
        prediction = model.predict(scaled_input)
        st.write(f"Predicted Flood Probability: {prediction[0]:.2f}")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
