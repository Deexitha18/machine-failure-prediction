import streamlit as st
import pandas as pd
import joblib

# Load trained Random Forest model
model = joblib.load("machine_failure_model.pkl")

# Title
st.title("Machine Failure Risk Prediction System")

st.write("Enter the machine parameters to predict failure risk.")

# Machine inputs
air_temperature = st.number_input(
    "Air Temperature [K]",
    value=305.0
)

process_temperature = st.number_input(
    "Process Temperature [K]",
    value=315.0
)

rotational_speed = st.number_input(
    "Rotational Speed [rpm]",
    value=1400
)

torque = st.number_input(
    "Torque [Nm]",
    value=70.0
)

tool_wear = st.number_input(
    "Tool Wear [min]",
    value=200
)

# Predict button
if st.button("Predict"):

    # Create input DataFrame
    new_machine = pd.DataFrame({
        "Air temperature [K]": [air_temperature],
        "Process temperature [K]": [process_temperature],
        "Rotational speed [rpm]": [rotational_speed],
        "Torque [Nm]": [torque],
        "Tool wear [min]": [tool_wear]
    })

    # Prediction
    prediction = model.predict(new_machine)

    # Failure probability
    failure_probability = model.predict_proba(
        new_machine
    )[0][1]

    # Risk level
    if failure_probability >= 0.70:
        risk = "HIGH RISK"
    elif failure_probability >= 0.40:
        risk = "MEDIUM RISK"
    else:
        risk = "LOW RISK"

    # Display prediction
    if prediction[0] == 1:
        st.error("Prediction: MACHINE FAILURE")
    else:
        st.success("Prediction: NO FAILURE")

    # Display probability
    st.write(
        "Failure Probability:",
        round(failure_probability * 100, 2),
        "%"
    )

    # Display risk
    st.write("Risk Level:", risk)