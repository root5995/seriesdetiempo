import streamlit as st
import pandas as pd
from joblib import load
from datetime import datetime
import numpy as np

# -------------------------APP DEPLOYMENT PROCESS------------------------------

# 01 --------------------------Load the Fitted Model-------------------------------------------
try:
    fitted_model = load('modelo_sarima.joblib')
except FileNotFoundError:
    st.error("Error: The fitted SARIMA model file 'sarima_results.joblib' was not found. Please ensure your trained model is in the same folder as this script.")
    st.stop()

# ------------------------Centered Title and Description----------------------------------
st.title("🌡️ Predicción de temperatura en Curitiba")
st.markdown("Esta app predice temperaturas promedio para la ciudad de Curitiba en un periodo seleccionado.")
st.markdown("---")

# -----------------------------------Prediction Form-------------------------------------
with st.form("temperature_form"):
    col1, col2 = st.columns(2)

    # Input fields in the first column
    with col1:
        start_year = st.number_input("**Año de inicio**", min_value=1900, max_value=2100, value=2025, step=1)
        months_to_predict = st.number_input("**Número de meses a predecir**", min_value=1, value=12, step=1)

    # Input fields in the second column
    with col2:
        month_options = ["Enero", "Febrero", "Marxo", "Abril", "Mayo", "Junio", "Julio",
                         "Agosto", "Setiembre", "Octubre", "Noviembre", "Diciembre"]
        start_month_str = st.selectbox("**Mes de inicio**", month_options)
        start_month_int = month_options.index(start_month_str) + 1

    predict_button = st.form_submit_button("Predecir temperaturas")

# Validate and execute prediction
if predict_button:
    try:
        # Define the start date for the forecast
        start_date = datetime(start_year, start_month_int, 1)

        # The end date is determined by the number of months to forecast
        end_date = start_date + pd.DateOffset(months=months_to_predict - 1)

        # Perform the prediction using the SARIMA model's `get_prediction` method
        forecast = fitted_model.get_prediction(start=start_date, end=end_date)
        
        # Extract the forecast values and their confidence intervals
        forecast_values = forecast.predicted_mean
        forecast_ci = forecast.conf_int()

        # CORRECTED PART: Use iloc to access columns by position (0 and 1)
        # This prevents errors if column names are not 'lower y' and 'upper y'
        results_df = pd.DataFrame({
            'Fecha a pronosticar': forecast_values.index,
            'Pronoóstico de temperatura': forecast_values.values,
            'IC inferior': forecast_ci.iloc[:, 0].values, # Access the first column
            'IC superiorI': forecast_ci.iloc[:, 1].values  # Access the second column
        })
        
        st.subheader("Resultados")
        st.dataframe(results_df.round(2))

        st.success(f"Se pronosticó exitosamente la temperatura para {months_to_predict} meses!")

    except Exception as e:
        st.error(f"Ha ocurrido un error: {e}. Por favor, revisa tus inputs y modelo.")

# ---------------------------Reset Button-------------------------------------
if st.button("Resetear"):
    st.rerun()