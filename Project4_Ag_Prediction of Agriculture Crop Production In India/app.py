import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import os
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Agriculture Crop Prediction India",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Define Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Features expected by models based on analysis
yield_features = [
    'cost_cultivation_A2_FL', 'cost_cultivation_C2', 'cost_production_C2',
    'crop_encoded', 'state_encoded', 'region_encoded', 'season_duration_days',
    'yield_kg_per_ha', 'area_million_ha', 'production_million_tons',
    'cost_efficiency', 'profitability_index', 'cost_area_interaction',
    'yield_cost_ratio', 'production_efficiency'
]

prod_features = [
    'year', 'year_squared', 'year_cubic', 'area_million_ha', 'yield_kg_per_ha',
    'cost_cultivation_C2', 'cost_production_C2', 'production_lag_1',
    'production_lag_2', 'production_rolling_3', 'production_growth', 'crop_encoded'
]

# --- Load Models and Scalers ---
@st.cache_resource
def load_models_and_data():
    try:
        # Load Models
        with open(os.path.join(MODELS_DIR, 'best_yield_model.pkl'), 'rb') as f:
            yield_model = joblib.load(f)
            
        with open(os.path.join(MODELS_DIR, 'best_production_model.pkl'), 'rb') as f:
            prod_model = joblib.load(f)
            
        # Load Scalers
        with open(os.path.join(MODELS_DIR, 'scaler_yield.pkl'), 'rb') as f:
            scaler_yield = joblib.load(f)
            
        with open(os.path.join(MODELS_DIR, 'scaler_prod.pkl'), 'rb') as f:
            scaler_prod = joblib.load(f)
            
        # Load Encoders
        with open(os.path.join(MODELS_DIR, 'label_encoder_crop.pkl'), 'rb') as f:
            le_crop = joblib.load(f)
            
        with open(os.path.join(MODELS_DIR, 'label_encoder_state.pkl'), 'rb') as f:
            le_state = joblib.load(f)
            
        # Load Datasets for sampling
        df_yield = pd.read_csv(os.path.join(BASE_DIR, 'master_table_with_features.csv'))
        df_prod = pd.read_csv(os.path.join(BASE_DIR, 'production_with_features.csv'))
        
        # Fit a label encoder for crop in production data to recreate crop_encoded
        from sklearn.preprocessing import LabelEncoder
        le_crop_prod = LabelEncoder()
        df_prod['crop_encoded'] = le_crop_prod.fit_transform(df_prod['crop'])
        
        # Fit the simple imputers using scaled datasets to learn which columns are 100% NaN (dropped)
        from sklearn.impute import SimpleImputer
        
        df_yield_scaled = scaler_yield.transform(df_yield[yield_features])
        imputer_yield = SimpleImputer(strategy='median')
        imputer_yield.fit(df_yield_scaled)
        
        df_prod_scaled = scaler_prod.transform(df_prod[prod_features])
        imputer_prod = SimpleImputer(strategy='median')
        imputer_prod.fit(df_prod_scaled)
        
        return yield_model, prod_model, scaler_yield, scaler_prod, le_crop, le_state, df_yield, df_prod, imputer_yield, imputer_prod
    except Exception as e:
        st.error(f"Error loading models or data: {e}")
        return None, None, None, None, None, None, None, None, None, None

yield_model, prod_model, scaler_yield, scaler_prod, le_crop, le_state, df_yield, df_prod, imputer_yield, imputer_prod = load_models_and_data()

# Features are now defined at the top.

# --- UI Application ---
st.title("🌾 Agricultural Crop Production & Yield Prediction")
st.markdown("""
Welcome to the Agricultural Prediction Model Demonstration application. 
Use the sidebar to choose between Yield Prediction or Production Prediction models, and select a data sample to test.
""")

st.sidebar.header("Settings")
task = st.sidebar.radio("Select Prediction Task", ("Yield Prediction", "Production Prediction"))

if task == "Yield Prediction":
    st.header("📈 Crop Yield Prediction")
    st.markdown("This model predicts the **Yield (Quintal per Hectare)** based on various costs, crop types, and regional factors.")
    
    if df_yield is not None:
        st.subheader("1. Select a Data Sample")
        # Let user pick a row to populate features
        sample_index = st.slider("Select Sample Index from Dataset", 0, len(df_yield) - 1, 0)
        
        sample_row = df_yield.iloc[sample_index]
        st.write("**Selected Data Details:**")
        
        # Display human readable details
        crop_name = sample_row.get('crop', 'Unknown')
        state_name = sample_row.get('state', 'Unknown')
        st.info(f"**Crop**: {crop_name} | **State**: {state_name} | **Actual Yield**: {sample_row.get('yield_quintal_per_hectare', 'N/A'):.2f} Quintals/Hectare")
        
        st.subheader("2. Model Input Features")
        st.dataframe(sample_row[yield_features].to_frame().T)
        
        if st.button("Run Prediction", type="primary"):
            with st.spinner("Predicting..."):
                input_data = sample_row[yield_features].values.reshape(1, -1)
                
                # We need to impute NaNs since models pipeline does it
                from sklearn.impute import SimpleImputer
                # Note: Assuming median imputation was used, or we just fillna here before scaling
                input_df = pd.DataFrame(input_data, columns=yield_features)
                input_df = input_df.fillna(df_yield[yield_features].median())
                
                # Scale
                input_scaled = scaler_yield.transform(input_df)
                
                # Impute (which drops the columns that were 100% NaN during training)
                input_imputed = imputer_yield.transform(input_scaled)
                
                # If model is a dict (like from GridSearchCV), get the best estimator, otherwise use directly
                model_to_use = yield_model['model'] if isinstance(yield_model, dict) else yield_model
                
                prediction = model_to_use.predict(input_imputed)
                
                st.success(f"### Predicted Yield: {prediction[0]:.2f} Quintals/Hectare")
                
                # Calculate Error
                actual = sample_row.get('yield_quintal_per_hectare', None)
                if pd.notna(actual):
                    error = abs(prediction[0] - actual)
                    st.metric(label="Absolute Error", value=f"{error:.2f}", delta=f"{(prediction[0]-actual):.2f}", delta_color="inverse")
                
    else:
        st.warning("Yield dataset not loaded properly.")

elif task == "Production Prediction":
    st.header("📦 Crop Production Prediction")
    st.markdown("This model predicts the **Production Next Year (Million Tons)** based on historical production data, costs, and area.")
    
    if df_prod is not None:
        st.subheader("1. Select a Data Sample")
        sample_index = st.slider("Select Sample Index from Dataset", 0, len(df_prod) - 1, 0)
        
        sample_row = df_prod.iloc[sample_index]
        st.write("**Selected Data Details:**")
        
        crop_name = sample_row.get('crop', 'Unknown')
        year = sample_row.get('year', 'Unknown')
        st.info(f"**Crop**: {crop_name} | **Year**: {year} | **Actual Production Next Year**: {sample_row.get('production_next_year', 'N/A'):.2f} Million Tons")
        
        st.subheader("2. Model Input Features")
        st.dataframe(sample_row[prod_features].to_frame().T)
        
        if st.button("Run Prediction", type="primary"):
            with st.spinner("Predicting..."):
                input_data = sample_row[prod_features].values.reshape(1, -1)
                
                input_df = pd.DataFrame(input_data, columns=prod_features)
                input_df = input_df.fillna(df_prod[prod_features].mean())
                
                # Scale
                input_scaled = scaler_prod.transform(input_df)
                
                # Impute (which drops the columns that were 100% NaN during training)
                input_imputed = imputer_prod.transform(input_scaled)
                
                # Predict
                model_to_use = prod_model['model'] if isinstance(prod_model, dict) else prod_model
                prediction = model_to_use.predict(input_imputed)
                
                st.success(f"### Predicted Production: {prediction[0]:.2f} Million Tons")
                
                # Calculate Error
                actual = sample_row.get('production_next_year', None)
                if pd.notna(actual):
                    error = abs(prediction[0] - actual)
                    st.metric(label="Absolute Error", value=f"{error:.2f}", delta=f"{(prediction[0]-actual):.2f}", delta_color="inverse")
    else:
        st.warning("Production dataset not loaded properly.")
