# app.py

import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import joblib
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import boxcox
from PIL import Image
import shap
import streamlit.components.v1 as components
from datetime import datetime

# --- Helper Function to Round Up Values ---
def round_up(x, base=500):
    return int(np.ceil(x / base)) * base

# --- Function to Display SHAP Plots in Streamlit ---
def st_shap(plot, height=None):
    """Display a SHAP plot in Streamlit."""
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

# --- Set Page Configuration ---
st.set_page_config(
    page_title="🏠 House Price Prediction Dashboard",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- Absolute Path Configuration ---
# Define absolute paths to models and data directories
MODELS_DIR = 'data/models'
DATA_DIR = 'data'

# --- Load Models and Data ---
@st.cache_resource
def load_models_and_data(models_dir, data_dir):
    """Load models and datasets from specified directories."""
    models = {}
    model_files = {
        'Linear Regression': 'linear_regression_model.joblib',
        'Ridge Regression': 'ridge_regression_model.joblib',
        'ElasticNet': 'elasticnet_model.joblib',
        'Lasso Regression': 'lasso_regression_model.joblib',
        'Gradient Boosting': 'gradient_boosting_model.joblib',
        'Random Forest': 'random_forest_model.joblib',
        'XGBoost': 'xgboost_model.joblib',
    }

    # Load models
    for name, file in model_files.items():
        model_path = os.path.join(models_dir, file)
        if os.path.exists(model_path):
            try:
                models[name] = joblib.load(model_path)
            except Exception as e:
                st.error(f"Error loading model '{name}': {e}")
        else:
            st.error(f"Model file '{file}' not found in '{models_dir}'.")

    # Load preprocessing objects
    scaler_path = os.path.join(models_dir, 'scaler.joblib')
    selected_features_path = os.path.join(models_dir, 'selected_features.pkl')
    lam_dict_path = os.path.join(models_dir, 'lam_dict.pkl')
    skewed_features_path = os.path.join(models_dir, 'skewed_features.pkl')

    # Initialize variables
    scaler = None
    selected_features = []
    lam_dict = {}
    skewed_features = []

    # Load scaler
    if os.path.exists(scaler_path):
        try:
            scaler = joblib.load(scaler_path)
        except Exception as e:
            st.error(f"Error loading scaler: {e}")
    else:
        st.error(f"Scaler file not found at '{scaler_path}'.")

    # Load selected_features
    if os.path.exists(selected_features_path):
        try:
            with open(selected_features_path, 'rb') as f:
                selected_features = pickle.load(f)
        except Exception as e:
            st.error(f"Error loading selected features: {e}")
    else:
        st.error(f"Selected features file not found at '{selected_features_path}'.")

    # Load lam_dict
    if os.path.exists(lam_dict_path):
        try:
            with open(lam_dict_path, 'rb') as f:
                lam_dict = pickle.load(f)
        except Exception as e:
            st.error(f"Error loading lam_dict: {e}")
    else:
        st.error(f"Lam_dict file not found at '{lam_dict_path}'.")

    # Load skewed_features
    if os.path.exists(skewed_features_path):
        try:
            with open(skewed_features_path, 'rb') as f:
                skewed_features = pickle.load(f)
        except Exception as e:
            st.error(f"Error loading skewed features: {e}")
    else:
        st.error(f"Skewed features file not found at '{skewed_features_path}'.")

    # Load datasets
    house_data_file = os.path.join(data_dir, 'house_prices_records.csv')
    evaluation_file = os.path.join(models_dir, 'model_evaluation.csv')
    inherited_predictions_file = os.path.join(models_dir, 'inherited_houses_predictions.csv')

    # Load house_data
    if os.path.exists(house_data_file):
        try:
            house_data = pd.read_csv(house_data_file)
        except Exception as e:
            st.error(f"Error loading house data: {e}")
            house_data = pd.DataFrame()
    else:
        st.error(f"House data file not found at '{house_data_file}'.")
        house_data = pd.DataFrame()

    # Load evaluation_df
    if os.path.exists(evaluation_file):
        try:
            evaluation_df = pd.read_csv(evaluation_file)
        except Exception as e:
            st.error(f"Error loading evaluation data: {e}")
            evaluation_df = pd.DataFrame()
    else:
        st.error(f"Evaluation data file not found at '{evaluation_file}'.")
        evaluation_df = pd.DataFrame()

    # Load inherited_predictions
    if os.path.exists(inherited_predictions_file):
        try:
            inherited_predictions = pd.read_csv(inherited_predictions_file)
        except Exception as e:
            st.error(f"Error loading inherited predictions: {e}")
            inherited_predictions = pd.DataFrame()
    else:
        st.error(f"Inherited predictions file not found at '{inherited_predictions_file}'.")
        inherited_predictions = pd.DataFrame()

    return models, scaler, selected_features, lam_dict, skewed_features, house_data, evaluation_df, inherited_predictions

# Load everything
(models, scaler, selected_features, lam_dict, skewed_features,
 house_data, evaluation_df, inherited_predictions) = load_models_and_data(MODELS_DIR, DATA_DIR)

# --- Check if Essential Components are Loaded ---
if not models or scaler is None or not selected_features or house_data.empty:
    st.warning("Some models or data files are missing. Please ensure all files are present in the specified directories.")
    st.stop()

# --- Feature Display Names with Units ---
feature_display = {
    'OverallQual': 'Overall Quality (%)',
    'GrLivArea': 'Above Grade Living Area',
    'TotalBsmtSF': 'Total Basement Area',
    'GarageArea': 'Garage Area',
    'YearBuilt': 'Year Built',
    '1stFlrSF': 'First Floor Area',
    'LotArea': 'Lot Area',
    '2ndFlrSF': 'Second Floor Area',
    'BedroomAbvGr': 'Bedrooms Above Ground',
    'BsmtExposure': 'Basement Exposure',
    'BsmtFinType1': 'Basement Finished Type',
    'BsmtFinSF1': 'Basement Finished Area',
    'BsmtUnfSF': 'Basement Unfinished Area',
    'KitchenQual': 'Kitchen Quality',
    'LotFrontage': 'Lot Frontage',
    'MasVnrArea': 'Masonry Veneer Area',
    'EnclosedPorch': 'Enclosed Porch',
    'OpenPorchSF': 'Open Porch Square Feet',
    'OverallCond': 'Overall Condition',
    'WoodDeckSF': 'Wood Deck Area',
    'GarageYrBlt': 'Garage Year Built'
}

# --- Helper Functions ---
def preprocess_input(input_data):
    """Preprocess user input data for prediction."""
    # Convert input data to DataFrame
    df = pd.DataFrame([input_data])

    # Handle missing values
    zero_fill_features = ['2ndFlrSF', 'EnclosedPorch', 'MasVnrArea', 'WoodDeckSF',
                          'BsmtFinSF1', 'TotalBsmtSF', '1stFlrSF', 'BsmtUnfSF']
    for feature in zero_fill_features:
        if feature not in df.columns:
            df[feature] = 0
        df[feature].fillna(0, inplace=True)

    # Fill other missing values with appropriate statistics or defaults
    fill_values = {
        'BedroomAbvGr': house_data['BedroomAbvGr'].mode()[0] if 'BedroomAbvGr' in house_data.columns else 3,
        'BsmtFinType1': 'None',
        'GarageFinish': 'Unf',
        'BsmtExposure': 'No',
        'KitchenQual': 'TA',
        'GarageYrBlt': int(house_data['GarageYrBlt'].median()) if 'GarageYrBlt' in house_data.columns else datetime.now().year,
        'LotFrontage': int(house_data['LotFrontage'].median()) if 'LotFrontage' in house_data.columns else 50,
        'OverallQual': int(house_data['OverallQual'].median()) if 'OverallQual' in house_data.columns else 5,
        'OverallCond': int(house_data['OverallCond'].median()) if 'OverallCond' in house_data.columns else 5,
        'YearBuilt': int(house_data['YearBuilt'].median()) if 'YearBuilt' in house_data.columns else 2000,
        'YearRemodAdd': int(house_data['YearRemodAdd'].median()) if 'YearRemodAdd' in house_data.columns else datetime.now().year
    }

    for feature, value in fill_values.items():
        if feature not in df.columns:
            df[feature] = value
        df[feature].fillna(value, inplace=True)

    # Encode categorical features
    ordinal_mappings = {
        'BsmtFinType1': {'None': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6},
        'KitchenQual': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
        'BsmtExposure': {'No': 0, 'Mn': 1, 'Av': 2, 'Gd': 3, 'None': 0},
        'GarageFinish': {'Unf': 1, 'RFn': 2, 'Fin': 3, 'None': 0}
    }
    for col, mapping in ordinal_mappings.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)
            # If mapping results in NaN, fill with 0 (assuming 'None')
            df[col].fillna(0, inplace=True)

    # Transform skewed features
    for feat in skewed_features:
        if feat in df.columns:
            data = df[feat]
            if (data <= 0).any():
                df[feat] = np.log1p(data)
            else:
                lam = lam_dict.get(feat)
                if lam is None or data.nunique() <= 1 or data.isnull().any():
                    df[feat] = np.log1p(data)
                else:
                    try:
                        transformed = boxcox(data.values + 1e-6, lmbda=lam)
                        if len(transformed) == len(data):
                            df[feat] = transformed
                        else:
                            df[feat] = np.log1p(data)
                    except ValueError:
                        df[feat] = np.log1p(data)

    # Feature engineering
    df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
    df['Qual_TotalSF'] = df['OverallQual'] * df['TotalSF']

    # Ensure the dataset matches selected features
    df = df.reindex(columns=selected_features, fill_value=0)

    # Scaling
    df_scaled = scaler.transform(df)

    return df_scaled

def get_input(feature, label, is_required):
    """Generate appropriate input widgets based on feature type."""
    unique_key = f"{feature}_input"

    categorical_features = ['BsmtExposure', 'BsmtFinType1', 'GarageFinish', 'KitchenQual']

    if feature in categorical_features:
        # Categorical selectbox with full descriptions
        unique_values = house_data[feature].dropna().unique().tolist()
        unique_values = sorted(unique_values)
        selection = st.selectbox(label, options=unique_values, key=unique_key, label_visibility='collapsed')
        # Handle 'None' or 'NA' representations
        if selection in ['None', 'N/A']:
            return 'None'
        return selection

    elif feature in ['YearBuilt', 'YearRemodAdd', 'GarageYrBlt']:
        # Number input for year features
        current_year = datetime.now().year
        default_year = int(house_data[feature].median()) if feature in house_data.columns else current_year
        selected_year = st.number_input(label, min_value=1800, max_value=current_year, value=default_year, step=1, key=unique_key, label_visibility='collapsed')
        return selected_year

    elif feature in ['1stFlrSF', '2ndFlrSF', 'TotalBsmtSF', 'GarageArea', 'GrLivArea']:
        # Sliders for square feet inputs with +10% max range and rounded up
        if feature in house_data.columns and pd.api.types.is_numeric_dtype(house_data[feature]):
            min_val = int(house_data[feature].min())
            max_val_raw = house_data[feature].max() * 1.10  # Adding 10% for outliers
            max_val = round_up(max_val_raw, base=500)
            median_val = int(house_data[feature].median())
        else:
            min_val = 0
            max_val = 10000
            median_val = 1000

        slider_val = st.slider(label, min_value=min_val, max_value=max_val, value=median_val, step=50, key=unique_key, label_visibility='collapsed')
        return slider_val  # Units will be displayed within the card

    elif feature in ['LotArea', 'LotFrontage', 'EnclosedPorch', 'OpenPorchSF', 'MasVnrArea', 'WoodDeckSF']:
        # Sliders for square feet inputs with +10% max range and rounded up
        if feature in house_data.columns and pd.api.types.is_numeric_dtype(house_data[feature]):
            min_val = int(house_data[feature].min())
            max_val_raw = house_data[feature].max() * 1.10  # Adding 10% for outliers
            max_val = round_up(max_val_raw, base=500)
            median_val = int(house_data[feature].median())
        else:
            min_val = 0
            max_val = 10000
            median_val = 500

        slider_val = st.slider(label, min_value=min_val, max_value=max_val, value=median_val, step=50, key=unique_key, label_visibility='collapsed')
        return slider_val  # Units will be displayed within the card

    elif feature == 'BedroomAbvGr':
        # Integer slider for bedrooms
        bedrooms = st.slider(label, min_value=1, max_value=10, value=int(house_data['BedroomAbvGr'].median()) if 'BedroomAbvGr' in house_data.columns else 3, step=1, key=unique_key, label_visibility='collapsed')
        return bedrooms  # Units will be displayed within the card

    else:
        # Default number input for other numerical features with rounded up max
        if feature in house_data.columns and pd.api.types.is_numeric_dtype(house_data[feature]):
            min_val = int(house_data[feature].min())
            max_val_raw = house_data[feature].max() * 1.10  # Adding 10% for outliers
            max_val = round_up(max_val_raw, base=1000)
            median_val = int(house_data[feature].median())
        else:
            min_val = 0
            max_val = 100000
            median_val = 0

        slider_val = st.slider(label, min_value=min_val, max_value=max_val, value=median_val, step=500, key=unique_key, label_visibility='collapsed')
        return slider_val  # Units will be displayed within the card

# --- Top Navigation Bar ---
selected_tab = option_menu(
    menu_title=None,  # Remove the menu title
    options=["🏠 Home", "📊 Data Exploration", "📈 Model Performance", "📁 Inherited Houses", "ℹ️ About"],
    icons=["house", "bar-chart", "graph-up-arrow", "folder", "info-circle"],
    menu_icon=None,  # Remove the menu icon
    default_index=0,
    orientation="horizontal",
    styles={
        "container": {"padding": "0!important", "background-color": "#ffffff"},
        "icon": {"color": "#007bff", "font-size": "20px"},
        "nav-link": {"font-size": "16px", "text-align": "center", "margin": "0px", "--hover-color": "#f0f2f6"},
        "nav-link-selected": {"background-color": "#007bff", "color": "white"},
    }
)

# --- Apply Custom CSS for Modern Look ---
st.markdown(
    """
    <style>
    /* Custom CSS styles */
    .stButton>button {
        color: white;
        background-color: #007bff;
        border-radius: 5px;
        padding: 10px 30px;
        font-size: 16px;
        border: none;
        transition: background-color 0.3s ease;
        cursor: pointer;
    }
    .stButton>button:hover {
        background-color: #0056b3;
    }
    .stProgress > div > div > div > div {
        background-color: #007bff;
    }
    .css-18e3th9 {
        padding-top: 1rem;
    }
    .css-1d391kg {
        padding-top: 0rem;
    }
    .main {
        background-color: #f9f9f9;
    }
    /* Adjust font sizes and styles */
    h1, h2, h3 {
        color: #333333;
        font-family: 'Arial', sans-serif;
    }
    /* Style tables */
    .dataframe {
        background-color: white;
        border-radius: 5px;
    }
    /* Modern scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #f1f1f1; 
    }
    ::-webkit-scrollbar-thumb {
        background: #888; 
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #555; 
    }
    /* Expander style */
    div[role="button"] > div {
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    /* Input Card Styling */
    .input-card {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 15px;
        background-color: #ffffff;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        margin-bottom: 15px;
    }
    /* Label Styling */
    .input-label {
        font-weight: bold;
        color: #333333;
        display: block;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Home Tab ---
if selected_tab == "🏠 Home":
    st.title("🏠 House Price Prediction")
    st.write("""
    Welcome to the **House Price Prediction Dashboard**. This tool allows you to input various features of a house
    and get an estimated sale price using advanced machine learning models.
    """)

    # Define the features based on your metadata
    feature_list = [
        '1stFlrSF', '2ndFlrSF', 'BedroomAbvGr', 'BsmtExposure', 'BsmtFinType1',
        'BsmtFinSF1', 'BsmtUnfSF', 'TotalBsmtSF', 'GarageArea', 'GarageFinish',
        'GarageYrBlt', 'GrLivArea', 'KitchenQual', 'LotArea', 'LotFrontage',
        'MasVnrArea', 'EnclosedPorch', 'OpenPorchSF', 'OverallCond', 'OverallQual',
        'WoodDeckSF', 'YearBuilt', 'YearRemodAdd'
    ]

    # Identify top 7 features based on feature importance (for highlighting required fields)
    top_7_features = [
        'OverallQual', 'GrLivArea', 'TotalBsmtSF', 'GarageArea',
        'YearBuilt', '1stFlrSF', 'LotArea'
    ]

    # Collect user inputs
    st.header("Enter House Features")
    input_features = {}

    st.write("**Please fill out the required features. Additional features are optional.**")

    # Display top features in a two-column layout
    top_features = [feat for feat in feature_list if feat in top_7_features]
    other_features = [feat for feat in feature_list if feat not in top_7_features]

    cols = st.columns(2)
    for i, feature in enumerate(top_features):
        label = feature_display.get(feature, feature)
        with cols[i % 2]:
            # Create a bordered card for each input
            st.markdown(
                f"""
                <div class="input-card">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span class="input-label">{label}</span>
                        <span class="input-value" id="{unique_key}_value">0</span>
                    </div>
                """,
                unsafe_allow_html=True,
            )
            # Call the widget with label_visibility='collapsed'
            value = get_input(feature, label, is_required=True)

            # Determine display value with units
            if feature in ['1stFlrSF', '2ndFlrSF', 'TotalBsmtSF', 'GarageArea', 'GrLivArea',
                          'LotArea', 'LotFrontage', 'MasVnrArea', 'EnclosedPorch', 'OpenPorchSF', 'WoodDeckSF']:
                display_val = f"{value} sqft"
            elif feature == 'BedroomAbvGr':
                display_val = f"{value} bedrooms"
            else:
                display_val = f"{value}"

            # Display the value on the right side
            st.markdown(
                f"""
                <script>
                document.getElementById("{unique_key}_value").innerText = "{display_val}";
                </script>
                """,
                unsafe_allow_html=True,
            )

            st.markdown('</div>', unsafe_allow_html=True)
            input_features[feature] = value

    # Expandable section for additional features
    with st.expander("🔍 Add More Features"):
        cols_exp = st.columns(2)
        for i, feature in enumerate(other_features):
            label = feature_display.get(feature, feature)
            with cols_exp[i % 2]:
                # Create a bordered card for each input
                unique_key = f"{feature}_input_exp"
                st.markdown(
                    f"""
                    <div class="input-card">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <span class="input-label">{label}</span>
                            <span class="input-value" id="{unique_key}_value">0</span>
                        </div>
                    """,
                    unsafe_allow_html=True,
                )
                # Call the widget with label_visibility='collapsed'
                value = get_input(feature, label, is_required=False)

                # Determine display value with units
                if feature in ['1stFlrSF', '2ndFlrSF', 'TotalBsmtSF', 'GarageArea', 'GrLivArea',
                              'LotArea', 'LotFrontage', 'MasVnrArea', 'EnclosedPorch', 'OpenPorchSF', 'WoodDeckSF']:
                    display_val = f"{value} sqft"
                elif feature == 'BedroomAbvGr':
                    display_val = f"{value} bedrooms"
                else:
                    display_val = f"{value}"

                # Display the value on the right side
                st.markdown(
                    f"""
                    <script>
                    document.getElementById("{unique_key}_value").innerText = "{display_val}";
                    </script>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown('</div>', unsafe_allow_html=True)
                input_features[feature] = value

    # Predict button
    st.markdown(
        """
        <div style="text-align: center;">
            <button class="predict-button">🔮 Predict</button>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Add CSS for the Predict button
    st.markdown(
        """
        <style>
        .predict-button {
            color: white;
            background-color: #007bff;
            border: none;
            border-radius: 5px;
            padding: 10px 30px;
            font-size: 16px;
            cursor: pointer;
            transition: background-color 0.3s ease;
            margin-top: 20px;
        }
        .predict-button:hover {
            background-color: #0056b3;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Use a hidden button for triggering prediction
    predict = st.button("🔮 Predict")

    if predict:
        # Preprocess input
        try:
            input_data = preprocess_input(input_features)
        except Exception as e:
            st.error(f"Error during preprocessing: {e}")
            st.stop()

        # Make predictions
        predictions = {}
        for name, model in models.items():
            try:
                pred_log = model.predict(input_data)
                pred_price = np.expm1(pred_log)[0]
                if np.isinf(pred_price) or np.isnan(pred_price):
                    pred_price = np.nan
                predictions[name] = pred_price
            except Exception as e:
                predictions[name] = np.nan
                st.error(f"Error predicting with {name}: {e}")

        # Display predictions
        st.subheader("💰 Predicted Sale Prices")
        pred_df = pd.DataFrame(list(predictions.items()), columns=['Model', 'Predicted Sale Price'])
        pred_df['Predicted Sale Price'] = pred_df['Predicted Sale Price'].apply(
            lambda x: f"${int(x):,}" if pd.notnull(x) else "N/A"
        )
        st.table(pred_df.style.set_properties(**{'text-align': 'left'}).set_table_styles([
            dict(selector='th', props=[('text-align', 'left')])
        ]))

        # Visualize predictions
        st.subheader("📊 Prediction Comparison")
        fig_pred, ax_pred = plt.subplots(figsize=(10, 6))
        # Convert 'Predicted Sale Price' back to numeric for plotting
        plot_df = pred_df[pred_df['Predicted Sale Price'] != "N/A"].copy()
        plot_df['Predicted Sale Price'] = plot_df['Predicted Sale Price'].replace('[\$,]', '', regex=True).astype(float)
        sns.barplot(x='Predicted Sale Price', y='Model', data=plot_df, palette='Blues_d', ax=ax_pred)
        ax_pred.set_xlabel('Predicted Sale Price ($)')
        ax_pred.set_ylabel('Model')
        ax_pred.set_title('Comparison of Predicted Sale Prices by Model')
        st.pyplot(fig_pred)

        # SHAP explanation (using XGBoost model)
        st.subheader("🔍 Model Explanation")
        st.write("Below is the SHAP analysis showing the impact of each feature on the prediction for the **XGBoost** model.")

        try:
            # Ensure the XGBoost model is present
            if 'XGBoost' in models:
                model = models['XGBoost']
                explainer = shap.Explainer(model)
                shap_values = explainer(pd.DataFrame(input_data, columns=selected_features))

                # Display SHAP force plot
                st_shap(shap.force_plot(explainer.expected_value, shap_values.values, pd.DataFrame(input_data, columns=selected_features)), height=300)
            else:
                st.error("XGBoost model not found.")
        except Exception as e:
            st.error(f"Error generating SHAP explanation: {e}")

# --- Data Exploration Tab ---
elif selected_tab == "📊 Data Exploration":
    st.title("📊 Data Exploration")
    st.write("Explore the dataset used to train the models.")

    # Display dataset
    st.subheader("📂 House Prices Dataset")
    st.write("Below is a preview of the dataset:")
    st.dataframe(house_data.head(50))

    # Interactive plots
    st.subheader("🔗 Feature Correlations")
    st.write("The heatmap below shows the correlations between different features in the dataset.")

    # Select only numerical columns for correlation
    numerical_data = house_data.select_dtypes(include=[np.number])

    if numerical_data.empty:
        st.error("No numerical data available for correlation analysis.")
    else:
        fig_corr, ax_corr = plt.subplots(figsize=(12, 10))
        correlation_matrix = numerical_data.corr()
        sns.heatmap(correlation_matrix, cmap='coolwarm', ax=ax_corr, annot=True, fmt=".2f", linewidths=0.5)
        ax_corr.set_title('🟢 Correlation Heatmap of Features')
        ax_corr.set_xlabel('Features')
        ax_corr.set_ylabel('Features')
        st.pyplot(fig_corr)

    st.subheader("📈 Sale Price Distribution")
    st.write("The histogram below shows the distribution of sale prices in the dataset.")
    fig_price, ax_price = plt.subplots(figsize=(10, 6))
    sns.histplot(house_data['SalePrice'], kde=True, ax=ax_price, color='green')
    ax_price.set_title('🟢 Distribution of Sale Prices')
    ax_price.set_xlabel('Sale Price ($)')
    ax_price.set_ylabel('Frequency')
    st.pyplot(fig_price)

# --- Model Performance Tab ---
elif selected_tab == "📈 Model Performance":
    st.title("📈 Model Performance")
    st.write("Review the performance metrics of each model.")

    # Display model evaluation metrics with descriptions
    st.subheader("📊 Model Evaluation Metrics")
    st.write("""
    - **MAE (Mean Absolute Error):** Average of absolute differences between predictions and actual values.
    - **RMSE (Root Mean Squared Error):** Square root of average squared differences between predictions and actual values.
    - **R² Score:** Proportion of variance in the dependent variable predictable from the independent variables.
    """)

    if evaluation_df.empty:
        st.error("No evaluation data available.")
    else:
        st.table(evaluation_df.style.format({"MAE": "{:,.2f}", "RMSE": "{:,.2f}", "R² Score": "{:.4f}"}))

        # Dynamic Model Descriptions based on performance
        st.subheader("📝 Model Descriptions")
        for _, row in evaluation_df.iterrows():
            model_name = row['Model']
            mae = row['MAE']
            rmse = row['RMSE']
            r2 = row['R² Score']

            # Generate descriptions based on metrics
            if r2 >= 0.8:
                performance = "Excellent"
                strength = "high accuracy and reliability in predictions."
                weakness = "may require more computational resources."
            elif r2 >= 0.6:
                performance = "Good"
                strength = "balanced accuracy and computational efficiency."
                weakness = "might not capture all complex patterns."
            elif r2 >= 0.4:
                performance = "Fair"
                strength = "simple and interpretable."
                weakness = "limited accuracy and may overlook important patterns."
            else:
                performance = "Poor"
                strength = "very simple models with minimal computational needs."
                weakness = "significantly lower accuracy and reliability."

            description = f"""
            **{model_name}**
            - **Performance:** {performance}
            - **MAE:** {mae:,.2f}
            - **RMSE:** {rmse:,.2f}
            - **R² Score:** {r2:.4f}

            **Overview:** The **{model_name}** model demonstrates **{performance.lower()}** performance, indicating {strength} However, it may {weakness}
            """

            st.markdown(description)

        # Visual comparison
        st.subheader("📉 Performance Comparison")
        metrics = ['MAE', 'RMSE', 'R² Score']
        for metric in metrics:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x='Model', y=metric, data=evaluation_df, palette='viridis', ax=ax)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            ax.set_title(f'📉 Model Comparison on {metric}')
            ax.set_ylabel(metric)
            ax.set_xlabel('Model')
            st.pyplot(fig)

# --- Inherited Houses Tab ---
elif selected_tab == "📁 Inherited Houses":
    st.title("📁 Inherited Houses Predictions")
    st.write("Review the predicted sale prices for the inherited houses.")

    # Handle any infinite or NaN values
    inherited_predictions = inherited_predictions.replace([np.inf, -np.inf], np.nan)
    inherited_predictions.fillna(inherited_predictions.mean(), inplace=True)

    st.subheader("💰 Predictions")
    st.write("Below are the predicted sale prices for each inherited house using different models.")
    st.dataframe(inherited_predictions.style.format("${:,.2f}"))

    # Average predictions
    st.subheader("📊 Average Predicted Sale Prices")
    avg_prices = inherited_predictions.mean().reset_index()
    avg_prices.columns = ['Model', 'Average Predicted Sale Price']
    st.table(avg_prices.style.format({"Average Predicted Sale Price": "${:,.2f}"}))

    # Visual comparison
    st.subheader("📈 Predicted Prices Comparison")
    fig_pred, ax_pred = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=inherited_predictions, palette='Set2', ax=ax_pred)
    ax_pred.set_ylabel('Predicted Sale Price ($)')
    ax_pred.set_title('📈 Distribution of Predicted Sale Prices by Model')
    st.pyplot(fig_pred)

# --- About Tab ---
elif selected_tab == "ℹ️ About":
    st.title("ℹ️ About")
    st.write("""
    ### House Price Prediction Dashboard
    Developed to provide accurate house price predictions using multiple advanced machine learning models.

    **Features:**
    - **User-Friendly Interface:** Modern design with intuitive navigation.
    - **Multiple Models:** Compare predictions from various models.
    - **Data Exploration:** Interactive visuals to understand the dataset.
    - **Model Performance:** Detailed metrics and comparison charts.
    - **Inherited Houses:** Specific predictions for inherited properties.

    **Developed By:**
    - *Your Name*
    - *Your Professional Title*
    - *Contact Information*

    **Technologies Used:**
    - **Python:** Data processing and model development.
    - **Streamlit:** Web app development.
    - **Machine Learning Libraries:** scikit-learn, XGBoost, SHAP.

    **Acknowledgments:**
    - Dataset provided by *[Dataset Source]*.
    """)

    # Display an image (optional)
    image_path = os.path.join(os.getcwd(), 'dashboard', 'images', 'house_image.jpg')
    if os.path.exists(image_path):
        image = Image.open(image_path)
        st.image(image, caption='House Price Prediction', use_column_width=True)
    else:
        # Fallback to an online image if local image is not found
        st.write("![House Image](https://images.unsplash.com/photo-1560184897-6ec56aed1c8d)")

# --- End of app.py ---