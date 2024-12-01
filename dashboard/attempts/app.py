# streamlit_app.py

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
import datetime
import shap

# Set page configuration
st.set_page_config(
    page_title="House Price Prediction Dashboard",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Load Models and Data ---

@st.cache_resource
def load_models_and_data():
    # Adjust the paths below if necessary
    models_dir = 'data/models'
    data_dir = 'data'

    # Load models
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
    for name, file in model_files.items():
        models[name] = joblib.load(os.path.join(models_dir, file))

    # Load preprocessing objects
    scaler = joblib.load(os.path.join(models_dir, 'scaler.joblib'))
    with open(os.path.join(models_dir, 'selected_features.pkl'), 'rb') as f:
        selected_features = pickle.load(f)
    with open(os.path.join(models_dir, 'lam_dict.pkl'), 'rb') as f:
        lam_dict = pickle.load(f)
    with open(os.path.join(models_dir, 'skewed_features.pkl'), 'rb') as f:
        skewed_features = pickle.load(f)

    # Load datasets
    house_data = pd.read_csv(os.path.join(data_dir, 'house_prices_records.csv'))
    evaluation_df = pd.read_csv(os.path.join(models_dir, 'model_evaluation.csv'))
    inherited_predictions = pd.read_csv(os.path.join(models_dir, 'inherited_houses_predictions.csv'))

    return models, scaler, selected_features, lam_dict, skewed_features, house_data, evaluation_df, inherited_predictions

# Load everything
(models, scaler, selected_features, lam_dict, skewed_features,
 house_data, evaluation_df, inherited_predictions) = load_models_and_data()

# --- Helper Functions ---

def preprocess_input(input_data):
    # Convert input data to DataFrame
    df = pd.DataFrame([input_data])

    # Handle missing values
    zero_fill_features = ['2ndFlrSF', 'EnclosedPorch', 'MasVnrArea', 'WoodDeckSF',
                          'BsmtFinSF1', 'TotalBsmtSF', '1stFlrSF', 'BsmtUnfSF']
    for feature in zero_fill_features:
        df[feature] = df.get(feature, 0)
        df[feature].fillna(0, inplace=True)

    df['BedroomAbvGr'] = df.get('BedroomAbvGr', house_data['BedroomAbvGr'].mode()[0])
    df['BedroomAbvGr'].fillna(house_data['BedroomAbvGr'].mode()[0], inplace=True)
    df['BsmtFinType1'] = df.get('BsmtFinType1', 'None')
    df['BsmtFinType1'].fillna('None', inplace=True)
    df['GarageFinish'] = df.get('GarageFinish', 'Unf')
    df['GarageFinish'].fillna('Unf', inplace=True)
    df['BsmtExposure'] = df.get('BsmtExposure', 'No')
    df['BsmtExposure'].fillna('No', inplace=True)
    df['KitchenQual'] = df.get('KitchenQual', 'TA')
    df['KitchenQual'].fillna('TA', inplace=True)
    df['GarageYrBlt'] = df.get('GarageYrBlt', house_data['GarageYrBlt'].median())
    df['GarageYrBlt'].fillna(house_data['GarageYrBlt'].median(), inplace=True)
    df['LotFrontage'] = df.get('LotFrontage', house_data['LotFrontage'].median())
    df['LotFrontage'].fillna(house_data['LotFrontage'].median(), inplace=True)
    df['OverallQual'] = df.get('OverallQual', house_data['OverallQual'].median())
    df['OverallQual'].fillna(house_data['OverallQual'].median(), inplace=True)
    df['OverallCond'] = df.get('OverallCond', house_data['OverallCond'].median())
    df['OverallCond'].fillna(house_data['OverallCond'].median(), inplace=True)
    df['YearBuilt'] = df.get('YearBuilt', house_data['YearBuilt'].median())
    df['YearBuilt'].fillna(house_data['YearBuilt'].median(), inplace=True)
    df['YearRemodAdd'] = df.get('YearRemodAdd', house_data['YearRemodAdd'].median())
    df['YearRemodAdd'].fillna(house_data['YearRemodAdd'].median(), inplace=True)

    # Feature engineering
    df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
    df['Qual_TotalSF'] = df['OverallQual'] * df['TotalSF']

    # Encode categorical features
    ordinal_mappings = {
        'BsmtFinType1': {'None': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6},
        'KitchenQual': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
        'BsmtExposure': {'None': 0, 'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4},
        'GarageFinish': {'None': 0, 'Unf': 1, 'RFn': 2, 'Fin': 3}
    }
    for col, mapping in ordinal_mappings.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)

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

    # Ensure the dataset matches selected features
    df = df.reindex(columns=selected_features, fill_value=0)

    # Scaling
    df_scaled = scaler.transform(df)

    return df_scaled

# --- Sidebar Navigation ---

with st.sidebar:
    selected_tab = option_menu(
        menu_title="Main Menu",
        options=["🏠 Home", "📊 Data Exploration", "📈 Model Performance", "📁 Inherited Houses", "ℹ️ About"],
        icons=["house", "bar-chart", "graph-up-arrow", "folder", "info-circle"],
        menu_icon="cast",
        default_index=0,
    )

# Apply custom CSS for modern look
st.markdown(
    """
    <style>
    /* Custom CSS styles */
    .stButton>button {
        color: white;
        background-color: #007bff;
        border-radius: 5px;
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
        background-color: #f0f2f6;
    }
    /* Adjust font sizes */
    h1, h2, h3 {
        color: #333333;
    }
    /* Style tables */
    .dataframe {
        background-color: white;
        border-radius: 5px;
    }
    /* Style sidebar */
    .css-1d391kg .css-fblp2m {
        background-color: #f8f9fa;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Function to get input based on feature type
def get_input(feature, label, is_required):
    if feature in ['YearBuilt', 'YearRemodAdd', 'GarageYrBlt']:
        # Date input, extract year
        year = st.number_input(
            label, min_value=1872, max_value=2023, value=int(house_data[feature].median()), key=feature)
        return int(year)
    elif feature in ['BsmtExposure', 'BsmtFinType1', 'GarageFinish', 'KitchenQual']:
        # Categorical features with selectbox
        unique_values = house_data[feature].dropna().unique().tolist()
        unique_values = sorted(unique_values)
        return st.selectbox(label, options=unique_values, key=feature)
    else:
        # Numerical features
        default_value = float(house_data[feature].median())
        min_value = float(house_data[feature].min())
        max_value = float(house_data[feature].max())
        return st.number_input(
            label, min_value=min_value, max_value=max_value, value=default_value, key=feature)

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

    # Identify top 7 features based on feature importance
    top_7_features = [
        'OverallQual', 'GrLivArea', 'TotalBsmtSF', 'GarageArea',
        'YearBuilt', '1stFlrSF', 'LotArea'
    ]

    # Collect user inputs
    st.header("Enter House Features")
    input_features = {}

    st.write("**Please fill out the required features (marked in bold). Other features are optional.**")

    # Organize inputs into columns
    cols = st.columns(2)
    for i, feature in enumerate(feature_list):
        if feature in top_7_features:
            is_required = True
            label = f"**{feature}** (Required)"
        else:
            is_required = False
            label = feature

        with cols[i % 2]:
            input_value = get_input(feature, label, is_required)
            input_features[feature] = input_value

    if st.button("Predict"):
        # Preprocess input
        input_data = preprocess_input(input_features)

        # Make predictions
        predictions = {}
        for name, model in models.items():
            pred_log = model.predict(input_data)
            pred_price = np.expm1(pred_log)[0]
            if np.isinf(pred_price) or np.isnan(pred_price):
                pred_price = np.nan
            predictions[name] = pred_price

        # Display predictions
        st.subheader("Predicted Sale Prices")
        pred_df = pd.DataFrame(list(predictions.items()), columns=['Model', 'Predicted Sale Price'])
        pred_df['Predicted Sale Price'] = pred_df['Predicted Sale Price'].apply(lambda x: f"${x:,.2f}" if pd.notnull(x) else "N/A")
        st.table(pred_df.style.set_properties(**{'text-align': 'left'}).set_table_styles([dict(selector='th', props=[('text-align', 'left')])]))

        # Visualize predictions
        st.subheader("Prediction Comparison")
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
        st.subheader("Model Explanation")
        st.write("Below is the SHAP analysis showing the impact of each feature on the prediction for the XGBoost model.")

        # Prepare data for SHAP
        model = models['XGBoost']
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(pd.DataFrame(input_data, columns=selected_features))

        # Display SHAP values
        shap.initjs()
        shap_df = pd.DataFrame(input_data, columns=selected_features)
        st_shap(shap.force_plot(explainer.expected_value, shap_values, shap_df), height=300)

# --- Data Exploration Tab ---

elif selected_tab == "📊 Data Exploration":
    st.title("📊 Data Exploration")
    st.write("Explore the dataset used to train the models.")

    # Display dataset
    st.subheader("House Prices Dataset")
    st.write("Below is a preview of the dataset:")
    st.dataframe(house_data.head(50))

    # Interactive plots
    st.subheader("Feature Correlations")
    st.write("The heatmap below shows the correlations between different features in the dataset.")
    fig_corr, ax_corr = plt.subplots(figsize=(12, 10))
    sns.heatmap(house_data.corr(), cmap='coolwarm', ax=ax_corr)
    ax_corr.set_title('Correlation Heatmap of Features')
    st.pyplot(fig_corr)

    st.subheader("Sale Price Distribution")
    st.write("The histogram below shows the distribution of sale prices in the dataset.")
    fig_price, ax_price = plt.subplots(figsize=(10, 6))
    sns.histplot(house_data['SalePrice'], kde=True, ax=ax_price, color='green')
    ax_price.set_title('Distribution of Sale Prices')
    ax_price.set_xlabel('Sale Price ($)')
    ax_price.set_ylabel('Frequency')
    st.pyplot(fig_price)

# --- Model Performance Tab ---

elif selected_tab == "📈 Model Performance":
    st.title("📈 Model Performance")
    st.write("Review the performance metrics of each model.")

    # Display model evaluation metrics with descriptions
    st.subheader("Model Evaluation Metrics")
    st.write("""
    - **MAE (Mean Absolute Error):** Average of absolute differences between predictions and actual values.
    - **RMSE (Root Mean Squared Error):** Square root of average squared differences between predictions and actual values.
    - **R² Score:** Proportion of variance in the dependent variable predictable from the independent variables.
    """)
    st.table(evaluation_df.style.format({"MAE": "{:,.2f}", "RMSE": "{:,.2f}", "R² Score": "{:.4f}"}))

    # Visual comparison
    st.subheader("Performance Comparison")
    metrics = ['MAE', 'RMSE', 'R² Score']
    for metric in metrics:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='Model', y=metric, data=evaluation_df, palette='viridis', ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        ax.set_title(f'Model Comparison on {metric}')
        ax.set_ylabel(metric)
        st.pyplot(fig)

# --- Inherited Houses Tab ---

elif selected_tab == "📁 Inherited Houses":
    st.title("📁 Inherited Houses Predictions")
    st.write("Review the predicted sale prices for the inherited houses.")

    # Handle any infinite or NaN values
    inherited_predictions = inherited_predictions.replace([np.inf, -np.inf], np.nan)
    inherited_predictions.fillna(inherited_predictions.mean(), inplace=True)

    st.subheader("Predictions")
    st.write("Below are the predicted sale prices for each inherited house using different models.")
    st.dataframe(inherited_predictions.style.format("{:,.2f}"))

    # Average predictions
    st.subheader("Average Predicted Sale Prices")
    avg_prices = inherited_predictions.mean().reset_index()
    avg_prices.columns = ['Model', 'Average Predicted Sale Price']
    st.table(avg_prices.style.format({"Average Predicted Sale Price": "${:,.2f}"}))

    # Visual comparison
    st.subheader("Predicted Prices Comparison")
    fig_pred, ax_pred = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=inherited_predictions, palette='Set2', ax=ax_pred)
    ax_pred.set_ylabel('Predicted Sale Price ($)')
    ax_pred.set_title('Distribution of Predicted Sale Prices by Model')
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
    image_path = os.path.join('images', 'house_image.jpg')
    if os.path.exists(image_path):
        image = Image.open(image_path)
        st.image(image, caption='House Price Prediction', use_column_width=True)

# --- Custom Function to Display SHAP Plots in Streamlit ---

def st_shap(plot, height=None):
    """Display a SHAP plot in Streamlit."""
    import streamlit.components.v1 as components
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

