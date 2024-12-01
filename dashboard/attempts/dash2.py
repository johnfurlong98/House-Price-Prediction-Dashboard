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

# --- Function to Display SHAP Plots in Streamlit ---

def st_shap(plot, height=None):
    """Display a SHAP plot in Streamlit."""
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

# --- Set Page Configuration ---

st.set_page_config(
    page_title="🏠 Ames House Price Prediction Dashboard",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Absolute Path Configuration ---

# Define absolute paths to models and data directories
MODELS_DIR = 'data/models'
DATA_DIR = 'data'

# --- Load Models and Data ---

@st.cache(allow_output_mutation=True)
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
            models[name] = joblib.load(model_path)
        else:
            st.error(f"Model file '{file}' not found in '{models_dir}'.")

    # Load preprocessing objects
    scaler = joblib.load(os.path.join(models_dir, 'scaler.joblib'))
    with open(os.path.join(models_dir, 'selected_features.pkl'), 'rb') as f:
        selected_features = pickle.load(f)
    with open(os.path.join(models_dir, 'lam_dict.pkl'), 'rb') as f:
        lam_dict = pickle.load(f)
    with open(os.path.join(models_dir, 'skewed_features.pkl'), 'rb') as f:
        skewed_features = pickle.load(f)

    # Load datasets
    house_data = pd.read_csv(os.path.join(DATA_DIR, 'house_prices_records.csv'))
    evaluation_df = pd.read_csv(os.path.join(models_dir, 'model_evaluation.csv'))
    inherited_predictions = pd.read_csv(os.path.join(models_dir, 'inherited_houses_predictions.csv'))

    return models, scaler, selected_features, lam_dict, skewed_features, house_data, evaluation_df, inherited_predictions

# Load everything
(models, scaler, selected_features, lam_dict, skewed_features,
 house_data, evaluation_df, inherited_predictions) = load_models_and_data(MODELS_DIR, DATA_DIR)

# --- Feature Display Names with Units ---

feature_display = {
    'OverallQual': 'Overall Quality (1-10)',
    'GrLivArea': 'Above Grade Living Area (sqft)',
    'TotalBsmtSF': 'Total Basement Area (sqft)',
    'GarageArea': 'Garage Area (sqft)',
    'YearBuilt': 'Year Built',
    '1stFlrSF': 'First Floor Area (sqft)',
    'LotArea': 'Lot Area (sqft)',
    '2ndFlrSF': 'Second Floor Area (sqft)',
    'BedroomAbvGr': 'Bedrooms Above Ground',
    'BsmtExposure': 'Basement Exposure',
    'BsmtFinType1': 'Basement Finish Type',
    'BsmtFinSF1': 'Basement Finished Area (sqft)',
    'BsmtUnfSF': 'Basement Unfinished Area (sqft)',
    'KitchenQual': 'Kitchen Quality',
    'LotFrontage': 'Lot Frontage (ft)',
    'MasVnrArea': 'Masonry Veneer Area (sqft)',
    'EnclosedPorch': 'Enclosed Porch (sqft)',
    'OpenPorchSF': 'Open Porch Area (sqft)',
    'OverallCond': 'Overall Condition (1-10)',
    'WoodDeckSF': 'Wood Deck Area (sqft)',
    'GarageYrBlt': 'Garage Year Built',
    'YearRemodAdd': 'Year Remodeled'
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
        df[feature].fillna(0, inplace=True)

    # Fill other missing values
    fill_values = {
        'BedroomAbvGr': 3,
        'BsmtFinType1': 'None',
        'GarageFinish': 'Unf',
        'BsmtExposure': 'No',
        'KitchenQual': 'TA',
        'GarageYrBlt': datetime.now().year,
        'LotFrontage': 60,
        'OverallQual': 5,
        'OverallCond': 5,
        'YearBuilt': 1990,
        'YearRemodAdd': 2000
    }

    for feature, value in fill_values.items():
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
            df[col].fillna(0, inplace=True)

    # Transform skewed features
    for feat in skewed_features:
        if feat in df.columns:
            data = df[feat]
            if (data <= 0).any():
                df[feat] = np.log1p(data)
            else:
                lam = lam_dict.get(feat)
                if lam is not None:
                    df[feat] = boxcox(data, lmbda=lam)
                else:
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

    categorical_features = ['BsmtExposure', 'BsmtFinType1', 'KitchenQual']

    if feature in categorical_features:
        # Categorical selectbox with full descriptions
        options = {
            'BsmtExposure': ['No', 'Mn', 'Av', 'Gd'],
            'BsmtFinType1': ['None', 'Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ'],
            'KitchenQual': ['Po', 'Fa', 'TA', 'Gd', 'Ex']
        }
        selection = st.selectbox(label, options=options[feature], key=unique_key)
        return selection

    elif feature in ['YearBuilt', 'YearRemodAdd', 'GarageYrBlt']:
        # Year input
        current_year = datetime.now().year
        value = st.number_input(label, min_value=1872, max_value=current_year, value=1990, key=unique_key)
        return int(value)

    else:
        # Numeric input
        value = st.number_input(label, min_value=0, value=0, key=unique_key)
        return value

# --- Sidebar Navigation ---

st.sidebar.title("Navigation")
selected_tab = st.sidebar.radio("Go to", ["Home", "Data Exploration", "Model Performance", "Inherited Houses", "About"])

# --- Apply Custom CSS for Modern Look ---

st.markdown(
    """
    <style>
    /* Custom CSS styles */
    .stButton>button {
        color: white;
        background-color: #007bff;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 16px;
    }
    .stButton>button:hover {
        background-color: #0056b3;
    }
    .stProgress > div > div > div > div {
        background-color: #007bff;
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
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Home Tab ---

if selected_tab == "Home":
    st.title("🏠 Ames House Price Prediction")
    st.write("""
    Welcome to the **Ames House Price Prediction Dashboard**. Use this tool to input various features of a house
    and get an estimated sale price using advanced machine learning models.
    """)

    # Collect user inputs
    st.header("Enter House Features")
    input_features = {}

    # Required features
    st.subheader("Required Features")
    required_features = [
        'OverallQual', 'GrLivArea', 'TotalBsmtSF', 'GarageArea',
        'YearBuilt', '1stFlrSF', 'LotArea'
    ]

    for feature in required_features:
        label = feature_display.get(feature, feature)
        input_value = get_input(feature, label, is_required=True)
        input_features[feature] = input_value

    # Optional features
    st.subheader("Optional Features")
    optional_features = [feat for feat in selected_features if feat not in required_features]

    for feature in optional_features:
        label = feature_display.get(feature, feature)
        input_value = get_input(feature, label, is_required=False)
        input_features[feature] = input_value

    # Predict button
    if st.button("🔮 Predict"):
        # Preprocess input
        input_data = preprocess_input(input_features)

        # Make predictions
        predictions = {}
        for name, model in models.items():
            pred_log = model.predict(input_data)
            pred_price = np.expm1(pred_log)[0]
            predictions[name] = pred_price

        # Display predictions
        st.subheader("💰 Predicted Sale Prices")
        pred_df = pd.DataFrame(list(predictions.items()), columns=['Model', 'Predicted Sale Price'])
        pred_df['Predicted Sale Price'] = pred_df['Predicted Sale Price'].apply(lambda x: f"${x:,.2f}")
        st.table(pred_df)

        # Visualize predictions
        st.subheader("📊 Prediction Comparison")
        fig_pred, ax_pred = plt.subplots(figsize=(10, 6))
        sns.barplot(x='Predicted Sale Price', y='Model', data=pred_df, palette='Blues_d', ax=ax_pred)
        ax_pred.set_xlabel('Predicted Sale Price ($)')
        ax_pred.set_ylabel('Model')
        ax_pred.set_title('Comparison of Predicted Sale Prices by Model')
        st.pyplot(fig_pred)

        # SHAP explanation (using XGBoost model)
        st.subheader("🔍 Model Explanation")
        st.write("Below is the SHAP analysis showing the impact of each feature on the prediction for the **XGBoost** model.")

        # Ensure the XGBoost model is present
        if 'XGBoost' in models:
            model = models['XGBoost']
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(pd.DataFrame(input_data, columns=selected_features))

            # Display SHAP force plot
            st_shap(shap.force_plot(explainer.expected_value, shap_values, pd.DataFrame(input_data, columns=selected_features)), height=300)
        else:
            st.error("XGBoost model not found.")

# --- Data Exploration Tab ---

elif selected_tab == "Data Exploration":
    st.title("📊 Data Exploration")
    st.write("Explore the dataset used to train the models.")

    # Display dataset
    st.subheader("📂 House Prices Dataset")
    st.write("Below is a preview of the dataset:")
    st.dataframe(house_data.head(50))

    # Interactive plots
    st.subheader("🔗 Feature Correlations")
    st.write("The heatmap below shows the correlations between different features in the dataset.")

    # Select numerical columns for correlation
    numerical_data = house_data.select_dtypes(include=[np.number])

    fig_corr, ax_corr = plt.subplots(figsize=(12, 10))
    correlation_matrix = numerical_data.corr()
    sns.heatmap(correlation_matrix, cmap='coolwarm', ax=ax_corr, annot=True, fmt=".2f", linewidths=0.5)
    ax_corr.set_title('Correlation Heatmap of Features')
    st.pyplot(fig_corr)

    st.subheader("📈 Sale Price Distribution")
    st.write("The histogram below shows the distribution of sale prices in the dataset.")
    fig_price, ax_price = plt.subplots(figsize=(10, 6))
    sns.histplot(house_data['SalePrice'], kde=True, ax=ax_price, color='green')
    ax_price.set_title('Distribution of Sale Prices')
    ax_price.set_xlabel('Sale Price ($)')
    ax_price.set_ylabel('Frequency')
    st.pyplot(fig_price)

# --- Model Performance Tab ---

elif selected_tab == "Model Performance":
    st.title("📈 Model Performance")
    st.write("Review the performance metrics of each model.")

    # Display model evaluation metrics with descriptions
    st.subheader("📊 Model Evaluation Metrics")
    st.write("""
    - **MAE (Mean Absolute Error):** Average of absolute differences between predictions and actual values.
    - **RMSE (Root Mean Squared Error):** Square root of average squared differences between predictions and actual values.
    - **R² Score:** Proportion of variance in the dependent variable predictable from the independent variables.
    """)

    st.table(evaluation_df.style.format({"MAE": "{:,.2f}", "RMSE": "{:,.2f}", "R² Score": "{:.4f}"}))

    # Model Overviews
    st.subheader("📝 Model Overviews")
    for _, row in evaluation_df.iterrows():
        model_name = row['Model']
        st.markdown(f"### {model_name}")

        # Detailed explanation of hyperparameters
        if model_name == 'Linear Regression':
            st.write("**Linear Regression** is a simple model that assumes a linear relationship between the input features and the target variable.")
        elif model_name == 'Ridge Regression':
            st.write("""
            **Ridge Regression** adds L2 regularization to Linear Regression to prevent overfitting.
            - **Alpha:** Controls the strength of the regularization. Selected through cross-validation.
            """)
        elif model_name == 'Lasso Regression':
            st.write("""
            **Lasso Regression** adds L1 regularization, which can shrink coefficients to zero, effectively performing feature selection.
            - **Alpha:** Controls the strength of the regularization. Selected through cross-validation.
            """)
        elif model_name == 'ElasticNet':
            st.write("""
            **ElasticNet** combines L1 and L2 regularization.
            - **Alpha:** Overall strength of regularization.
            - **L1_ratio:** Balance between L1 and L2 regularization.
            """)
        elif model_name == 'Gradient Boosting':
            st.write("""
            **Gradient Boosting Regressor** builds an ensemble of weak prediction models, typically decision trees, in a stage-wise fashion.
            - **n_estimators:** Number of boosting stages.
            - **Learning_rate:** Shrinks the contribution of each tree.
            - **Max_depth:** Maximum depth of the individual regression estimators.
            """)
        elif model_name == 'Random Forest':
            st.write("""
            **Random Forest Regressor** is an ensemble method that fits multiple decision trees on various sub-samples of the dataset.
            - **n_estimators:** Number of trees in the forest.
            - **Max_features:** Number of features to consider when looking for the best split.
            """)
        elif model_name == 'XGBoost':
            st.write("""
            **XGBoost Regressor** is an optimized implementation of gradient boosting.
            - **n_estimators:** Number of gradient boosted trees.
            - **Learning_rate:** Step size shrinkage used to prevent overfitting.
            - **Max_depth:** Maximum depth of a tree.
            """)

        # Model performance metrics
        st.write(f"**MAE:** {row['MAE']:,.2f}")
        st.write(f"**RMSE:** {row['RMSE']:,.2f}")
        st.write(f"**R² Score:** {row['R² Score']:.4f}")

    # Visual comparison
    st.subheader("📉 Performance Comparison")
    metrics = ['MAE', 'RMSE', 'R² Score']
    for metric in metrics:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='Model', y=metric, data=evaluation_df, palette='viridis', ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_title(f'Model Comparison on {metric}')
        ax.set_ylabel(metric)
        ax.set_xlabel('Model')
        st.pyplot(fig)

# --- Inherited Houses Tab ---

elif selected_tab == "Inherited Houses":
    st.title("📁 Inherited Houses Predictions")
    st.write("Review the predicted sale prices for the inherited houses.")

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
    ax_pred.set_title('Distribution of Predicted Sale Prices by Model')
    st.pyplot(fig_pred)

# --- About Tab ---

elif selected_tab == "About":
    st.title("ℹ️ About")
    st.write("""
    ### Ames House Price Prediction Dashboard
    Developed to provide accurate house price predictions using multiple advanced machine learning models.

    **Features:**
    - **User-Friendly Interface:** Modern design with intuitive navigation.
    - **Multiple Models:** Compare predictions from various models.
    - **Data Exploration:** Interactive visuals to understand the dataset.
    - **Model Performance:** Detailed metrics and comparison charts.
    - **Inherited Houses:** Specific predictions for inherited properties.

    **Developed By:**
    - *Your Name*
    - *Data Scientist*

    **Technologies Used:**
    - **Python:** Data processing and model development.
    - **Streamlit:** Web app development.
    - **Machine Learning Libraries:** scikit-learn, XGBoost, SHAP.

    **Acknowledgments:**
    - Dataset provided by [Kaggle - Ames Housing Dataset](https://www.kaggle.com/c/house-prices-advanced-regression-techniques).
    """)

    # Display an image
    st.image("https://images.unsplash.com/photo-1560184897-6ec56aed1c8d", caption='House Price Prediction', use_column_width=True)