# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import boxcox
import pickle

# Set up page configuration
st.set_page_config(page_title="House Price Prediction Dashboard", layout="wide")

# Load models and data
@st.cache_data
def load_data():
    data = pd.read_csv('data/house_prices_records.csv')
    inherited_houses = pd.read_csv('data/inherited_houses.csv')
    return data, inherited_houses

@st.cache_resource
def load_models():
    models = {}
    models_dir = '/Users/conor/Desktop/JohnProject/notebook/models'
    # Load all models
    model_files = {
        'Linear Regression': 'linear_regression_model.joblib',
        'Ridge Regression': 'ridge_regression_model.joblib',
        'ElasticNet': 'elasticnet_model.joblib',
        'Lasso Regression': 'lasso_regression_model.joblib',
        'Gradient Boosting': 'gradient_boosting_model.joblib',
        'Random Forest': 'random_forest_model.joblib',
        'XGBoost': 'xgboost_model.joblib'
    }
    for name, filename in model_files.items():
        models[name] = joblib.load(os.path.join(models_dir, filename))

    scaler = joblib.load(os.path.join(models_dir, 'scaler.joblib'))
    selected_features = pickle.load(open(os.path.join(models_dir, 'selected_features.pkl'), 'rb'))
    skewed_features = pickle.load(open(os.path.join(models_dir, 'skewed_features.pkl'), 'rb'))
    lam_dict = pickle.load(open(os.path.join(models_dir, 'lam_dict.pkl'), 'rb'))
    feature_importances = pd.read_csv(os.path.join(models_dir, 'feature_importances.csv'))
    return models, scaler, selected_features, skewed_features, lam_dict, feature_importances

# Load data
data, inherited_houses = load_data()

# Load models and related data
models, scaler, selected_features, skewed_features, lam_dict, feature_importances = load_models()

# Define feature engineering function
def feature_engineering(df):
    df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
    df['Qual_TotalSF'] = df['OverallQual'] * df['TotalSF']
    return df

# Define preprocessing function
def preprocess_data(df):
    df_processed = df.copy()
    # Handle missing values
    zero_fill_features = ['2ndFlrSF', 'EnclosedPorch', 'MasVnrArea', 'WoodDeckSF',
                          'BsmtFinSF1', 'TotalBsmtSF', '1stFlrSF', 'BsmtUnfSF']
    for feature in zero_fill_features:
        if feature in df_processed.columns:
            df_processed[feature] = df_processed[feature].fillna(0)

    # Fill categorical features
    categorical_mode_fill = {
        'BsmtFinType1': 'None',
        'GarageFinish': 'Unf',
        'BsmtExposure': 'No',
        'KitchenQual': 'TA'
    }
    for feature, value in categorical_mode_fill.items():
        if feature in df_processed.columns:
            df_processed[feature] = df_processed[feature].fillna(value)

    # Fill numerical features
    numerical_median_fill = ['BedroomAbvGr', 'GarageYrBlt', 'LotFrontage', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd']
    for feature in numerical_median_fill:
        if feature in df_processed.columns:
            df_processed[feature] = df_processed[feature].fillna(data[feature].median())

    # Encode categorical features
    ordinal_mappings = {
        'BsmtFinType1': {'None': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6},
        'KitchenQual': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
        'BsmtExposure': {'None': 0, 'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4},
        'GarageFinish': {'None': 0, 'Unf': 1, 'RFn': 2, 'Fin': 3}
    }
    for col, mapping in ordinal_mappings.items():
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].map(mapping)

    # Feature engineering
    df_processed = feature_engineering(df_processed)

    # Transform skewed features
    for feat in skewed_features:
        if feat in df_processed.columns:
            if (df_processed[feat] <= 0).any():
                df_processed[feat] = np.log1p(df_processed[feat])
            else:
                lam = lam_dict.get(feat)
                if lam is not None:
                    try:
                        df_processed[feat] = boxcox(df_processed[feat], lam)
                    except ValueError:
                        df_processed[feat] = np.log1p(df_processed[feat])
                else:
                    df_processed[feat] = np.log1p(df_processed[feat])

    return df_processed

# Preprocess the data
data = preprocess_data(data)

# Metadata for features (from the provided metadata)
feature_metadata = {
    '1stFlrSF': 'First Floor square feet (334 - 4692)',
    '2ndFlrSF': 'Second floor square feet (0 - 2065)',
    'BedroomAbvGr': 'Bedrooms above grade (0 - 8)',
    'BsmtExposure': 'Walkout or garden level walls (Gd, Av, Mn, No, None)',
    'BsmtFinType1': 'Rating of basement finished area (GLQ to None)',
    'BsmtFinSF1': 'Type 1 finished square feet (0 - 5644)',
    'BsmtUnfSF': 'Unfinished basement area (0 - 2336)',
    'TotalBsmtSF': 'Total basement area (0 - 6110)',
    'GarageArea': 'Garage size in square feet (0 - 1418)',
    'GarageFinish': 'Garage interior finish (Fin, RFn, Unf, None)',
    'GarageYrBlt': 'Year garage was built (1900 - 2010)',
    'GrLivArea': 'Above grade living area (334 - 5642)',
    'KitchenQual': 'Kitchen quality (Ex, Gd, TA, Fa, Po)',
    'LotArea': 'Lot size in square feet (1300 - 215245)',
    'LotFrontage': 'Linear feet of street connected to property (21 - 313)',
    'MasVnrArea': 'Masonry veneer area (0 - 1600)',
    'EnclosedPorch': 'Enclosed porch area (0 - 286)',
    'OpenPorchSF': 'Open porch area (0 - 547)',
    'OverallCond': 'Overall condition rating (1 - 10)',
    'OverallQual': 'Overall material and finish rating (1 - 10)',
    'WoodDeckSF': 'Wood deck area (0 - 736)',
    'YearBuilt': 'Original construction date (1872 - 2010)',
    'YearRemodAdd': 'Remodel date (1950 - 2010)',
    'TotalSF': 'Total square feet of house (including basement)',
    'Qual_TotalSF': 'Product of OverallQual and TotalSF'
}

# Sidebar navigation
st.sidebar.title("Navigation")
pages = ["Project Summary", "Feature Correlations", "House Price Predictions", "Project Hypotheses", "Model Performance"]
selection = st.sidebar.radio("Go to", pages)

# Project Summary Page
if selection == "Project Summary":
    st.title("House Price Prediction Dashboard")
    st.write("""
    ## Project Summary

    Welcome to the House Price Prediction Dashboard. This project aims to build a predictive model to estimate the sale prices of houses based on various features. By analyzing the data and developing robust models, we provide insights into the key factors that influence house prices.

    **Key Objectives:**

    - **Data Analysis and Preprocessing:** Understand and prepare the data for modeling.
    - **Feature Engineering:** Create new features to improve model performance.
    - **Model Development:** Train and evaluate multiple regression models.
    - **Deployment:** Develop an interactive dashboard for predictions and insights.

    **Instructions:**

    - Use the sidebar to navigate between different sections.
    - Explore data correlations, make predictions, and understand the model performance.
    """)
    # Optional: Include an image or logo if available
    # st.image('path_to_image.jpg', use_column_width=True)

# Feature Correlations Page
elif selection == "Feature Correlations":
    st.title("Feature Correlations")
    st.write("""
    Understanding the relationships between different features and the sale price is crucial for building an effective predictive model.
    """)

    # Compute correlation matrix
    corr_matrix = data.corr()
    top_corr_features = corr_matrix.index[abs(corr_matrix['SalePrice']) > 0.5]

    # Plot correlation heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(data[top_corr_features].corr(), annot=True, cmap='RdBu')
    plt.title('Correlation Heatmap of Top Features', fontsize=16)
    st.pyplot(plt)

    st.write("""
    The heatmap above shows the correlations among the top features and the sale price. Features like `OverallQual`, `GrLivArea`, and `TotalSF` have strong positive correlations with `SalePrice`.
    """)

    # Additional visualization: Pairplot with top features
    st.write("### Pairplot of Top Correlated Features")
    # Select top 5 features excluding 'SalePrice'
    top_features = top_corr_features.drop('SalePrice').tolist()[:5]
    sns.set(style="ticks")
    pairplot_fig = sns.pairplot(data[top_features + ['SalePrice']], diag_kind='kde', height=2.5)
    st.pyplot(pairplot_fig)

    st.write("""
    The pairplot displays pairwise relationships between the top correlated features and the sale price. It helps visualize potential linear relationships and distributions.
    """)

# House Price Predictions Page
elif selection == "House Price Predictions":
    st.title("House Price Predictions")

    # Inherited Houses Predictions
    st.header("Inherited Houses")
    st.write("Below are the predicted sale prices for the inherited houses based on the best-performing model.")

    # Preprocess and predict for inherited houses
    inherited_processed = preprocess_data(inherited_houses)
    inherited_scaled = scaler.transform(inherited_processed[selected_features])
    best_model_name = 'XGBoost'  # Assuming XGBoost is the best model
    selected_model = models[best_model_name]
    predictions_log = selected_model.predict(inherited_scaled)
    predictions_actual = np.expm1(predictions_log)
    predictions_actual[predictions_actual < 0] = 0  # Handle negative predictions

    inherited_houses['Predicted SalePrice'] = predictions_actual
    st.dataframe(inherited_houses[['Predicted SalePrice'] + selected_features])

    total_predicted_price = predictions_actual.sum()
    st.success(f"The total predicted sale price for all inherited houses is **${total_predicted_price:,.2f}**.")

    # Real-Time Prediction
    st.header("Real-Time House Price Prediction")
    st.write("Input house attributes to predict the sale price using the best-performing model.")

    def user_input_features():
        input_data = {}
        with st.form(key='house_features'):
            cols = st.columns(2)
            for idx, feature in enumerate(selected_features):
                if feature in data.columns:
                    if data[feature].dtype in [np.float64, np.int64]:
                        min_val = float(data[feature].min())
                        max_val = float(data[feature].max())
                        mean_val = float(data[feature].mean())
                        help_text = feature_metadata.get(feature, '')
                        with cols[idx % 2]:
                            input_data[feature] = st.number_input(
                                feature,
                                min_value=min_val,
                                max_value=max_val,
                                value=mean_val,
                                help=help_text
                            )
                    else:
                        unique_values = sorted(data[feature].dropna().unique())
                        help_text = feature_metadata.get(feature, '')
                        with cols[idx % 2]:
                            input_data[feature] = st.selectbox(
                                feature,
                                unique_values,
                                help=help_text
                            )
                else:
                    # For engineered features, they will be calculated later
                    pass  # Do not collect input for engineered features

            submit_button = st.form_submit_button(label='Predict Sale Price')

        if submit_button:
            input_df = pd.DataFrame(input_data, index=[0])
            # Calculate engineered features
            input_df = feature_engineering(input_df)
            return input_df
        else:
            return None

    user_input = user_input_features()
    if user_input is not None:
        user_processed = preprocess_data(user_input)
        user_scaled = scaler.transform(user_processed[selected_features])
        user_pred_log = selected_model.predict(user_scaled)
        user_pred_actual = np.expm1(user_pred_log)
        user_pred_actual[user_pred_actual < 0] = 0  # Handle negative predictions
        st.success(f"The predicted sale price is **${user_pred_actual[0]:,.2f}**.")

# Project Hypotheses Page
elif selection == "Project Hypotheses":
    st.title("Project Hypotheses")
    st.write("""
    ## Hypothesis Validation

    **Hypothesis 1:** Higher overall quality of the house leads to a higher sale price.

    - **Validation:** The `OverallQual` feature shows a strong positive correlation with the sale price, confirming this hypothesis.

    **Hypothesis 2:** Larger living areas result in higher sale prices.

    - **Validation:** Features like `GrLivArea` and `TotalSF` have high correlations with the sale price, supporting this hypothesis.

    **Hypothesis 3:** Recent renovations positively impact the sale price.

    - **Validation:** The `YearRemodAdd` feature correlates with the sale price, indicating that more recent remodels can increase the house value.
    """)

    # Visualization for Hypotheses
    st.write("### Visualization of Hypotheses")

    # OverallQual vs SalePrice
    st.write("#### SalePrice vs OverallQual")
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='OverallQual', y='SalePrice', data=data, palette='Set2')
    plt.title('SalePrice vs OverallQual', fontsize=16)
    plt.xlabel('Overall Quality', fontsize=12)
    plt.ylabel('Sale Price', fontsize=12)
    st.pyplot(plt)

    # TotalSF vs SalePrice
    st.write("#### SalePrice vs TotalSF")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='TotalSF', y='SalePrice', data=data, hue='OverallQual', palette='coolwarm')
    plt.title('SalePrice vs TotalSF', fontsize=16)
    plt.xlabel('Total Square Footage', fontsize=12)
    plt.ylabel('Sale Price', fontsize=12)
    st.pyplot(plt)

    # YearRemodAdd vs SalePrice
    st.write("#### SalePrice vs YearRemodAdd")
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='YearRemodAdd', y='SalePrice', data=data, ci=None, color='green')
    plt.title('SalePrice vs Year Remodeled', fontsize=16)
    plt.xlabel('Year Remodeled', fontsize=12)
    plt.ylabel('Average Sale Price', fontsize=12)
    st.pyplot(plt)

# Model Performance Page
elif selection == "Model Performance":
    st.title("Model Performance")
    st.header("Performance Metrics")
    results_df = pd.read_csv('/Users/conor/Desktop/JohnProject/notebook/models/model_evaluation.csv')
    st.dataframe(results_df.style.format({'MAE': '{:,.2f}', 'RMSE': '{:,.2f}', 'R² Score': '{:.4f}'}))

    # Determine best model
    best_model_name = results_df.sort_values('RMSE').iloc[0]['Model']
    st.write(f"Best performing model is **{best_model_name}** based on RMSE.")

    st.write("""
    The table above presents the performance metrics of various regression models. The best-performing model outperforms others with the lowest MAE and RMSE, and the highest R² Score.
    """)

    st.header("Pipeline Steps")
    st.write("""
    ### 1. Data Collection and Understanding
    - **Datasets Used:** Historical house sale data and inherited houses data.
    - **Exploration:** Initial exploration of data shapes, types, and sample records.

    ### 2. Data Cleaning
    - **Missing Values Handling:**
      - Filled missing numerical features with zeros or medians.
      - Filled missing categorical features with modes or default values.
    - **Verification:** Ensured no missing values remained.

    ### 3. Feature Engineering
    - **Encoding Categorical Features:** Applied ordinal mappings to convert categories to numerical values.
    - **Creating New Features:**
      - `TotalSF`: Sum of `TotalBsmtSF`, `1stFlrSF`, and `2ndFlrSF`.
      - `Qual_TotalSF`: Product of `OverallQual` and `TotalSF`.

    ### 4. Feature Transformation
    - **Handling Skewness:** Applied log or box-cox transformations to skewed features to normalize distributions.

    ### 5. Feature Selection
    - **Random Forest Importance:** Selected top features based on importance scores from a Random Forest model.

    ### 6. Data Scaling
    - **Standardization:** Used `StandardScaler` to standardize features for better model performance.

    ### 7. Model Training
    - **Models Trained:** Linear Regression, Ridge Regression, Lasso Regression, ElasticNet, Random Forest, Gradient Boosting, and XGBoost.
    - **Hyperparameter Tuning:** Adjusted parameters to prevent overfitting and improve accuracy.

    ### 8. Model Evaluation
    - **Metrics Used:** MAE, RMSE, and R² Score.
    - **Best Model Selection:** Chose XGBoost as the best model based on evaluation metrics.

    ### 9. Deployment
    - **Interactive Dashboard:** Developed using Streamlit to allow users to interact with the model and visualize results.

    """)

    st.header("Feature Importances")
    # Display feature importances from the Random Forest model
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importances.sort_values(by='Importance', ascending=False))
    plt.title('Feature Importances from Random Forest', fontsize=16)
    plt.xlabel('Importance', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    st.pyplot(plt)

    st.write("""
    The bar chart illustrates the relative importance of each feature in predicting the sale price. Features like `GrLivArea`, `TotalSF`, and `OverallQual` are among the most significant.
    """)

    st.header("Actual vs Predicted Prices")
    selected_model = models[best_model_name]
    X_train, X_test, y_train, y_test = joblib.load('/Users/conor/Desktop/JohnProject/notebook/models/train_test_data.joblib')
    y_pred_log = selected_model.predict(X_test)
    y_pred_actual = np.expm1(y_pred_log)
    y_pred_actual[y_pred_actual < 0] = 0  # Handle negative predictions
    y_test_actual = np.expm1(y_test)

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_test_actual, y=y_pred_actual, color='purple')
    plt.xlabel('Actual Sale Price', fontsize=12)
    plt.ylabel('Predicted Sale Price', fontsize=12)
    plt.title('Actual vs Predicted Sale Prices', fontsize=16)
    plt.plot([y_test_actual.min(), y_test_actual.max()], [y_test_actual.min(), y_test_actual.max()], 'r--')
    st.pyplot(plt)

    st.write("""
    The scatter plot compares the actual sale prices with the predicted sale prices. The red dashed line represents perfect predictions. Most points are close to this line, indicating good model performance.
    """)

    st.header("Residual Analysis")
    residuals = y_test_actual - y_pred_actual
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True, color='coral')
    plt.title('Residuals Distribution', fontsize=16)
    plt.xlabel('Residuals', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    st.pyplot(plt)

    st.write("""
    The residuals are centered around zero and approximately normally distributed, suggesting that the model's errors are random and unbiased.
    """)

else:
    st.write("Please select a page from the sidebar.")
