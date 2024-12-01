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
    models_dir = 'notebook/models'
    models['Linear Regression'] = joblib.load(os.path.join(models_dir, 'linear_regression_model.joblib'))
    models['Ridge Regression'] = joblib.load(os.path.join(models_dir, 'ridge_regression_model.joblib'))
    models['ElasticNet'] = joblib.load(os.path.join(models_dir, 'elasticnet_model.joblib'))
    models['Lasso Regression'] = joblib.load(os.path.join(models_dir, 'lasso_regression_model.joblib'))
    models['Gradient Boosting'] = joblib.load(os.path.join(models_dir, 'gradient_boosting_model.joblib'))
    models['Random Forest'] = joblib.load(os.path.join(models_dir, 'random_forest_model.joblib'))
    models['XGBoost'] = joblib.load(os.path.join(models_dir, 'xgboost_model.joblib'))
    scaler = joblib.load(os.path.join(models_dir, 'scaler.joblib'))
    selected_features = pickle.load(open(os.path.join(models_dir, 'selected_features.pkl'), 'rb'))
    skewed_features = pickle.load(open(os.path.join(models_dir, 'skewed_features.pkl'), 'rb'))
    lam_dict = pickle.load(open(os.path.join(models_dir, 'lam_dict.pkl'), 'rb'))
    feature_importances = pd.read_csv(os.path.join(models_dir, 'feature_importances.csv'))
    return models, scaler, selected_features, skewed_features, lam_dict, feature_importances

data, inherited_houses = load_data()
models, scaler, selected_features, skewed_features, lam_dict, feature_importances = load_models()

# Feature engineering on data
def feature_engineering(df):
    df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
    df['Qual_TotalSF'] = df['OverallQual'] * df['TotalSF']
    return df

data = feature_engineering(data)

# Sidebar navigation
st.sidebar.title("Navigation")
pages = ["Project Summary", "Feature Correlations", "House Price Predictions", "Project Hypotheses", "Model Performance"]
selection = st.sidebar.radio("Go to", pages)

# Project Summary Page
if selection == "Project Summary":
    st.title("Project Summary")
    # [Include the rest of your code for this page]

# Feature Correlations Page
elif selection == "Feature Correlations":
    st.title("Feature Correlations")
    # [Include the rest of your code for this page]

# House Price Predictions Page
elif selection == "House Price Predictions":
    st.title("House Price Predictions")
    st.header("Inherited Houses")
    st.write("Below are the attributes and predicted sale prices for the inherited houses.")

    # Prepare inherited houses data
    def preprocess_inherited_houses(df):
        df_processed = df.copy()
        # Handle missing values
        zero_fill_features = ['2ndFlrSF', 'EnclosedPorch', 'MasVnrArea', 'WoodDeckSF',
                              'BsmtFinSF1', 'TotalBsmtSF', '1stFlrSF', 'BsmtUnfSF']
        for feature in zero_fill_features:
            if feature in df_processed.columns:
                df_processed[feature] = df_processed[feature].fillna(0)

        # List of features to fill with mode
        mode_fill_features = ['BedroomAbvGr', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd']
        for feature in mode_fill_features:
            if feature in df_processed.columns:
                df_processed[feature] = df_processed[feature].fillna(data[feature].mode()[0])

        # Features to fill with median
        median_fill_features = ['GarageYrBlt', 'LotFrontage']
        for feature in median_fill_features:
            if feature in df_processed.columns:
                df_processed[feature] = df_processed[feature].fillna(data[feature].median())

        # Categorical features
        df_processed['BsmtFinType1'] = df_processed.get('BsmtFinType1', 'None').fillna('None')
        df_processed['GarageFinish'] = df_processed.get('GarageFinish', 'Unf').fillna('Unf')
        df_processed['BsmtExposure'] = df_processed.get('BsmtExposure', 'No').fillna('No')
        df_processed['KitchenQual'] = df_processed.get('KitchenQual', 'TA').fillna('TA')

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

        # Ensure the features match
        df_processed = df_processed.reindex(columns=selected_features, fill_value=0)
        return df_processed

    # Process and predict for inherited houses
    inherited_processed = preprocess_inherited_houses(inherited_houses)
    inherited_scaled = scaler.transform(inherited_processed)
    selected_model = models['XGBoost']
    predictions_log = selected_model.predict(inherited_scaled)
    predictions_actual = np.expm1(predictions_log)
    inherited_houses['Predicted SalePrice'] = predictions_actual
    st.write(inherited_houses[['Predicted SalePrice'] + selected_features])

    total_predicted_price = predictions_actual.sum()
    st.success(f"The total predicted sale price for all 4 inherited houses is **${total_predicted_price:,.2f}**.")

    st.header("Real-Time House Price Prediction")
    st.write("Input house attributes to predict the sale price.")

    # Interactive input widgets
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
                        with cols[idx % 2]:
                            input_data[feature] = st.number_input(
                                feature,
                                min_value=min_val,
                                max_value=max_val,
                                value=mean_val,
                                help=f"Enter the value for {feature}"
                            )
                    else:
                        unique_values = sorted(data[feature].dropna().unique())
                        with cols[idx % 2]:
                            input_data[feature] = st.selectbox(
                                feature,
                                unique_values,
                                help=f"Select the value for {feature}"
                            )
                else:
                    input_data[feature] = 0.0  # Default value for engineered features

            submit_button = st.form_submit_button(label='Predict Sale Price')

        if submit_button:
            return pd.DataFrame(input_data, index=[0])
        else:
            return None

    user_input = user_input_features()
    if user_input is not None:
        user_processed = preprocess_inherited_houses(user_input)
        user_scaled = scaler.transform(user_processed)
        user_pred_log = selected_model.predict(user_scaled)
        user_pred_actual = np.expm1(user_pred_log)
        st.success(f"The predicted sale price is **${user_pred_actual[0]:,.2f}**.")

# Project Hypotheses Page
elif selection == "Project Hypotheses":
    st.title("Project Hypotheses")
    st.write("""
    **Hypothesis 1:** Higher overall quality of the house leads to a higher sale price.
    
    **Validation:** The `OverallQual` feature shows a strong positive correlation with the sale price, confirming this hypothesis.

    **Hypothesis 2:** Larger living areas result in higher sale prices.

    **Validation:** Features like `GrLivArea` and `TotalSF` have high correlations with the sale price, supporting this hypothesis.

    **Hypothesis 3:** Recent renovations positively impact the sale price.

    **Validation:** The `YearRemodAdd` feature correlates with the sale price, indicating that more recent remodels can increase the house value.
    """)
    st.header("Hypotheses Validation")

    # Plotting OverallQual vs SalePrice
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='OverallQual', y='SalePrice', data=data, palette='Set2')
    plt.title('SalePrice vs OverallQual', fontsize=16)
    plt.xlabel('Overall Quality', fontsize=12)
    plt.ylabel('Sale Price', fontsize=12)
    st.pyplot(plt)

    st.write("""
    The boxplot demonstrates that as the overall quality increases, the median sale price also increases, validating our first hypothesis.
    """)

    # Plotting TotalSF vs SalePrice
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='TotalSF', y='SalePrice', data=data, hue='OverallQual', palette='coolwarm')
    plt.title('SalePrice vs TotalSF', fontsize=16)
    plt.xlabel('Total Square Footage', fontsize=12)
    plt.ylabel('Sale Price', fontsize=12)
    st.pyplot(plt)

    st.write("""
    The scatter plot shows a positive relationship between total square footage and sale price, supporting our second hypothesis. Higher quality homes (indicated by color) tend to have higher prices even at similar square footages.
    """)

    # Plotting YearRemodAdd vs SalePrice
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='YearRemodAdd', y='SalePrice', data=data, ci=None, color='green')
    plt.title('SalePrice vs YearRemodeled', fontsize=16)
    plt.xlabel('Year Remodeled', fontsize=12)
    plt.ylabel('Average Sale Price', fontsize=12)
    st.pyplot(plt)

    st.write("""
    The line plot indicates that homes remodeled more recently tend to have higher sale prices, which confirms our third hypothesis.
    """)

# Model Performance Page
elif selection == "Model Performance":
    st.title("Model Performance")
    st.header("Performance Metrics")
    results_df = pd.read_csv('models/model_evaluation.csv')
    st.dataframe(results_df.style.format({'MAE': '{:,.2f}', 'RMSE': '{:,.2f}', 'R² Score': '{:.4f}'}))

    st.write("""
    The table above presents the performance metrics of various regression models. The XGBoost model outperforms others with the lowest MAE and RMSE, and the highest R² Score.
    """)

    st.header("Pipeline Steps")
    st.write("""
    1. **Data Collection and Understanding:** Gathered and explored the dataset to understand the variables and their distributions.

    2. **Data Cleaning:** Handled missing values by imputing or filling with appropriate statistics.

    3. **Feature Engineering:** Created new features like `TotalSF` and `Qual_TotalSF` to capture more information.

    4. **Feature Selection:** Employed Random Forest to select the most impactful features.

    5. **Data Scaling:** Standardized features to improve model performance.

    6. **Model Training:** Trained multiple regression models and selected the best based on performance metrics.

    7. **Model Evaluation:** Assessed model performance using MAE, RMSE, and R² Score.

    8. **Deployment:** Developed an interactive dashboard for predictions and insights.
    """)

    st.header("Feature Importances")
    # Display feature importances from the Random Forest model
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importances)
    plt.title('Feature Importances from Random Forest', fontsize=16)
    plt.xlabel('Importance', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    st.pyplot(plt)

    st.write("""
    The bar chart illustrates the relative importance of each feature in predicting the sale price. Features like `GrLivArea`, `TotalBsmtSF`, and `OverallQual` are among the most significant.
    """)

    st.header("Actual vs Predicted Prices")
    selected_model = models['XGBoost']
    X_train, X_test, y_train, y_test = joblib.load('models/train_test_data.joblib')
    y_pred = selected_model.predict(X_test)
    y_pred_actual = np.expm1(y_pred)
    y_test_actual = np.expm1(y_test)

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_test_actual, y=y_pred_actual, color='purple')
    plt.xlabel('Actual Sale Price', fontsize=12)
    plt.ylabel('Predicted Sale Price', fontsize=12)
    plt.title('Actual vs Predicted Sale Prices', fontsize=16)
    plt.plot([y_test_actual.min(), y_test_actual.max()], [y_test_actual.min(), y_test_actual.max()], 'r--')
    st.pyplot(plt)

    st.write("""
    The scatter plot compares the actual sale prices with the predicted sale prices. The red dashed line represents a perfect prediction. Most points are close to this line, indicating good model performance.
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
    The residuals are centered around zero and approximately normally distributed, suggesting that the model's errors are random and not biased.
    """)

else:
    st.write("Please select a page from the sidebar.")
