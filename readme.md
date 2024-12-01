# House Price Prediction Dashboard

## Project Overview

This project implements a **House Price Prediction Dashboard** to predict house sale prices in Ames, Iowa, based on historical housing data. The solution is tailored for **Lydia Doe**, who inherited properties in Ames and seeks accurate sale price predictions to maximize her profits. The dashboard offers insights into key factors affecting house prices and enables real-time predictions for any house in Ames.

---

## Dataset

The dataset is sourced from [Kaggle](https://www.kaggle.com/codeinstitute/housing-prices-data) and includes approximately **1,500 rows** of data on houses built between **1872 and 2010** in Ames, Iowa. Features encompass house attributes such as size, quality, year built, and sale price.

### Key Features

| Feature           | Description                                               | Values/Units                                                                                                           |
|-------------------|-----------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------|
| **1stFlrSF**      | First-floor square footage                                | 334 - 4,692                                                                                                          |
| **2ndFlrSF**      | Second-floor square footage                               | 0 - 2,065                                                                                                            |
| **OverallQual**   | Overall material and finish quality                       | Ordinal scale: 1 (Very Poor) to 10 (Excellent)                                                                       |
| **GrLivArea**     | Above-ground living area (sq. ft.)                        | 334 - 5,642                                                                                                          |
| **YearBuilt**     | Year the house was built                                  | 1872 - 2010                                                                                                          |
| **SalePrice**     | Sale price of the house                                   | \$34,900 - \$755,000                                                                                                 |

---

## Business Requirements

The project addresses two core objectives:

1. **Exploratory Analysis**:
   - Identify key house attributes that influence `SalePrice`.
   - Provide interactive visualizations to illustrate these relationships.

2. **Predictive Modeling**:
   - Build robust machine learning models to predict house prices.
   - Enable predictions for Lydia's inherited houses and other houses in Ames.

---

## Hypotheses

### Hypothesis 1
**Higher `OverallQual` leads to higher `SalePrice`.**

- **Validation Approach**:
  - Compute correlation coefficient between `OverallQual` and `SalePrice`.
  - Visualize relationships using scatter and box plots.

### Hypothesis 2
**Larger `GrLivArea` results in higher `SalePrice`.**

- **Validation Approach**:
  - Compute correlation coefficient and generate scatter plots.

### Hypothesis 3
**Recently remodeled houses (`YearRemodAdd`) have higher `SalePrice`.**

- **Validation Approach**:
  - Analyze trends of average sale prices over remodel years.

---

## Solution Design

### 1. **Data Preprocessing**

- **Handling Missing Values**:
  - **Numerical Features**: Replaced with **median** values to avoid skewness from extreme outliers.
  - **Categorical Features**: Replaced with appropriate **default categories** (`None` for `GarageFinish`, `TA` for `KitchenQual`, etc.).

- **Skewness Handling**:
  - Applied **log transformation** to reduce skewness in features like `SalePrice`.
  - Used **Box-Cox transformation** for strictly positive features to achieve near-normal distribution.

- **Feature Engineering**:
  - Created `TotalSF` as the sum of first-floor, second-floor, and basement areas.
  - Introduced `Qual_TotalSF` as the product of `OverallQual` and `TotalSF` to capture combined effects of quality and size.

---

### 2. **Feature Selection**

Feature selection was driven by **Random Forest feature importance**:
- Selected the top **20 features** based on importance scores.
- Ensured these features balance interpretability and predictive power.

---

### 3. **Model Training**

#### Models Tested
1. **Linear Regression**:
   - Baseline model with no regularization.
   - Assumes linear relationships between features and `SalePrice`.

2. **Ridge Regression**:
   - Introduces L2 regularization to penalize large coefficients.
   - Hyperparameter: `alpha` (range: \(10^{-3}\) to \(10^3\)) was tuned using **cross-validation**.

3. **Lasso Regression**:
   - Employs L1 regularization to perform feature selection by shrinking less important features to zero.
   - Hyperparameter: `alpha` (range: \(10^{-3}\) to \(10^{-0.5}\)) was tuned with cross-validation.

4. **ElasticNet**:
   - Combines L1 and L2 regularization.
   - Hyperparameters: `alpha` and `l1_ratio` (tested values: 0.1, 0.5, 0.9) optimized via cross-validation.

5. **Random Forest**:
   - Ensemble model using decision trees.
   - Parameters:
     - `n_estimators` (number of trees): Set to **100**.
     - `max_depth`: Unlimited, to capture complex relationships.
     - `max_features`: Set to `sqrt` for efficient training.

6. **Gradient Boosting**:
   - Ensemble model focusing on reducing errors of previous trees.
   - Parameters:
     - `n_estimators`: Set to **300**.
     - `learning_rate`: Optimized at **0.05** for balance between training speed and accuracy.

7. **XGBoost**:
   - Highly efficient gradient boosting algorithm.
   - Parameters:
     - `n_estimators`: **300**.
     - `learning_rate`: **0.05**.
     - `max_depth`: **5** to avoid overfitting.

#### Evaluation Metrics
- **Mean Absolute Error (MAE)**: Measures average prediction error.
- **Root Mean Squared Error (RMSE)**: Penalizes larger errors more heavily.
- **R² Score**: Measures variance explained by the model.

---

### 4. **Model Evaluation**

- **Best Performing Model**: **XGBoost**
  - Achieved the lowest RMSE and highest R² score on both training and test sets.
  - Selected for deployment due to its superior balance of accuracy and computational efficiency.

---

## Dashboard Design

### Streamlit Dashboard

The dashboard includes the following interactive pages:

1. **Project Summary**:
   - Overview of objectives, dataset, and methodology.

2. **Feature Correlations**:
   - Heatmaps of correlation matrices.
   - Scatter plots of top correlated features.

3. **House Price Predictions**:
   - Predicted sale prices for Lydia's inherited houses.
   - Real-time prediction tool for any house.

4. **Model Performance**:
   - Comparisons of trained models.
   - Metrics and visualizations (e.g., residual plots, actual vs. predicted prices).

---

## Deployment

### Steps

1. **Set Up Environment**:
   - Install dependencies:
     ```bash
     pip install -r requirements.txt
     ```

2. **Run Locally**:
   - Start the dashboard:
     ```bash
     streamlit run app.py
     ```

3. **Deploy to Streamlit Cloud**:
   - Push code to GitHub.
   - Connect repository to Streamlit Cloud.
   - Configure deployment (e.g., Python version, requirements file).

---

## Requirements

- Python 3.11.10
- Libraries:
  - pandas
  - numpy
  - scikit-learn
  - xgboost
  - streamlit
  - joblib
  - pathlib

---

## Known Bugs

- **None**: All features have been rigorously tested and validated.

---

## Future Enhancements

- Enhance dashboard visualizations.
- Implement ensemble stacking for improved predictive accuracy.
- Automate data updates through web scraping.

---

## Credits

- **Dataset**: [Kaggle Housing Prices](https://www.kaggle.com/codeinstitute/housing-prices-data)
- **Tools**: Python, Streamlit, Scikit-Learn, XGBoost, Matplotlib

---

This README provides comprehensive details, ensuring clarity for both developers and stakeholders. It elaborates on decision-making, methodology, and deployment, delivering a high standard of documentation.
