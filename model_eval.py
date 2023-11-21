import json
import pathlib
import pickle
from typing import List
from typing import Tuple

import pandas
from sklearn import model_selection
from sklearn import neighbors
from sklearn import pipeline
from sklearn import preprocessing

import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

SALES_PATH = "data/kc_house_data.csv"  # path to CSV with home sale data
DEMOGRAPHICS_PATH = "data/kc_house_data.csv"  # path to CSV with demographics
# List of columns (subset) that will be taken from home sale data
SALES_COLUMN_SELECTION = [
    'price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
    'sqft_above', 'sqft_basement', 'zipcode'
]
OUTPUT_DIR = "model"  # Directory where output artifacts will be saved


def load_data(
    sales_path: str, demographics_path: str, sales_column_selection: List[str]
) -> Tuple[pandas.DataFrame, pandas.Series]:
    """Load the target and feature data by merging sales and demographics.

    Args:
        sales_path: path to CSV file with home sale data
        demographics_path: path to CSV file with home sale data
        sales_column_selection: list of columns from sales data to be used as
            features

    Returns:
        Tuple containg with two elements: a DataFrame and a Series of the same
        length.  The DataFrame contains features for machine learning, the
        series contains the target variable (home sale price).

    """
    data = pandas.read_csv(sales_path,
                           usecols=sales_column_selection,
                           dtype={'zipcode': str})
    demographics = pandas.read_csv("data/zipcode_demographics.csv",
                                   dtype={'zipcode': str})

    merged_data = data.merge(demographics, how="left",
                             on="zipcode").drop(columns="zipcode")
    # Remove the target variable from the dataframe, features will remain
    y = merged_data.pop('price')
    x = merged_data

    return x, y


def main():
    """Load data, train model, and export artifacts."""
    x, y = load_data(SALES_PATH, DEMOGRAPHICS_PATH, SALES_COLUMN_SELECTION)
    x_train, _x_test, y_train, _y_test = model_selection.train_test_split(
        x, y, random_state=42)

    model_knn = pipeline.make_pipeline(preprocessing.RobustScaler(),
                                   neighbors.KNeighborsRegressor()).fit(
                                       x_train, y_train)

    model_new = pipeline.make_pipeline(GradientBoostingRegressor()).fit(
                                       x_train, y_train)

    y_pred_cv_knn = cross_val_predict(model_knn, _x_test, _y_test, cv=5)
    mse_knn = mean_squared_error(_y_test, y_pred_cv_knn)
    mae_knn = mean_absolute_error(_y_test, y_pred_cv_knn)
    r2_knn = r2_score(_y_test, y_pred_cv_knn)

    print(f'KNN Model:')
    print(f'Mean Squared Error (CV): {mse_knn:.2f}')
    print(f'Mean Absolute Error (CV): {mae_knn:.2f}')
    print(f'R-squared (CV): {r2_knn:.2f}')

    y_pred_cv_new = cross_val_predict(model_new, _x_test, _y_test, cv=5)
    mse_new = mean_squared_error(_y_test, y_pred_cv_new)
    mae_new = mean_absolute_error(_y_test, y_pred_cv_new)
    r2_new = r2_score(_y_test, y_pred_cv_new)

    print(f'Improved Model:')
    print(f'Mean Squared Error (CV): {mse_new:.2f}')
    print(f'Mean Absolute Error (CV): {mae_new:.2f}')
    print(f'R-squared (CV): {r2_new:.2f}')

    plt.subplot(1, 2, 1)
    plt.scatter(_y_test, y_pred_cv_knn, label='Model Predictions (KNN)')
    plt.plot([min(_y_test), max(_y_test)], [min(_y_test), max(_y_test)], '--', color='red', label='Perfect Prediction')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs. Predicted Values (KNN)')

    plt.subplot(1, 2, 2)
    plt.scatter(_y_test, y_pred_cv_new, label='Model Predictions (Improved)')
    plt.plot([min(_y_test), max(_y_test)], [min(_y_test), max(_y_test)], '--', color='red', label='Perfect Prediction')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs. Predicted Values (Improved)')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
