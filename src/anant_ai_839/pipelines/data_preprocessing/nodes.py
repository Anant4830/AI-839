"""
This is a boilerplate pipeline 'data_preprocessing'
generated using Kedro 0.19.8
"""
import pandas as pd
from sklearn.preprocessing import StandardScaler
from kedro.io import DataCatalog

from sklearn.model_selection import train_test_split
import typing as t

import pickle


def split_data(df: pd.DataFrame, parameter: t.Dict) -> t.Tuple:
    """
    Splits the input dataframe into training and testing datasets.
    
    Args:
        df: The input dataframe containing the feature columns and target column 'y'.
        parameter: A dictionary with configuration parameters, e.g., "test_size".
    
    Returns:
        A tuple containing:
            - X_train: Training feature data.
            - X_test: Testing feature data.
            - y_train: Training target data.
            - y_test: Testing target data.
    """
    #X = df[parameter["features"]]
    X = df.drop(columns=["y"])
    y = df["y"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=parameter["test_size"]
    )
    return X_train, X_test, y_train, y_test

def preprocess_data(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, scaler_path: str, 
                    columns_path: str) -> t.Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    
    """Preprocess the data, including encoding and scaling."""
    
    # 1. Check for missing values
    print("Missing values:", X_train.isnull().sum())
    
    # 2. One-hot encode categorical columns
    X_train_encoded = pd.get_dummies(X_train, columns=[
        'checking_status', 'credit_history', 'purpose', 'savings_status', 
        'employment', 'personal_status', 'other_parties', 'property_magnitude', 
        'other_payment_plans', 'housing', 'job', 'own_telephone', 'foreign_worker', 
        'health_status'
    ])

    X_test_encoded = pd.get_dummies(X_test, columns=[
        'checking_status', 'credit_history', 'purpose', 'savings_status', 
        'employment', 'personal_status', 'other_parties', 'property_magnitude', 
        'other_payment_plans', 'housing', 'job', 'own_telephone', 'foreign_worker', 
        'health_status'
    ])

    # Align columns in X_test_encoded to match X_train_encoded
    X_test_encoded = X_test_encoded.reindex(columns=X_train_encoded.columns, fill_value=0)
    
    # 3. Standardize numerical features
    scaler = StandardScaler()
    numerical_cols = ['duration', 'credit_amount', 'age', 'installment_commitment', 
                      'residence_since', 'existing_credits', 'num_dependents', 
                      'X_1', 'X_2', 'X_3', 'X_4', 'X_5', 'X_6', 'X_7', 
                      'X_8', 'X_9', 'X_10', 'X_11']
    X_train_encoded[numerical_cols] = scaler.fit_transform(X_train_encoded[numerical_cols])
    X_test_encoded[numerical_cols] = scaler.transform(X_test_encoded[numerical_cols])
    
    # 4. Convert the target variable
    y_train_int = y_train.astype(int)
    
    # Feature Engineering
    # 1. Bin numerical columns
    # df_encoded['age_bin'] = pd.cut(df['age'], bins=[18, 30, 45, 60, 100], labels=['18-30', '31-45', '46-60', '60+'])
    # df_encoded['credit_amount_bin'] = pd.qcut(df['credit_amount'], q=4, labels=['low', 'medium', 'high', 'very high'])
    
    # # 2. Create interaction features
    # df_encoded['employment_credit_interaction'] = df['employment'] + "_" + df['credit_history']
    
    # # 3. Extract numerical information from text
    # df_encoded['employment_min'] = df['employment'].str.extract(r'(\d+)', expand=False).astype(float)
    # df_encoded['checking_min'] = df['checking_status'].str.extract(r'(\d+)', expand=False).astype(float)

    # # Ensure bin columns remain categorical (they won't be standardized)
    # #df_encoded['age_bin'] = df_encoded['age_bin'].astype(str)
    # #df_encoded['credit_amount_bin'] = df_encoded['credit_amount_bin'].astype(str)
    
    # # 4. Target encoding for 'purpose'
    # purpose_mean_target = df.groupby('purpose')['y'].mean()
    # df_encoded['purpose_target_mean'] = df['purpose'].map(purpose_mean_target)

    # Save the scaler and columns
    with open(scaler_path, 'wb') as scaler_file:
        pickle.dump(scaler, scaler_file)

    with open(columns_path, 'wb') as columns_file:
        pickle.dump(X_train_encoded.columns.tolist(), columns_file)
    
    print("Shape of training set is:", X_train_encoded.shape)
    print("Shape of testing set is:", X_test_encoded.shape)

    return X_train_encoded, X_test_encoded, y_train_int

import pandas as pd
from sklearn.preprocessing import StandardScaler

def process_data_at_inference(record: pd.DataFrame, scaler: StandardScaler, training_columns: list) -> pd.DataFrame:
    """
    Preprocess a single data record at inference time, including encoding and scaling.
    
    Parameters:
    - record: A DataFrame containing a single record to be processed.
    - scaler: A fitted StandardScaler instance used for numerical scaling.
    - training_columns: List of columns from the training dataset to align one-hot encoding and feature order.
    
    Returns:
    - A preprocessed DataFrame with one row, ready for inference.
    """
    # 1. One-hot encode categorical columns
    record_encoded = pd.get_dummies(record, columns=[
        'checking_status', 'credit_history', 'purpose', 'savings_status', 
        'employment', 'personal_status', 'other_parties', 'property_magnitude', 
        'other_payment_plans', 'housing', 'job', 'own_telephone', 'foreign_worker', 
        'health_status'
    ])

    # 2. Align columns in record_encoded to match training columns, fill missing with 0
    record_encoded = record_encoded.reindex(columns=training_columns, fill_value=0)
    
    # 3. Standardize numerical features
    numerical_cols = ['duration', 'credit_amount', 'age', 'installment_commitment', 
                      'residence_since', 'existing_credits', 'num_dependents', 
                      'X_1', 'X_2', 'X_3', 'X_4', 'X_5', 'X_6', 'X_7', 
                      'X_8', 'X_9', 'X_10', 'X_11']
    
    # Only standardize columns that are in both the data and training set
    cols_to_standardize = [col for col in numerical_cols if col in record_encoded.columns]
    record_encoded[cols_to_standardize] = scaler.transform(record_encoded[cols_to_standardize])

    return record_encoded

