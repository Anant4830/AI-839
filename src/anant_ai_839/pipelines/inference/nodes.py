"""
This is a boilerplate pipeline 'inference'
generated using Kedro 0.19.8
"""
import pandas as pd
import joblib
import mlflow

import joblib  # or you can use pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler

from typing import List

def load_model(model_path: str) -> LogisticRegression:
    """
    Loads the trained model from the specified file path.
    
    Args:
        model_path: Path to the saved model.
        
    Returns:
        The loaded logistic regression model.
    """
    model = joblib.load(model_path)
    return model

def predict(model: LogisticRegression, X_new: pd.DataFrame) -> pd.Series:
    """
    Uses the trained model to predict the target for new input data.
    
    Args:
        model: The trained logistic regression model.
        X_new: New input data for prediction.
    
    Returns:
        The predicted target values.
    """
    # 2. One-hot encode categorical columns
    X_new_encoded = pd.get_dummies(X_new, columns=[
        'checking_status', 'credit_history', 'purpose', 'savings_status', 
        'employment', 'personal_status', 'other_parties', 'property_magnitude', 
        'other_payment_plans', 'housing', 'job', 'own_telephone', 'foreign_worker', 
        'health_status'
    ])
    
    # 3. Standardize numerical features
    scaler = StandardScaler()
    numerical_cols = ['duration', 'credit_amount', 'age', 'installment_commitment', 
                      'residence_since', 'existing_credits', 'num_dependents', 
                      'X_1', 'X_2', 'X_3', 'X_4', 'X_5', 'X_6', 'X_7', 
                      'X_8', 'X_9', 'X_10', 'X_11']
    X_new_encoded[numerical_cols] = scaler.fit_transform(X_new_encoded[numerical_cols])
    
    #y is not present
    # 4. Convert the target variable
    #X_new_encoded['y'] = X_new_encoded['y'].astype(int)

    # Get feature names from training (adjust as needed)
    expected_columns = model.feature_names_in_
    print("Expected columns:\t", expected_columns)
    print("\n\n\n\n")
    
    # Align columns
    X_new = X_new_encoded.reindex(columns=expected_columns, fill_value=0)

    predictions = model.predict(X_new)
    #return predictions

    confidence_scores = model.predict_proba(X_new).max(axis=1)  # optional confidence score
    return {
        "predictions": predictions.tolist(),
        "confidence_scores": confidence_scores.tolist()
    }


def log_inference_data(model_name: str, X: pd.DataFrame, predictions: List[dict]):
    """
    Log inference data to MLflow for monitoring.
    """
    if mlflow.active_run():
        mlflow.end_run()
    with mlflow.start_run():
        mlflow.log_param("model_name", model_name)
        mlflow.log_metric("num_samples", len(X))
        #mlflow.log_artifact("inference", predictions)
        mlflow.log_dict(predictions, "predictions.json")
