"""
This is a boilerplate pipeline 'data_preprocessing'
generated using Kedro 0.19.8
"""
import pandas as pd
from sklearn.preprocessing import StandardScaler
from kedro.io import DataCatalog

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the data, including encoding and scaling."""
    
    # 1. Check for missing values
    print("Missing values:", df.isnull().sum())
    
    # 2. One-hot encode categorical columns
    df_encoded = pd.get_dummies(df, columns=[
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
    df_encoded[numerical_cols] = scaler.fit_transform(df_encoded[numerical_cols])
    
    # 4. Convert the target variable
    df_encoded['y'] = df_encoded['y'].astype(int)
    
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
    
    print("Shape of dataframe is:", df_encoded.shape)
    return df_encoded
