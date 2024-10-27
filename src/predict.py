import mlflow
logged_model = 'runs:/f6d03ea7b25541cbac9116494fe85743/model'

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)

# Predict on a Pandas DataFrame.
import pandas as pd
import numpy as np

# Example random data generation for each feature
data = {
    'checking_status': np.random.choice(['<0', '0<=X<200', '>=200', 'no checking'], 1),
    'duration': np.random.randint(1, 100, 1),
    'credit_history': np.random.choice(['no credits/all paid', 'all paid', 'existing paid', 'delayed previously', 'critical/other existing credit'], 1),
    'purpose': np.random.choice(['car', 'furniture/equipment', 'radio/tv', 'domestic appliance', 'repairs', 'education', 'vacation/others'], 1),
    'credit_amount': np.random.randint(100, 10000, 1),
    'savings_status': np.random.choice(['<100', '100<=X<500', '>=500', 'no savings'], 1),
    'employment': np.random.choice(['unemployed', '<1', '1<=X<4', '4<=X<7', '>=7'], 1),
    'installment_commitment': np.random.randint(1, 5, 1),
    'personal_status': np.random.choice(['male single', 'male div/sep', 'male mar/wid', 'female div/dep/mar'], 1),
    'other_parties': np.random.choice(['none', 'co applicant', 'guarantor'], 1),
    'residence_since': np.random.randint(1, 5, 1),
    'property_magnitude': np.random.choice(['real estate', 'life insurance', 'car', 'no known property'], 1),
    'age': np.random.randint(18, 75, 1),
    'other_payment_plans': np.random.choice(['none', 'bank', 'stores'], 1),
    'housing': np.random.choice(['own', 'for free', 'rent'], 1),
    'existing_credits': np.random.randint(1, 5, 1),
    'job': np.random.choice(['unemp/unskilled non res', 'unskilled resident', 'skilled', 'high qualif/self emp/mgmt'], 1),
    'num_dependents': np.random.randint(1, 5, 1),
    'own_telephone': np.random.choice(['none', 'yes'], 1),
    'foreign_worker': np.random.choice(['no', 'yes'], 1),
    'health_status': np.random.choice(['good', 'fair', 'poor'], 1),
}

# Generating additional features X_1 to X_11 with random values
for i in range(1, 12):
    data[f'X_{i}'] = np.random.random(1)

# Convert to DataFrame
df = pd.DataFrame(data)

#importing the preproceess function from the daat_preprocess pipeline
from anant_ai_839.pipelines.data_preprocessing.nodes import process_data_at_inference

import joblib

# Load the scaler and training columns list
scaler = joblib.load("data/04_feature/scaler.pkl")  # The StandardScaler fitted during training
training_columns = joblib.load("data/04_feature/columns.pkl")  # The list of columns from training data

df_processed = process_data_at_inference(df, scaler, training_columns)

loaded_model.predict(pd.DataFrame(df_processed))

print(loaded_model.predict(pd.DataFrame(df_processed)))