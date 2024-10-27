from mlflow.models import validate_serving_input

import mlflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")
#current_uri = mlflow.get_tracking_uri()
#print(f"Current MLflow Tracking URI: {current_uri}")

model_uri = 'runs:/f6d03ea7b25541cbac9116494fe85743/model'

# The logged model does not contain an input_example.
# Manually generate a serving payload to verify your model prior to deployment.
from mlflow.models import convert_input_example_to_serving_input

# Define INPUT_EXAMPLE via assignment with your own input example to the model
# A valid input example is a data instance suitable for pyfunc prediction
# INPUT_EXAMPLE = {
#     "duration": 18,
#     "credit_amount": 5000,
#     "age": 25,
#     "employment": "1<=X<4",
#     "savings_status": "<100",
#     # Add any other required fields...
# }
# #serving_payload = convert_input_example_to_serving_input(INPUT_EXAMPLE)

# import pandas as pd
# serving_input = pd.DataFrame([INPUT_EXAMPLE])
# serving_payload = convert_input_example_to_serving_input(serving_input)

import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

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

# # Get feature names from training (adjust as needed)
# expected_columns = ['duration' 'credit_amount' 'installment_commitment' 'residence_since'
#  'age' 'existing_credits' 'num_dependents' 'X_1' 'X_2' 'X_3' 'X_4' 'X_5'
#  'X_6' 'X_7' 'X_8' 'X_9' 'X_10' 'X_11' 'checking_status_0<=X<200'
#  'checking_status_<0' 'checking_status_>=200'
#  'checking_status_no checking' 'credit_history_all paid'
#  'credit_history_critical/other existing credit'
#  'credit_history_delayed previously' 'credit_history_existing paid'
#  'credit_history_no credits/all paid' 'purpose_business'
#  'purpose_education' 'purpose_furniture/equipment' 'purpose_new car'
#  'purpose_other' 'purpose_radio/tv' 'purpose_repairs' 'purpose_used car'
#  'savings_status_100<=X<500' 'savings_status_500<=X<1000'
#  'savings_status_<100' 'savings_status_>=1000'
#  'savings_status_no known savings' 'employment_1<=X<4' 'employment_4<=X<7'
#  'employment_<1' 'employment_>=7' 'employment_unemployed'
#  'personal_status_female div/dep/mar' 'personal_status_male div/sep'
#  'personal_status_male mar/wid' 'personal_status_male single'
#  'other_parties_co applicant' 'other_parties_guarantor'
#  'other_parties_none' 'property_magnitude_car'
#  'property_magnitude_life insurance'
#  'property_magnitude_no known property' 'property_magnitude_real estate'
#  'other_payment_plans_bank' 'other_payment_plans_none'
#  'other_payment_plans_stores' 'housing_for free' 'housing_own'
#  'housing_rent' 'job_high qualif/self emp/mgmt' 'job_skilled'
#  'job_unemp/unskilled non res' 'job_unskilled resident'
#  'own_telephone_none' 'own_telephone_yes' 'foreign_worker_no'
#  'foreign_worker_yes' 'health_status_bad' 'health_status_good']

# Align columns
#df_aligned = df_processed.reindex(columns=expected_columns, fill_value=0)

serving_payload = convert_input_example_to_serving_input(df_processed)

print(serving_payload)

# Validate the serving payload works on the model
validate_serving_input(model_uri, serving_payload)

#from kedro.framework.context import load_context

#context = load_context(r"C:\Users\Admin\Downloads\AI-839\MLOps\kedro-projects\anant-ai-839\src\anant_ai_839")