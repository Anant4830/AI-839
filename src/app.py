import mlflow

current_uri = mlflow.get_tracking_uri()
print(f"Current MLflow Tracking URI: {current_uri}")

mlflow.set_tracking_uri("http://127.0.0.1:5000")

converted_uri = mlflow.get_tracking_uri()

print(f"Current MLflow Tracking URI: {converted_uri}")

#List Experiments with search_experiments
#print(mlflow.search_experiments())
experiments = mlflow.search_experiments()
print(experiments)

#Confirm run access
from mlflow.tracking import MlflowClient
client = MlflowClient()
run = client.get_run("95f2476da1df477fac3413f1b73a77ae")  # Replace with the specific run ID
print(run)

#Verify model loading
model_uri = "runs:/95f2476da1df477fac3413f1b73a77ae/model"
model = mlflow.pyfunc.load_model(model_uri)

print("Model is\n", model)



