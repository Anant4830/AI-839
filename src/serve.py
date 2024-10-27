import mlflow.pyfunc
import mlflow

# Load the model
model_uri = "runs:/f6d03ea7b25541cbac9116494fe85743/model"
model = mlflow.pyfunc.load_model(model_uri)

print("Model loaded successfully.")

# Serve the model
print("Starting the model server...")
mlflow.models.serve(model_uri=model_uri, host="127.0.0.1", port=5000)
print("Model server is running.")
