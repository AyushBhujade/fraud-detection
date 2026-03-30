import os
import dagshub
import mlflow
import json
import pickle
from mlflow.exceptions import MlflowException
from src.logger import logging

class ModelRegistry:
    def __init__(self):
        logging.info("Model registry initialized.")
        self.repo_name=os.getenv("REPO_NAME")
        self.repo_owner=os.getenv("REPO_OWNER")

        if self.repo_owner and self.repo_name:
            try:
                dagshub.init(repo_owner=self.repo_owner, repo_name=self.repo_name, mlflow=True)
                mlflow.set_tracking_uri(f"https://dagshub.com/{self.repo_owner}/{self.repo_name}.mlflow")
                logging.info("DagsHub tracking enabled for model registry.")
            except Exception as e:
                logging.warning(f"Could not initialize DagsHub tracking for model registry: {e}. Falling back to local MLflow tracking.")
        else:
            logging.info("DagsHub repo info not set. Using local MLflow tracking for model registry.")
    
    def load_model_info(file_path: str) -> dict:
        """Load the model info from a JSON file."""
        try:
            with open(file_path, 'r') as file:
                model_info = json.load(file)
            logging.debug('Model info loaded from %s', file_path)
            return model_info
        except FileNotFoundError:
            logging.error('File not found: %s', file_path)
            raise
        except Exception as e:
            logging.error('Unexpected error occurred while loading the model info: %s', e)
            raise

    def register_model(model_name: str, model_info: dict):
        """Register the model to the MLflow Model Registry."""
        try:
            model_uri = f"runs:/{model_info['run_id']}/{model_info['model_path']}"
            
            # Register the model
            model_version = mlflow.register_model(model_uri, model_name)
            
            # Transition the model to "Staging" stage
            client = mlflow.tracking.MlflowClient()
            client.transition_model_version_stage(
                name=model_name,
                version=model_version.version,
                stage="Staging"
            )
            
            logging.debug(f'Model {model_name} version {model_version.version} registered and transitioned to Staging.')
        except Exception as e:
            logging.error('Error during model registration: %s', e)
            raise

    def main():
        try:
            registry=ModelRegistry()
            model_info_path = 'report/model_info.json'
            model_info = registry.load_model_info(model_info_path)
            
            model_name = "my_model"
            registry.register_model(model_name, model_info)
        except Exception as e:
            logging.error('Failed to complete the model registration process: %s', e)
            print(f"Error: {e}")
