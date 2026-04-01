import os
import dagshub
import mlflow
import json
import pickle
from mlflow.exceptions import MlflowException
from utils.config_utils.env_loader import load_env
from src.logger import logging

class ModelRegistry:
    def __init__(self):
        logging.info("Model registry initialized.")
        # Ensure .env is loaded so registry uses the same tracking as evaluation
        dagshub_token=os.getenv("DAGSHUB_AUTH_TOKEN")
        self.repo_name="fraud-detection"
        self.repo_owner="ayushbhujade2005"
        if dagshub_token:
            os.environ["MLFLOW_TRACKING_USERNAME"] = self.repo_owner
            os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token
            logging.info("DagsHub authentication configured via MLflow.")
        else:
            raise ValueError("DAGSHUB_AUTH_TOKEN not found.")
        
        # ✅ Set tracking URI directly
        mlflow.set_tracking_uri(f"https://dagshub.com/{self.repo_owner}/{self.repo_name}.mlflow") 
        
        
    
    def load_model_info(self,file_path: str) -> dict:
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

    def register_model(self,model_name: str, model_info: dict):
        """Register the model to the MLflow Model Registry."""
        try:
            # Log + register locally to avoid remote artifact sync delays.
            local_model_path = model_info.get("local_model_path")
            if not local_model_path:
                raise ValueError("local_model_path is missing in model_info.json")

            with mlflow.start_run(run_name="Model Registry"):
                with open(local_model_path, "rb") as f:
                    model = pickle.load(f)
                model_info_obj = mlflow.sklearn.log_model(
                    model,
                    "registry_model",
                    registered_model_name=model_name,
                )
                model_version = getattr(model_info_obj, "registered_model_version", None)
                if model_version is None:
                    client = mlflow.tracking.MlflowClient()
                    latest = client.get_latest_versions(model_name)
                    if latest:
                        model_version = latest[0].version

            # Transition the model to "Staging" stage
            client = mlflow.tracking.MlflowClient()
            client.transition_model_version_stage(
                name=model_name,
                version=model_version,
                stage="Staging",
                archive_existing_versions=False
            )
            
            logging.debug(f"Model {model_name} version {model_version} registered and transitioned to Staging.")
        except Exception as e:
            logging.error('Error during model registration: %s', e)
            raise

def main():
    try:
        registry=ModelRegistry()
        model_info_path = 'report/model_info.json'
        model_info = registry.load_model_info(model_info_path)
        
        model_name = "new_XGBoost"
        registry.register_model(model_name, model_info)
    except Exception as e:
        logging.error('Failed to complete the model registration process: %s', e)
        print(f"Error: {e}")
        
        
if __name__ == "__main__":
    main()        
