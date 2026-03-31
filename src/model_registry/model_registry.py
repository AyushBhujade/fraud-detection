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
        self.repo_name = load_env("REPO_NAME")
        self.repo_owner = load_env("REPO_OWNER")

        if self.repo_owner and self.repo_name:
            try:
                dagshub.init(repo_owner=self.repo_owner, repo_name=self.repo_name, mlflow=True)
                mlflow.set_tracking_uri(f"https://dagshub.com/{self.repo_owner}/{self.repo_name}.mlflow")
                logging.info("DagsHub tracking enabled for model registry.")
            except Exception as e:
                logging.warning(f"Could not initialize DagsHub tracking for model registry: {e}. Falling back to local MLflow tracking.")
        else:
            logging.info("DagsHub repo info not set. Using local MLflow tracking for model registry.")
    
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
            run_id = model_info["run_id"]
            primary_path = model_info["model_path"]
            # Try primary path first; fall back to legacy "model" if needed
            candidate_paths = [primary_path]
            if primary_path != "model":
                candidate_paths.append("model")

            model_version = None
            last_error = None
            for artifact_path in candidate_paths:
                model_uri = f"runs:/{run_id}/{artifact_path}"
                try:
                    model_version = mlflow.register_model(model_uri, model_name)
                    logging.info(f"Registered model from artifact path '{artifact_path}'.")
                    break
                except Exception as e:
                    last_error = e
                    if "Unable to find a logged_model with artifact_path" not in str(e):
                        raise

            if model_version is None:
                raise last_error
            
            # Transition the model to "Staging" stage
            client = mlflow.tracking.MlflowClient()
            client.transition_model_version_stage(
                name=model_name,
                version=model_version.version,
                stage="Staging",
                archive_existing_versions=False   # 🔥 ADD THIS
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
        
        model_name = "XGBoost"
        registry.register_model(model_name, model_info)
    except Exception as e:
        logging.error('Failed to complete the model registration process: %s', e)
        print(f"Error: {e}")
        
        
if __name__ == "__main__":
    main()        
