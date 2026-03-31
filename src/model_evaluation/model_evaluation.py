import pandas as pd
import pickle
import json
import dagshub
import os
import mlflow
from utils.config_utils.env_loader import load_env
from src.logger import logging
from sklearn.metrics import accuracy_score,precision_score,recall_score , roc_auc_score
from dagshub.common.errors import DagsHubRepoNotFoundError


class ModelEvaluation:
    def __init__(self):
        logging.info("Model evaluation initialized.")
        self.repo_name=load_env("REPO_NAME")
        self.repo_owner=load_env("REPO_OWNER")

        if self.repo_owner and self.repo_name:
            try:
                dagshub.init(repo_owner=self.repo_owner, repo_name=self.repo_name, mlflow=True)
                mlflow.set_tracking_uri(f"https://dagshub.com/{self.repo_owner}/{self.repo_name}.mlflow")
                logging.info("DagsHub tracking enabled.")
            except DagsHubRepoNotFoundError:
                logging.warning("DagsHub repo not found. Falling back to local MLflow tracking.")
            except Exception as e:
                logging.warning(f"Could not initialize DagsHub tracking: {e}. Falling back to local MLflow tracking.")
        else:
            logging.info("DagsHub repo info not set. Using local MLflow tracking.")
        mlflow.set_experiment("XG boost")

    def load_model(self,path:str):
        try:
            with open(path,"rb") as f:
                model=pickle.load(f)
            logging.info(f"Model loaded from {path} successfully.")
            return model
        except Exception as e:
            logging.error(f"Error occurred while loading model from {path}: {e}")
            raise
    
    def evaluate_model(self,model,X_test,y_test):
        try:
            
            logging.info("Model evaluation started.")
            y_pred=model.predict(X_test)
            recall=recall_score(y_test,y_pred)
            precision=precision_score(y_test,y_pred)
            roc_auc=roc_auc_score(y_test,y_pred)
            accuracy=accuracy_score(y_test,y_pred)
            logging.info("Model evaluation completed successfully.")
                
            metric_dict={
                "recall":recall,
                "precision":precision,
                "roc_auc":roc_auc,
                "accuracy":accuracy
            }
                
            logging.info(f"Recall: {recall}")
            logging.info(f"Precision: {precision}") 
            logging.info(f"ROC AUC Score: {roc_auc}")
            logging.info(f"Accuracy: {accuracy}")
            return metric_dict
        except Exception as e:
            logging.error(f"Error occurred during model evaluation: {e}")
            raise
    def save_metric(self,metric_dict:dict,path:str):    
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path,"w") as file:
                json.dump(metric_dict,file,indent=4)
            logging.info(f"Metrics saved to {path} successfully.")
        except Exception as e:
            logging.error(f"Error occurred while saving metrics to {path}: {e}")
            raise

    def save_model_info(self,run_id,model_path,file_path:str,local_model_path:str=None):
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            model_info = {"run_id": run_id, "model_path": model_path}
            if local_model_path:
                model_info["local_model_path"] = local_model_path
            with open(file_path,'w') as file:
                json.dump(model_info,file,indent=4)
            logging.info(f"Model info saved to {file_path} successfully.")
        except Exception as e:
            logging.error(f"Error occurred while saving model info to {file_path}: {e}")
            raise

    def main(self):
        with mlflow.start_run(run_name="Model Evaluation"):
            model=self.load_model("models/xgboost_model.pkl")
            X_test=pd.read_csv("./data/preprocessed_data/X_test.csv")
            y_test=pd.read_csv("./data/preprocessed_data/y_test.csv")
            eval_metric=self.evaluate_model(model,X_test,y_test)
            self.save_metric(eval_metric,"report/evaluation_metrics.json")
            # Keep model artifact path aligned with model_info.json
            self.save_model_info(
                mlflow.active_run().info.run_id,
                "registry_model",
                "report/model_info.json",
                local_model_path="models/model.pkl",
            )
            for metric_name,metric_value in eval_metric.items():
                mlflow.log_metric(metric_name,metric_value)
            
            if hasattr(model,"get_params"):
                params=model.get_params()
                for name, value in params.items():
                    mlflow.log_param(name, value)
            # Only log the model here; registration is handled in model_registry.py
            mlflow.sklearn.log_model(model, "registry_model")
            
            mlflow.log_artifact("report/evaluation_metrics.json",artifact_path="evaluation_metrics")    

            mlflow.log_artifact("report/model_info.json",artifact_path="model_info")
if __name__ == "__main__":
    model_evaluation=ModelEvaluation()
    model_evaluation.main()
