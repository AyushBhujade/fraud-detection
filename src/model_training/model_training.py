import pickle
import os
from src.logger import logging
import pandas as pd
from xgboost import XGBClassifier

class ModelTrainer:
    def __init__(self):
        logging.info("Model trainer initialized.")
        
    def load_data(self,path:str):
        try:
            df=pd.read_csv(path)
            logging.info(f"Data loaded from {path} successfully.")
            return df
        except Exception as e:
            logging.error(f"Error occurred while loading data from {path}: {e}")
            raise
    def train_model(self,X_train,y_train):
        try:
            logging.info("Model training started.")
            model=XGBClassifier(subsample=0.8,n_estimators=200,learning_rate=0.1,max_depth=10,colsample_bytree=0.8)
            model.fit(X_train,y_train)
            logging.info("Model training completed successfully.")
            return model
        except Exception as e:
            logging.error(f"Error occurred during model training: {e}")
            raise
        
    def save_model(self,model,path:str):
            try:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open (path,"wb") as f:
                    pickle.dump(model,f)
                logging.info(f"Model saved to {path} successfully.")
            except Exception as e:
                logging.error(f"Error occurred while saving model to {path}: {e}")
                raise
    
    def main(self):
        X_train=self.load_data("./data/preprocessed_data/X_train.csv")
        y_train=self.load_data("./data/preprocessed_data/y_train.csv")
        model=self.train_model(X_train,y_train)
        self.save_model(model,"models/xgboost_model.pkl") 
        
if __name__ == "__main__":
    model_trainer=ModelTrainer()
    model_trainer.main()               