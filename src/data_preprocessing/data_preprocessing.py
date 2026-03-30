import os
from src.logger import logging
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

class DataPreprocessor:
    def __init__(self):
        logging.info("DataPreprocessor initialized.")
        self.train_data=pd.read_csv("./data/raw_data/train_data.csv")
        self.test_data=pd.read_csv("./data/raw_data/test_data.csv")
    
    def preprocess_data(self):
        logging.info("Starting data preprocessing.")
        try:
            self.train_data.dropna(inplace=True)
            self.test_data.dropna(inplace=True)

            X_train=self.train_data.drop(columns=["Class"])
            y_train=self.train_data["Class"]
            X_test=self.test_data.drop(columns=["Class"])
            y_test=self.test_data["Class"]
            smote=SMOTE(random_state=42)
            X_train_resampled,y_train_resampled=smote.fit_resample(X_train,y_train)
            logging.info("Data preprocessing completed.")
            return X_train_resampled,y_train_resampled,X_test,y_test
        except Exception as e:
            logging.error(f"Error occurred during data preprocessing: {e}")
            raise
    
    def save_preprocessed_data(self,X_train,y_train,X_test,y_test,path:str):
        try:
            logging.info("Saving preprocessed data started.")
            os.makedirs(path, exist_ok=True)
            pd.DataFrame(X_train).to_csv(os.path.join(path,"X_train.csv"), index=False)
            pd.DataFrame(y_train).to_csv(os.path.join(path,"y_train.csv"), index=False)
            pd.DataFrame(X_test).to_csv(os.path.join(path,"X_test.csv"), index=False)
            pd.DataFrame(y_test).to_csv(os.path.join(path,"y_test.csv"), index=False)
            logging.info(f"Preprocessed data saved to {path} successfully.")
        except Exception as e:
            logging.error(f"Error occurred while saving preprocessed data: {e}")    

        
            
if __name__ == "__main__":
    data_preprocessor=DataPreprocessor()
    X_train,y_train,X_test,y_test=data_preprocessor.preprocess_data()
    data_preprocessor.save_preprocessed_data(X_train,y_train,X_test,y_test,"data/preprocessed_data")    
    