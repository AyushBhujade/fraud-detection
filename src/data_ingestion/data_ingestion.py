import os
import pandas as pd
from src.logger import logging
from utils.db_connection import Mongoclient
from utils.config_utils.env_loader import load_env
from sklearn.model_selection import train_test_split

class DataIngestion:
    def __init__(self):
        client=Mongoclient()
        self.client = client.connect_db()
    def load_data(self):
        try:
                logging.info("data loading started from MongoDB")
                data = self.client["fraud-detection"]["raw_data"].find()
                df = pd.DataFrame(list(data))
                logging.info("data loaded successfully from MongoDB")
                return df
        except Exception as e:
            logging.error(f"Error occurred while loading data from MongoDB: {e}")
            return pd.DataFrame()
    def preprocess_data(self, df):
        try:
            logging.info("data preprocessing started")
            df.drop(columns=["_id"], inplace=True)
            logging.info("data preprocessing completed successfully")
            return df
        except Exception as e:
            logging.error(f"Error occurred during data preprocessing: {e}")
            return df
    def save_data(self,train_data:pd.DataFrame,test_data:pd.DataFrame,path:str):
        try:
            logging.info("saving raw data started")
            os.makedirs(path, exist_ok=True)
            train_data.to_csv(os.path.join(path,"train_data.csv"), index=False)
            test_data.to_csv(os.path.join(path,"test_data.csv"), index=False)
            logging.info(f"train and test data saved to {path} successfully")
        except Exception as e:
            logging.error(f"Error occurred while saving train and test data: {e}")
    

def main():
    data_ingestion = DataIngestion()
    df = data_ingestion.load_data()
    df = data_ingestion.preprocess_data(df)
    train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)
    data_ingestion.save_data(train_data, test_data,"data/raw_data")

if __name__ == "__main__":
    main()
