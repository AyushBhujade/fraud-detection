from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import pandas as pd
from src.logger import logging
from utils.config_utils.env_loader import load_env


class Mongoclient:
    def __init__(self):
        monog_db_uri = load_env("MONGO_DB_URI")
        # Create a new client and connect to the server
        self.client = MongoClient(monog_db_uri, server_api=ServerApi("1"))

        # Send a ping to confirm a successful connection
        try:
            self.client.admin.command("ping")
            logging.info(
                "Pinged your deployment. You successfully connected to MongoDB!"
            )
        except Exception as e:
            logging.error(f"Error occurred while connecting to MongoDB: {e}")

    def connect_db(self):
        return self.client

    def store_data(self, path):
        df = pd.read_csv(path)
        data = df.to_dict(orient="records")
        db = self.client["fraud-detection"]
        collection = db["raw_data"]
        collection.insert_many(data)
        logging.info("data is stored successfully in MongoDB")
