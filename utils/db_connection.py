
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import pandas as pd
from src.logger import logging

    
class Mongoclient:
    def __init__(self):
        # Create a new client and connect to the server
        self.client = MongoClient(uri, server_api=ServerApi('1'))

        # Send a ping to confirm a successful connection
        try:
            self.client.admin.command('ping')
            logging.info("Pinged your deployment. You successfully connected to MongoDB!")
        except Exception as e:
            print(e)
    
    def store_data(self,path):
        df = pd.read_csv("data\creditcard.csv")
        data=df.to_dict(orient="records")
        db = self.client["fraud-detection"]
        collection = db["raw_data"]
        collection.insert_many(data)  
        logging.info("data is store seccessfully")      
            
            
clint=Mongoclient()     
clint.store_data("data\creditcard.csv")       
