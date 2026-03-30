import os,sys
from src.logger import logging
from dotenv import load_dotenv

# Load .env once at import time so os.getenv can see the values.
load_dotenv()
def load_env(name:str):
    try:
        env_var=os.getenv(name)
        return env_var
    except Exception as e:
        logging.error(f"during fetching env error occured:{e}")
        
    
