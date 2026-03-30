import os,sys
from src.logger import logging
def load_env(name:str):
    try:
        env_var=os.getenv(name)
        return env_var
    except Exception as e:
        logging.error(f"during fetching env error occured:{e}")
        
    