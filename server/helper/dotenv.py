from dotenv import load_dotenv
import os

load_dotenv("../config/.env")

def get_dotenv(env_name):
    return os.getenv(env_name)
