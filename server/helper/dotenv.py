from dotenv import load_dotenv
import os
from pathlib import Path

env_path = Path(__file__).resolve().parent.parent / "config" / ".env"
load_dotenv(env_path)

def get_dotenv(env_name):
    return os.getenv(env_name)
