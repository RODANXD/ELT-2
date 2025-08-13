import os

from dotenv import load_dotenv

load_dotenv()


APP_TITLE = os.getenv("APP_TITLE")
API_KEY = os.getenv("API_KEY")
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")
API_VERSION = os.getenv("API_VERSION")