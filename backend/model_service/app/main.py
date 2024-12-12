import os
from fastapi import FastAPI
from app.routes import test
from app.helpers.get_pem import download_pem_from_s3

# Download pem file
BUCKET_NAME = "llm-platform-general"
S3_KEY = "jumper.pem"  # Replace with the key of your PEM file
LOCAL_PATH = os.path.join(os.getcwd(), "jumper.pem")  # Relative path for the PEM file
download_pem_from_s3(BUCKET_NAME, S3_KEY, LOCAL_PATH)

from app.routes import gemini
from app.routes import openai
from app.routes import llama
from app.routes import claude


app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome to the model service!"}

# Include routes
app.include_router(test.router)
app.include_router(gemini.router)
app.include_router(openai.router)
app.include_router(llama.router)
app.include_router(claude.router)