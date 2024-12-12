import asyncio
import subprocess
import os
import redis.asyncio as redis
from fastapi import FastAPI
from app.helpers.get_pem import download_pem_from_s3
from app.routes import test
from app.routes import signup
from app.routes import login
from app.routes import chathistory
from app.routes import model_fetch
from app.routes import chatflow
from app.routes import chat_socket
from app.routes import model_upload
from app.routes import pod_creation
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Download pem file
BUCKET_NAME = "llm-platform-general"
S3_KEY = "jumper.pem"  # Replace with the key of your PEM file
LOCAL_PATH = os.path.join(os.getcwd(), "jumper.pem")  # Relative path for the PEM file
download_pem_from_s3(BUCKET_NAME, S3_KEY, LOCAL_PATH)

app = FastAPI()

#CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://myfrontend.com"],  # Add your frontends here
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["Authorization", "Content-Type"],
)

@app.get("/")
def read_root():
    return {"message": "Welcome to the app backend service!"}

# Include routes
app.include_router(test.router)
app.include_router(signup.router)
app.include_router(login.router)
app.include_router(model_fetch.router)
app.include_router(chathistory.router)
app.include_router(chatflow.router)
app.include_router(chat_socket.router)
app.include_router(model_upload.router)
app.include_router(pod_creation.router)