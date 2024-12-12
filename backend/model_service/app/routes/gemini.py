import json
import os
import google.generativeai as genai
import subprocess
import redis
import time
from datetime import datetime
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi import APIRouter
from redis import Redis, ConnectionError

from app.models.common_models import PromptRequest
from app.helpers.get_secrets import get_secret

cred_details = get_secret()

router = APIRouter()

# Configure Gemini API client
gemini_api_key = cred_details["GEMINI_API_KEY"]
genai.configure(api_key=gemini_api_key)

# Bastion host configuration
BASTION_HOST = "34.229.219.213"  # Public IP of the bastion host
BASTION_USER = "ec2-user"
BASTION_KEY_PATH = os.path.join(os.getcwd(), "jumper.pem")  # Path to the SSH private key

# Redis configuration
REDIS_HOST = "127.0.0.1"  # Localhost, forwarded by the SSH tunnel
REDIS_PORT = 6379

def create_redis_client():
    ssh_command = [
        "ssh",
        "-i", BASTION_KEY_PATH,  # Path to your SSH key
        "-o", "StrictHostKeyChecking=no",
        "-L", "127.0.0.1:6383:127.0.0.1:6379",
        f"{BASTION_USER}@{BASTION_HOST}",
        "-N"
    ]

    try:
        subprocess.Popen(ssh_command)  # Start the SSH tunnel in the background
    except Exception as e:
        print(f"Error starting SSH tunnel: {e}")
        raise

    # Connect to Redis through the tunnel
    return redis.StrictRedis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

# Initialize Redis client
redis_client = create_redis_client()

def redis_ping():
    try:
        if redis_client.ping():
            return {"message": "Redis connection successful!"}
        else:
            return {"error": "Redis connection failed!"}, 500
    except Exception as e:
        return {"error": str(e)}, 500

redis_ping()

@router.post("/chat_gemini/{ws_session_id}")
async def generate_text(request: PromptRequest, ws_session_id: str, tasks: BackgroundTasks):
    try:
        print("Ws Session ID: ",ws_session_id)
        # Select the desired model
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        user_id = request.user_id
        sess_id = request.session_id
        model_id = request.model_id
        prompt = request.prompt
        prompt_timestamp = datetime.now().isoformat()
        response = model.generate_content(request.prompt)
        response_text = response.text.strip()
        response_timestamp = datetime.now().isoformat()

        # TODO:
        ws_session_id = ws_session_id
        # redis_key = f"{session_id}_model"
        redis_key = f"{user_id}_{sess_id}_{model_id}"

        new_entry = {
            "prompt": prompt,
            "response": response_text,
            "prompt_timestamp": prompt_timestamp,
            "response_timestamp": response_timestamp,
            "model_id": model_id
        }

        try:
            redis_client.rpush(redis_key, json.dumps(new_entry))
            redis_client.publish(ws_session_id, json.dumps(new_entry))
            
            return {
                "redis_key": redis_key,
                "published_channel": ws_session_id,
                # "generated_text": response.text
                "data_stored": new_entry
            }
        except Exception as e:
            return {"error": str(e)}, 500

        # store in redis
        # store_data_in_redis("session_id", session_id)
        # return some result to user
        # background process -> store to dynamo db
        # tasks.add_task(nav_master, created_file.id, file_save_path, row_start, row_end, 2, 5, 8)
        
        return {"generated_text": response.text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

