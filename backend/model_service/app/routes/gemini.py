import os
import google.generativeai as genai
import subprocess
import redis

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi import APIRouter
from redis import Redis, ConnectionError

from app.models.common_models import PromptRequest

router = APIRouter()

# Configure Gemini API client
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Bastion host details
BASTION_HOST = "3.230.206.206"  # Public IP of the bastion host
BASTION_USER = "ec2-user"
BASTION_KEY_PATH = "/Users/sumanthramesh/Documents/dev/cloud/jumper.pem"  # Path to the SSH private key

# Configure Redis
REDIS_HOST = "127.0.0.1"  # Localhost, forwarded by the SSH tunnel
REDIS_PORT = 6379

def create_redis_client():
    # Set up SSH tunnel (one-time setup)
    ssh_command = [
        "ssh",
        "-i", "/Users/sumanthramesh/Documents/dev/cloud/jumper.pem",  # Path to your SSH key
        "-o", "StrictHostKeyChecking=no",
        "-L", "127.0.0.1:6379:127.0.0.1:6379",
        "ec2-user@3.230.206.206",
        "-N"
    ]

    try:
        subprocess.Popen(ssh_command)  # Start the SSH tunnel in the background
    except Exception as e:
        print(f"Error starting SSH tunnel: {e}")
        raise

    # Connect to Redis through the tunnel
    return redis.StrictRedis(host="127.0.0.1", port=6379, decode_responses=True)

# Initialize Redis client
redis_client = create_redis_client()

def redis_ping():
    try:
        # Test Redis connection
        if redis_client.ping():
            return {"message": "Redis connection successful!"}
        else:
            return {"error": "Redis connection failed!"}, 500
    except Exception as e:
        return {"error": str(e)}, 500

redis_ping()

@router.post("/chat_gemini")
async def generate_text(request: PromptRequest, tasks: BackgroundTasks):
    try:
        # Select the desired model
        model = genai.GenerativeModel('gemini-1.5-flash')
        # Generate content based on the prompt
        response = model.generate_content(request.prompt)
        
        # TODO:
        session_id = '123'

        try:
            # Perform basic Redis operations
            redis_client.set("test_key", "test_value")
            value = redis_client.get("test_key")
            redis_client.rpush("test_list", "item1", "item2", "item3")
            list_items = redis_client.lrange("test_list", 0, -1)

            return {
                "test_key_value": value,
                "test_list_items": list_items,
                "generated_text": response.text
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

