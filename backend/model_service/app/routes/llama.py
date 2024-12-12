import json
import os
from llamaapi import LlamaAPI
import subprocess
import redis
from datetime import datetime
from fastapi import HTTPException, BackgroundTasks, APIRouter
import traceback

from app.models.common_models import PromptRequest
from app.helpers.get_secrets import get_secret
from app.helpers.get_pem import download_pem_from_s3

cred_details = get_secret()

router = APIRouter()

# Configure OpenAI API client
llama_api_key = cred_details["LLAMA_API_KEY"]
client = LlamaAPI(llama_api_key)


# Configure Redis
BASTION_HOST = "34.229.219.213"  # Public IP of the bastion host
BASTION_USER = "ec2-user"
LOCAL_PATH = os.path.join(os.getcwd(), "jumper.pem")  # Relative path for the PEM file
BASTION_KEY_PATH = LOCAL_PATH
REDIS_HOST = "127.0.0.1"  # Localhost, forwarded by the SSH tunnel
REDIS_PORT = 6379

def create_redis_client():
    """Set up the Redis client with SSH tunnel."""
    ssh_command = [
        "ssh",
        "-i", BASTION_KEY_PATH,  # Path to your SSH key
        "-o", "StrictHostKeyChecking=no",
        "-L", "127.0.0.1:6381:127.0.0.1:6379",
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
    """Ping Redis to check connectivity."""
    try:
        if redis_client.ping():
            return {"message": "Redis connection successful!"}
        else:
            return {"error": "Redis connection failed!"}, 500
    except Exception as e:
        return {"error": str(e)}, 500

redis_ping()

@router.post("/chat_llama/{ws_session_id}")
async def generate_text(request: PromptRequest, ws_session_id: str):
    """
    Generate a response using the Llama API with function calls.

    :param request: The request containing user_id, session_id, model_id, and prompt.
    :param ws_session_id: The WebSocket session ID.
    :return: The response content from LlamaAPI.
    """
    try:
        # Extract request details
        user_id = request.user_id
        sess_id = request.session_id
        model_id = request.model_id
        prompt = request.prompt
        prompt_timestamp = datetime.now().isoformat()

        # Build the API request for LlamaAPI
        api_request_json = {
            "model": "llama3.1-70b",
            "messages": [
                {"role": "user", "content": prompt},
            ],
            "stream": False,
        }

        # Execute the API request
        try:
            response = client.run(api_request_json)
            response_data = response.json()
            
            # Extract only the response content
            response_content = response_data["choices"][0]["message"]["content"].strip()
            response_timestamp = datetime.now().isoformat()
        except Exception as e:
            print(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"Error with Llama API: {str(e)}")

        # Build and return the response
        return {
            "ws_session_id": ws_session_id,
            "user_id": user_id,
            "session_id": sess_id,
            "model_id": model_id,
            "prompt": prompt,
            "prompt_timestamp": prompt_timestamp,
            "response": response_content,  # Only include the response content
            "response_timestamp": response_timestamp,
        }

    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error processing the request: {str(e)}")



