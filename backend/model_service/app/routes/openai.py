import json
import os
from openai import OpenAI
import subprocess
import redis
from datetime import datetime
from fastapi import HTTPException, BackgroundTasks, APIRouter

from app.models.common_models import PromptRequest
from app.helpers.get_secrets import get_secret

cred_details = get_secret()

router = APIRouter()

# Configure OpenAI API client
openai_api_key = cred_details["OPENAI_API_KEY"]
client = OpenAI(
    api_key=openai_api_key
)

# Bastion host details
BASTION_HOST = "3.230.206.206"  # Public IP of the bastion host
BASTION_USER = "ec2-user"
BASTION_KEY_PATH = "/Users/sumanthramesh/Documents/dev/cloud/jumper.pem"  # Path to the SSH private key

# Configure Redis
REDIS_HOST = "127.0.0.1"  # Localhost, forwarded by the SSH tunnel
REDIS_PORT = 6379

def create_redis_client():
    """Set up the Redis client with SSH tunnel."""
    ssh_command = [
        "ssh",
        "-i", BASTION_KEY_PATH,  # Path to your SSH key
        "-o", "StrictHostKeyChecking=no",
        "-L", "127.0.0.1:6380:127.0.0.1:6379",
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

@router.post("/chat_openai/{ws_session_id}")
async def generate_text(request: PromptRequest, ws_session_id: str, tasks: BackgroundTasks):
    """
    Generate text using OpenAI's GPT model, store and publish results to Redis.

    :param request: The request containing user_id, session_id, model_id, and prompt.
    :param ws_session_id: The WebSocket session ID.
    :param tasks: Background tasks for additional processing.
    :return: The Redis key and data stored.
    """
    try:
        print("Ws Session ID: ", ws_session_id)
        print("OPEN AI APY KEY: ", os.environ["OPENAI_API_KEY"])

        # Extract request details
        user_id = request.user_id
        sess_id = request.session_id
        model_id = request.model_id
        prompt = request.prompt
        prompt_timestamp = datetime.now().isoformat()

        # Use OpenAI API to generate a response
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            response_timestamp = datetime.now().isoformat()
            response_text = response.choices[0].message.content
        except Exception as e:
            import traceback
            print(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"Error with OpenAI API: {str(e)}")

        # Redis keys and entry
        redis_key = f"{user_id}_{sess_id}_{model_id}"
        new_entry = {
            "prompt": prompt,
            "response": response_text,
            "prompt_timestamp": prompt_timestamp,
            "response_timestamp": response_timestamp,
            "model_id": model_id
        }

        try:
            # Store and publish results in Redis
            redis_client.rpush(redis_key, json.dumps(new_entry))
            redis_client.publish(ws_session_id, json.dumps(new_entry))
            return {
                "redis_key": redis_key,
                "published_channel": ws_session_id,
                "data_stored": new_entry
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error with Redis operations: {str(e)}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
