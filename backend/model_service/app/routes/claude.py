import json
import os
import subprocess
import redis
from datetime import datetime
from fastapi import HTTPException, APIRouter
import anthropic
import traceback

from app.models.common_models import PromptRequest
from app.helpers.get_secrets import get_secret

cred_details = get_secret()

router = APIRouter()

# Configure Claude API client
claude_api_key = cred_details["CLAUDE_API_KEY"]
client = anthropic.Anthropic(api_key=claude_api_key)

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
        "-L", "127.0.0.1:6382:127.0.0.1:6379",
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

@router.post("/chat_claude/{ws_session_id}")
async def generate_text(request: PromptRequest, ws_session_id: str):
    """
    Generate a response using the Claude API and store the results in Redis.

    :param request: The request containing user_id, session_id, model_id, and prompt.
    :param ws_session_id: The WebSocket session ID.
    :return: The response content from Claude API.
    """
    try:
        # Extract request details
        user_id = request.user_id
        sess_id = request.session_id
        model_id = request.model_id
        prompt = request.prompt
        prompt_timestamp = datetime.now().isoformat()

        messages = [
            {"role": "user", "content": prompt}  # User input
        ]

        # Call the Claude API
        try:
            response = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1024,
                messages=messages
            )

            response_content = response.content[0].text.strip()
            response_timestamp = datetime.now().isoformat()
            try:
                # If the response is JSON-like, parse it
                parsed_response = json.loads(response_content)
                if isinstance(parsed_response, dict):
                # Check for common keys and extract response
                    if "content" in parsed_response:
                        response_content = parsed_response["content"]
                    elif "text" in parsed_response:  # Handle cases where "text" is used
                        response_content = parsed_response["text"]
                    elif "response" in parsed_response:
                        response_content = parsed_response["response"]
            except json.JSONDecodeError:
                            # If it's not JSON, leave it as plain text
                pass

            
        except Exception as e:
            print(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"Error with Claude API: {str(e)}")

        # Redis keys and entry
        redis_key = f"{user_id}_{sess_id}_{model_id}"
        new_entry = {
            "prompt": prompt,
            "response": response_content,
            "prompt_timestamp": prompt_timestamp,
            "response_timestamp": response_timestamp,
            "model_id": model_id
        }

        # Store and publish results in Redis
        try:
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
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error processing the request: {str(e)}")
