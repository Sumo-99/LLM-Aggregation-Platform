import datetime
import redis
import subprocess
import asyncio
import boto3
import os
import requests
import json
import httpx
from redis import Redis, ConnectionError
from boto3.dynamodb.conditions import Key, Attr
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from pydantic import BaseModel
from typing import List, Dict
from app.config import settings

# Initialize APIRouter
router = APIRouter()

# DynamoDB setup
aws_access_key_id = os.environ['AWS_ACCESS_KEY_ID']
aws_secret_access_key = os.environ['AWS_SECRET_ACCESS_KEY']
region_name = 'us-east-1'  # Replace with your region

dynamodb = boto3.resource(
    'dynamodb',
    region_name=region_name,
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key
)

# Reference the DynamoDB tables
model_table = dynamodb.Table('model')

# Http Client
http_client = httpx.AsyncClient()

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
        "-L", "127.0.0.1:6380:127.0.0.1:6379",
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

# Temporary in-memory storage for session data
session_store: Dict[str, Dict] = {}

class WebSocketRequest(BaseModel):
    model_ids: List[str]
    prompt: str
    session_id: str
    user_id: str

# In-memory store for active WebSocket connections
class WebSocketManager:
    """Manages WebSocket connections."""
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, session_id: str, websocket: WebSocket):
        """Accept and store an incoming WebSocket connection."""
        await websocket.accept()
        self.active_connections[session_id] = websocket

    def disconnect(self, session_id: str):
        """Remove a disconnected WebSocket connection."""
        if session_id in self.active_connections:
            del self.active_connections[session_id]

    async def send_message(self, session_id: str, message: str):
        """Send a message to a specific WebSocket connection."""
        connection = self.active_connections.get(session_id)
        if connection:
            await connection.send_text(message)

# Instantiate the WebSocketManager
manager = WebSocketManager()

@router.post("/start-session")
async def start_session(request: WebSocketRequest):
    """Handles the initial HTTP request for starting a WebSocket session."""
    timestamp = datetime.datetime.now().isoformat()

    # Store session data in memory
    session_store[request.session_id] = {
        "user_id": request.user_id,
        "session_id": request.session_id,
        "model_ids": request.model_ids,
        "all_model_data": []
    }
    # model = model_table.query(
    #     KeyConditionExpression=Key('model_id').eq(model_id)  # Query using the partition key
    # )
    for model_id in request.model_ids:
        # Query the table for the model
        response = model_table.query(
            KeyConditionExpression=Key('model_id').eq(model_id)  # Use query since model_id is the partition key
        )
        
        # Check if the response contains items
        if "Items" in response and response["Items"]:
            # Extract the model_name field
            model_name = response["Items"][0].get("model_name")
            session_store[request.session_id]["all_model_data"].append({
                "model_id": model_id,
                "model_name": model_name
            })
            print(f"Model Name for {model_id}: {model_name}")
        else:
            print(f"Model not found for {model_id}")

    return {
        "message": "WebSocket session initialization received.",
        "timestamp": timestamp,
        "ws_session_id": request.session_id
    }

@router.websocket("/ws/{ws_session_id}")
async def websocket_endpoint(websocket: WebSocket, ws_session_id: str):
    """WebSocket endpoint to handle the connection."""
    # Check if session exists
    if ws_session_id not in session_store:
        await websocket.close(code=4000, reason="Invalid session ID")
        return

    # Connect the WebSocket
    await manager.connect(ws_session_id, websocket)
    all_model_data = session_store[ws_session_id]["all_model_data"]
    try:
        # Process incoming messages
        asyncio.create_task(listen_to_redis(ws_session_id, websocket))
        while True:
            message = await websocket.receive_text()
            print(f"WebSocket message received: {message}")
            asyncio.create_task(process_prompt(message, all_model_data, ws_session_id))
    except WebSocketDisconnect:
        manager.disconnect(ws_session_id)
        print(f"WebSocket disconnected for session {ws_session_id}")

async def process_prompt(message, all_model_data, ws_session_id):
    """Trigger requests to model pods asynchronously."""
    base_url = f"http://127.0.0.1:8080/"
    for model_dict in all_model_data:
        model_api_endpoint = settings.MODEL_API_MAP[model_dict["model_name"]]
        headers = {'Content-Type': 'application/json'}
        payload = {
            "user_id": session_store[ws_session_id]["user_id"],
            "session_id": session_store[ws_session_id]["session_id"],
            "model_id": model_dict["model_id"],
            "prompt": message
        }
        asyncio.create_task(send_request(base_url, model_api_endpoint, payload, headers))

async def send_request(base_url, model_api_endpoint, payload, headers):
    """Send asynchronous HTTP request using the shared client."""
    try:
        response = await http_client.post(f"{base_url}{model_api_endpoint}", json=payload, headers=headers)
        print(f"Model trigger response: {response.text}")
    except Exception as e:
        print(f"Error triggering model: {e}")

def create_pubsub_client():
    return redis.StrictRedis(host="127.0.0.1", port=6379, decode_responses=True)

# Use this new client in listen_to_redis
pubsub_client = create_pubsub_client()

async def listen_to_redis(session_id: str, websocket: WebSocket):
    """Subscribe to Redis updates and send them to the WebSocket client asynchronously."""
    pubsub = pubsub_client.pubsub()
    try:
        pubsub.subscribe(session_id)
        print(f"Subscribed to Redis channel: {session_id}")
        while True:
            # Check for new messages
            message = pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
            if message:
                print(f"Redis message received: {message}")
                if message["type"] == "message":
                    data = message["data"]
                    print(f"Publishing to WebSocket: {data}")
                    await manager.send_message(session_id, data)
            else:
                print("No message received, continuing to wait...")
            await asyncio.sleep(0.1)  # Yield control to the event loop
    except Exception as e:
        print(f"Error while listening to Redis: {e}")
    finally:
        print(f"Unsubscribing from Redis channel: {session_id}")
        pubsub.unsubscribe(session_id)
        pubsub.close()
