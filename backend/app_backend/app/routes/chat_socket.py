import datetime
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from pydantic import BaseModel
from typing import List, Dict

# Initialize APIRouter
router = APIRouter()

# Temporary in-memory storage for session data (this can be replaced with a database)
session_store: Dict[str, Dict] = {}

class WebSocketRequest(BaseModel):
    model_ids: List[str]
    prompt: str
    session_id: str

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
    # Generate a timestamp when the request is received
    timestamp = datetime.datetime.now().isoformat()

    # Store session data in memory
    session_store[request.session_id] = {
        "model_ids": request.model_ids,
        "prompt": request.prompt,
        "timestamp": timestamp
    }

    return {
        "message": "WebSocket session initialization received.",
        "timestamp": timestamp,
        "session_id": request.session_id
    }

@router.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint to handle the connection."""
    # Check if session exists
    if session_id not in session_store:
        await websocket.close(code=4000, reason="Invalid session ID")
        return

    # Retrieve session data
    session_data = session_store[session_id]
    model_ids = session_data["model_ids"]
    prompt = session_data["prompt"]

    # Connect the WebSocket
    await manager.connect(session_id, websocket)
    try:
        # Log the session data
        print(f"Session {session_id} started with model_ids: {model_ids}, prompt: '{prompt}'")

        # Handle incoming messages from the client
        while True:
            data = await websocket.receive_text()
            response = f"Received: {data} | Model IDs: {model_ids} | Prompt: {prompt}"

            # trigger EKS with the prompt

            # Check redis for result

            # send result as response

            await manager.send_message(session_id, response)
    except WebSocketDisconnect:
        manager.disconnect(session_id)
        print(f"WebSocket disconnected for session {session_id}")
