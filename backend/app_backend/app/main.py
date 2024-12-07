import asyncio
import subprocess
import redis.asyncio as redis
from fastapi import FastAPI
from app.routes import test
from app.routes import signup
from app.routes import login
from app.routes import model_fetch
from app.routes import chat_socket
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes.chat_socket import router as chat_socket_router

# Redis connection and SSH tunneling configuration
BASTION_HOST = "3.230.206.206"  # Public IP of your bastion host
BASTION_USER = "ec2-user"
BASTION_KEY_PATH = "/Users/sumanthramesh/Documents/dev/cloud/jumper.pem"  # Path to SSH private key
REDIS_HOST = "127.0.0.1"  # Localhost, forwarded by SSH tunnel
REDIS_PORT = 6379

redis_client = None
ssh_tunnel_process = None

async def setup_ssh_tunnel():
    """Set up an SSH tunnel to connect to Redis."""
    global ssh_tunnel_process

    ssh_command = [
        "ssh",
        "-i", BASTION_KEY_PATH,
        "-o", "StrictHostKeyChecking=no",
        "-L", f"{REDIS_PORT}:{REDIS_HOST}:{REDIS_PORT}",
        f"{BASTION_USER}@{BASTION_HOST}",
        "-N"
    ]

    # Start the SSH tunnel as a subprocess
    try:
        ssh_tunnel_process = subprocess.Popen(ssh_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("SSH tunnel established.")
    except Exception as e:
        print(f"Error starting SSH tunnel: {e}")
        raise


async def setup_redis():
    """Set up Redis asynchronously after establishing the SSH tunnel."""
    global redis_client

    # Set up the SSH tunnel first
    await setup_ssh_tunnel()

    # Connect to Redis through the tunnel
    redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
    try:
        if await redis_client.ping():
            print("Redis connection successful!")
        else:
            print("Redis connection failed!")
    except Exception as e:
        print(f"Error connecting to Redis: {e}")


async def close_ssh_tunnel():
    """Close the SSH tunnel when shutting down the app."""
    global ssh_tunnel_process
    if ssh_tunnel_process:
        ssh_tunnel_process.terminate()
        ssh_tunnel_process.wait()
        print("SSH tunnel closed.")

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
app.include_router(chat_socket.router)

# @app.on_event("startup")
# async def startup_event():
#     """Initialize Redis connection during app startup."""
#     await setup_redis()

# @app.on_event("shutdown")
# async def shutdown_event():
#     """Close Redis connection and SSH tunnel during app shutdown."""
#     if redis_client:
#         await redis_client.close()
#         print("Redis connection closed.")
#     await close_ssh_tunnel()

