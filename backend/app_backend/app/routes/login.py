import uuid
import os
from fastapi import APIRouter, HTTPException
from passlib.context import CryptContext
import boto3
from boto3.dynamodb.conditions import Key
from datetime import datetime
from app.models.common_models import LoginRequest  # Assuming you have a LoginRequest model defined

# Initialize the APIRouter
router = APIRouter()

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# DynamoDB setup
aws_access_key_id = os.environ['AWS_ACCESS_KEY_ID']
aws_secret_access_key = os.environ['AWS_SECRET_ACCESS_KEY']
dynamodb = boto3.resource(
    'dynamodb',
    region_name='us-east-1',  # Replace with your region
    aws_access_key_id="aws_access_key_id" ,  # Replace with your access key ID
    aws_secret_access_key="aws_secret_access_key"  # Replace with your secret key
)

# Reference to the DynamoDB tables
user_table = dynamodb.Table('user')  # Replace 'user' with your user table name
session_table = dynamodb.Table('session')  # Replace 'session' with your session table name

# Helper function to verify password
def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

# Helper function to get user by email
def get_user_by_email(email: str):
    response = user_table.scan(
        FilterExpression=Key('email').eq(email)
    )
    if 'Items' in response and len(response['Items']) > 0:
        return response['Items'][0]
    return None

@router.post("/login")
def login(user: LoginRequest):
    # Fetch user by email
    db_user = get_user_by_email(user.email)
    if not db_user:
        raise HTTPException(status_code=400, detail="Invalid email or password")
    
    # Verify the password
    if not verify_password(user.password, db_user['password']):
        raise HTTPException(status_code=400, detail="Invalid email or password")

    # Generate a unique session ID
    session_id = str(uuid.uuid4())

    # Create a session entry
    session_item = {
        'session_id': session_id,
        'model_ids': [],
        'chat_name': "",  # Placeholder
        'started_at_timestamp': datetime.utcnow().isoformat(),  # Current timestamp
        'ended_at_timestamp': None,  # Placeholder
        'status': "active",  # Example status
    }

    try:
        session_table.put_item(Item=session_item)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating session: {str(e)}")

    return {
        "message": "Login successful",
        "session_id": session_id,
        "user_id": db_user['user_id']
    }
