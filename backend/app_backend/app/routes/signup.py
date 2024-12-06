import uuid
import os
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, EmailStr
from typing import Dict
import boto3
from boto3.dynamodb.conditions import Key
from passlib.context import CryptContext
from app.models.common_models import SignupRequest

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
    aws_access_key_id=aws_access_key_id,  # Replace with your access key ID
    aws_secret_access_key=aws_secret_access_key  # Replace with your secret key
)

# Reference to the DynamoDB table
user_table = dynamodb.Table('user')  # Replace 'user' with your table name

# Helper function to hash passwords
def hash_password(password: str) -> str:
    return pwd_context.hash(password)

# Helper function to check if an email already exists
def email_exists(email: str) -> bool:
    # Perform a scan to fetch all items in the table
    response = user_table.scan(
        FilterExpression=Key('email').eq(email)
    )
    # Check if the response contains any matching items
    return 'Items' in response and len(response['Items']) > 0

@router.post("/signup")
def signup(user: SignupRequest):
    # Check if the email already exists
    if email_exists(user.email):
        raise HTTPException(status_code=400, detail="Email already registered")

    # Generate a unique user ID
    user_id = str(uuid.uuid4())

    # Hash the password
    hashed_password = hash_password(user.password)

    # Prepare the item to insert
    item = {
        'user_id': user_id,
        'name': user.name,
        'email': user.email,
        'password': hashed_password
    }

    # Store the user in the DynamoDB table
    try:
        user_table.put_item(Item=item)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving user: {str(e)}")

    return {"message": "User signed up successfully!", "user_id": user_id}
