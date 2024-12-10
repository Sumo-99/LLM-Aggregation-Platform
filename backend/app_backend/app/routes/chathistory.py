import boto3
from fastapi import APIRouter, HTTPException
from boto3.dynamodb.conditions import Key
from app.models.common_models import ChatHistory  # Importing the ChatHistory model
from boto3.dynamodb.conditions import Key
import os
from fastapi import APIRouter, HTTPException

# Initialize the APIRouter
router = APIRouter()

# DynamoDB setups
aws_access_key_id = os.environ['AWS_ACCESS_KEY_ID']
aws_secret_access_key = os.environ['AWS_SECRET_ACCESS_KEY']
dynamodb = boto3.resource(
    'dynamodb',
    region_name='us-east-1',  # Replace with your region
    aws_access_key_id="aws_access_key_id" ,  # Replace with your access key ID
    aws_secret_access_key="aws_secret_access_key"  # Replace with your secret key
)

# Reference to the DynamoDB table
chat_table = dynamodb.Table('chat_history')  # Replace 'chat' with your chat table name

@router.post("/chathistory")
def get_chat_history(request: ChatHistory):
    try:
        # Query DynamoDB for chats belonging to the user, sorted by created_at_timestamp
        response = chat_table.query(
            KeyConditionExpression=Key('user_id').eq(request.user_id),
            ScanIndexForward=False  # Descending order
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching chat history: {str(e)}")

    # If no chats found, return an empty list
    if 'Items' not in response or not response['Items']:
        return []

    # Extract the required fields from each chat record
    chat_history = [
        {
            "chat_id": item["chat_id"],
            "chat_name": item["chat_name"],
            "created_at_timestamp": item["created_at_timestamp"],
            "session_id": item["session_id"]
        }
        for item in response['Items']
    ]

    return chat_history
