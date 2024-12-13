import boto3
from fastapi import APIRouter, HTTPException
from boto3.dynamodb.conditions import Key
from app.models.common_models import ChatFlow
from boto3.dynamodb.conditions import Key
import os
from fastapi import APIRouter, HTTPException


# Initialize the APIRouter
router = APIRouter()

# DynamoDB setup
aws_access_key_id = os.environ['AWS_ACCESS_KEY_ID']
aws_secret_access_key = os.environ['AWS_SECRET_ACCESS_KEY']
dynamodb = boto3.resource(
    'dynamodb',
    region_name='us-east-1',  # Replace with your region
    aws_access_key_id="aws_access_key_id" ,  # Replace with your access key ID
    aws_secret_access_key="aws_secret_access_key"  # Replace with your secret key
)

# References to DynamoDB tables
chat_history_table = dynamodb.Table('chat_history')  # Replace 'chat' with your chat history table name
chat_messages_table = dynamodb.Table('chat_messages')  # Replace 'chat_messages' with your messages table name

@router.post("/chatflow")
def get_chat_flow(request: ChatFlow):
    # Fetch session_id from chat_history using chat_name
    try:
        response = chat_history_table.query(
            IndexName='chat_name-index',  # Replace with the appropriate index name if chat_name is not the partition key
            KeyConditionExpression=Key('chat_name').eq(request.chat_name)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching chat history: {str(e)}")
    
    if 'Items' not in response or not response['Items']:
        raise HTTPException(status_code=404, detail="Chat not found")

    # Extract session_id from the result
    session_data = response['Items'][0]
    session_id = session_data['session_id']

    # Fetch chat messages from chat_messages table using session_id
    try:
        messages_response = chat_messages_table.query(
            KeyConditionExpression=Key('session_id').eq(session_id)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching chat messages: {str(e)}")

    if 'Items' not in messages_response or not messages_response['Items']:
        return {"session_id": session_id, "models": {}}

    # Process messages to interleave by timestamp
    messages = sorted(messages_response['Items'], key=lambda x: x['created_at_timestamp'])
    models = {"gemini": [], "gpt": [], "claude": [], "llama": []}  # Predefined model categories (adjust if dynamic)

    for message in messages:
        model_name = message['sender']
        if model_name in models:  # Ensure sender matches a model name
            models[model_name].append({
                "sender": message['sender'],
                "content": message['content'],
                "timestamp": message['created_at_timestamp']
            })
        # Add user messages to all models for interleaving
        elif model_name == "user":
            for model in models.values():
                model.append({
                    "sender": message['sender'],
                    "content": message['content'],
                    "timestamp": message['created_at_timestamp']
                })

    # Construct the response
    chat_flow = {
        "session_id": session_id,
        "models": models
    }

    return chat_flow
