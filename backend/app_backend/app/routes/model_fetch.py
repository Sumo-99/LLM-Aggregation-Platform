import os
import boto3
from fastapi import APIRouter, HTTPException
from boto3.dynamodb.conditions import Key, Attr
from typing import List, Dict

# Initialize the APIRouter
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
model_table = dynamodb.Table('model')  # Replace with your model table name


# Helper function to get models based on user_id and is_default
def get_models(user_id: str) -> List[Dict]:
    # Query the model table to fetch default models and models for a specific user_id
    response = model_table.scan(
        FilterExpression=Attr('is_default').eq(True) | (Attr('user_id').exists() & Attr('user_id').eq(user_id))
    )
    return response.get('Items', [])


@router.get("/models")
def fetch_models(user_id: str):
    # Get the models from the model table
    try:
        models = get_models(user_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching models: {str(e)}")

    return {"models": models}
