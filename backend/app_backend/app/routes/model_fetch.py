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


def get_models(user_id: str) -> List[Dict]:
    try:
        # Query the model table to fetch all default models
        response_default = model_table.scan(
            FilterExpression=Attr('is_default').eq(True)
        )
        default_models = response_default.get('Items', [])

        # Query the model table to fetch models specific to the user
        response_user = model_table.scan(
            FilterExpression=Attr('user_id').eq(user_id)
        )
        user_models = response_user.get('Items', [])

        # Combine default models and user models
        models = default_models + user_models

        # Extract only model_id and model_name
        return [{"model_id": model["model_id"], "model_name": model["model_name"]} for model in models]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error querying models: {str(e)}")





@router.get("/models")
def fetch_models(user_id: str):
    try:
        # Get the models from the model table
        models = get_models(user_id)
        return {"models": models}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching models: {str(e)}")
