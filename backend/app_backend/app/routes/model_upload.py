import uuid
import os
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, EmailStr, validator
from typing import Dict, List
import boto3
import asyncio
from boto3.dynamodb.conditions import Key
import traceback
import uuid
from botocore.exceptions import ClientError
from app.routes.pod_creation import create_docker_and_pod

# Initialize the APIRouter
router = APIRouter()

# Initialize S3 client with error handling
try:
    aws_access_key_id = os.environ['AWS_ACCESS_KEY_ID']
    aws_secret_access_key = os.environ['AWS_SECRET_ACCESS_KEY']
    region_name = 'us-east-1'  # Replace with your region

    s3_client = boto3.client(
        's3',
        region_name=region_name,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key
    )
except KeyError as e:
    raise Exception(f"Missing required environment variable: {str(e)}")
except Exception as e:
    raise Exception(f"Failed to initialize S3 client: {str(e)}")


def download_folder_from_s3(s3_client, bucket_name, folder_path, local_dir):
    """
    Download all contents of an S3 folder to a local directory.
    
    Args:
        s3_client: Initialized boto3 S3 client
        bucket_name (str): Name of the S3 bucket
        folder_path (str): Path to the folder in S3 (e.g., 'models/v1/')
        local_dir (str): Local directory to save the files
    """
    try:
        # Ensure folder_path ends with '/'
        if not folder_path.endswith('/'):
            folder_path += '/'
            
        # List all objects in the folder
        paginator = s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket_name, Prefix=folder_path)
        
        # Create local directory if it doesn't exist
        os.makedirs(local_dir, exist_ok=True)
        
        for page in pages:
            if 'Contents' not in page:
                print(f"No objects found in {folder_path}")
                return
                
            for obj in page['Contents']:
                # Get the relative path by removing the folder prefix
                s3_path = obj['Key']
                relative_path = s3_path[len(folder_path):] if s3_path != folder_path else ''
                
                # Skip if this is a folder marker (empty object ending with '/')
                if not relative_path or s3_path.endswith('/'):
                    continue
                    
                # Create the local file path
                local_file_path = os.path.join(local_dir, relative_path)
                
                # Create directories if they don't exist
                os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
                
                # Download the file
                print(f"Downloading: {s3_path}")
                s3_client.download_file(bucket_name, s3_path, local_file_path)
                
        print(f"Successfully downloaded all files to {local_dir}")
                
    except ClientError as e:
        print(f"AWS Error: {str(e)}")
    except Exception as e:
        print(f"Error downloading folder: {str(e)}")

class S3PathModel(BaseModel):
    s3_bucket: str
    s3_key: str
    app_name: str

    @validator('s3_bucket')
    def validate_bucket(cls, v):
        if not v:
            raise ValueError('S3 bucket name cannot be empty')
        return v

    @validator('s3_key')
    def validate_key(cls, v):
        if not v:
            raise ValueError('S3 key cannot be empty')
        # Ensure the key doesn't start with a slash
        return v.lstrip('/')

@router.post("/upload_model")
async def upload_model(request: S3PathModel, background_tasks: BackgroundTasks):
    process_id = str(uuid.uuid4())
    
    try:

        background_tasks.add_task(create_docker_and_pod, request.app_name, request.s3_bucket, request.s3_key)
        
        return {
            "message": f"Directory download will be processed. Process ID: {process_id}",
            "process_id": process_id,
            "s3_bucket": request.s3_bucket,
            "s3_key": request.s3_key
        }
    
    except HTTPException:
        raise
    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred: {str(e)}"
        )