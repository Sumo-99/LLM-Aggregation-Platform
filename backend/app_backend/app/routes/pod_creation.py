import uuid
import os
from xml.dom import ValidationErr
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, EmailStr, validator
from typing import Dict, List
import boto3
import asyncio
from boto3.dynamodb.conditions import Key
import traceback
import uuid
from botocore.exceptions import ClientError
import shutil
import requests
import time
import subprocess
from kubernetes import client, config

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

def generate_files(app_name, requirements, folder_path):
    
    # Create app.py
    app_path = os.path.join(folder_path, "app.py")
    with open(app_path, 'w') as f:
        f.write("""
from flask import Flask, request, jsonify
import logging
import json
import subprocess
import redis
from datetime import datetime

from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoModel,
)
import traceback
from huggingface_hub import login
import os
import boto3

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)


# Download pem file
BUCKET_NAME = "llm-platform-general"
S3_KEY = "jumper.pem"  # Replace with the key of your PEM file
LOCAL_PATH = os.path.join(os.getcwd(), "jumper.pem")  # Relative path for the PEM file

error_msg = "ERROR!"

def download_pem_from_s3(bucket_name, s3_key, local_path):
    # aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
    # aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    region_name = os.getenv("AWS_REGION", "us-east-1")
    aws_access_key_id = ""
    aws_secret_access_key = ""
    # Create S3 clients
    s3_client = boto3.client(
        "s3",
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=region_name,
    )

    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        # Download the file from S3
        print(f"Downloading {s3_key} from bucket {bucket_name} to {local_path}...")
        s3_client.download_file(bucket_name, s3_key, local_path)
        print("Download successful!")

        # Set correct permissions (chmod 400)
        os.chmod(local_path, 0o400)  # User read-only permission
        print(f"Permissions for {local_path} set to 400 (read-only for owner).")
    except Exception as e:
        print(f"Error downloading or setting permissions for file: {e}")
        raise
                

                
download_pem_from_s3(BUCKET_NAME, S3_KEY, LOCAL_PATH)

MODEL = None  # Hugging Face model object
TOKENIZER = None  # Hugging Face tokenizer object
CONFIG = None  # Hugging Face configuration object
ARCHITECTURE = None  # Store the architecture type provided during initialization

# Bastion host configuration
BASTION_HOST = "34.229.219.213"  # Public IP of the bastion host
BASTION_USER = "ec2-user"
BASTION_KEY_PATH = os.path.join(os.getcwd(), "jumper.pem")  # Path to the SSH private key

# Redis configuration
REDIS_HOST = "127.0.0.1"  # Localhost, forwarded by the SSH tunnel
REDIS_PORT = 6383

def create_redis_client():
    ssh_command = [
        "ssh",
        "-i", BASTION_KEY_PATH,  # Path to your SSH key
        "-o", "StrictHostKeyChecking=no",
        "-L", "127.0.0.1:6383:127.0.0.1:6379",
        f"{BASTION_USER}@{BASTION_HOST}",
        "-N"
    ]

    try:
        subprocess.Popen(ssh_command)  # Start the SSH tunnel in the background
    except Exception as e:
        print(f"Error starting SSH tunnel: {e}")
        raise

    # Connect to Redis through the tunnel
    return redis.StrictRedis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

# Initialize Redis client
redis_client = create_redis_client()

def redis_ping():
    try:
        if redis_client.ping():
            return {"message": "Redis connection successful!"}
        else:
            return {"error": "Redis connection failed!"}, 500
    except Exception as e:
        return {"error": str(e)}, 500

redis_ping()
 
# Initialize S3 client with error handling
try:
    # aws_access_key_id = os.environ['AWS_ACCESS_KEY_ID']
    # aws_secret_access_key = os.environ['AWS_SECRET_ACCESS_KEY']
    aws_access_key_id = "AKIA6JKEYFSAUNJZIXWV"
    aws_secret_access_key = "Qq5HZ4gfqxHxYBpEdhpkLk0+KOaGWtLBSW+3bWZG"
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
                
    except Exception as e:
        print(f"Error downloading folder: {str(e)}")

def load_model_and_tokenizer(
    model_name_or_path, 
    task="generation", 
    local_files_only=False
):
    try:
        logger.info(f"Loading model, tokenizer, and config for: {model_name_or_path}")
        
        # Load tokenizer with more robust settings
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            local_files_only=local_files_only,
            use_fast=True,  # Works with more models
            fallback_to_first=True  # Fallback to first available tokenizer
        )

        # Ensure the tokenizer has a pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load configuration
        config = AutoConfig.from_pretrained(
            model_name_or_path,
            local_files_only=local_files_only,
        )

        # More flexible model loading
        if task == "generation":
            try:
                # First attempt: Causal Language Model
                model = AutoModelForCausalLM.from_pretrained(
                    model_name_or_path,
                    config=config,
                    local_files_only=local_files_only,
                )
            except Exception as causal_error:
                try:
                    # Fallback: Sequence-to-Sequence Language Model
                    model = AutoModelForSeq2SeqLM.from_pretrained(
                        model_name_or_path,
                        config=config,
                        local_files_only=local_files_only,
                    )
                except Exception as seq2seq_error:
                    # Final fallback: Generic AutoModel
                    model = AutoModel.from_pretrained(
                        model_name_or_path,
                        config=config,
                        local_files_only=local_files_only,
                    )
        elif task == "classification":
            # Example: Load a generic AutoModel for classification tasks
            model = AutoModel.from_pretrained(
                model_name_or_path,
                config=config,
                local_files_only=local_files_only,
            )
        else:
            raise ValueError(f"Unsupported task: {task}")

        logger.info("Model, tokenizer, and config loaded successfully!")
        return model, tokenizer, config

    except Exception as e:
        logger.error(f"Error during loading: {traceback.format_exc()}")
        raise

def run_inference(model, tokenizer, input_text, task="generation", max_new_tokens=50):
    try:
        # Encode the input text
        inputs = tokenizer(input_text, return_tensors="pt", add_special_tokens=True)

        if task == "generation":
            # Detect the appropriate generation method based on model type
            if hasattr(model, 'generate'):
                # Generate with more controlled parameters
                outputs = model.generate(
                    inputs['input_ids'], 
                    attention_mask=inputs.get('attention_mask'),
                    max_new_tokens=max_new_tokens,
                    do_sample=True,  # Enable sampling for more diverse output
                    temperature=0.7,  # Control randomness
                    top_p=0.9,        # Nucleus sampling
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                    no_repeat_ngram_size=2  # Reduce repetition
                )
                
                # Decode, removing the input prompt
                full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                generated_text = full_text[len(input_text):].strip()
                return generated_text
            else:
                # Fallback for models without generate method
                raise ValueError("Model does not support text generation")
        
        elif task == "classification":
            # Example: Process the outputs for classification tasks
            outputs = model(**inputs)
            return outputs.last_hidden_state  # Return last hidden state for classification
        else:
            raise ValueError(f"Unsupported task: {task}")
    except Exception as e:
        logger.error(f"Error during inference: {traceback.format_exc()}")
        raise

@app.route('/init', methods=['POST'])
def init_model():
    global MODEL, TOKENIZER, CONFIG, ARCHITECTURE
    login(token=os.environ["HUGGINGFACE_TOKEN"])
    try:
        # Get S3 and model parameters from request
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
            
        required_fields = ['s3_bucket', 's3_key']
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({"error": f"Missing required fields: {missing_fields}"}), 400
            
        # Login to Hugging Face
        try:
            login(token=os.environ["HUGGINGFACE_TOKEN"])
        except Exception as e:
            return jsonify({"error": f"Failed to login to Hugging Face: {str(e)}"}), 500

        # Create a local directory for the model
        local_model_dir = f"./downloaded_models/{data['s3_key'].replace('/', '_')}"
        
        # Download model from S3
        try:
            download_folder_from_s3(
                s3_client=s3_client,
                bucket_name=data['s3_bucket'],
                folder_path=data['s3_key'],
                local_dir=local_model_dir
            )
        except Exception as e:
            return jsonify({"error": f"Failed to download model from S3: {str(e)}"}), 500

        # Load model and tokenizer
        try:
            MODEL, TOKENIZER, CONFIG = load_model_and_tokenizer(
                model_name_or_path=local_model_dir,
                task=data.get('task', 'generation'),
                local_files_only=True
            )
            ARCHITECTURE = data.get('task', 'generation')
        except Exception as e:
            return jsonify({"error": f"Failed to load model and tokenizer: {str(e)}"}), 500

        return jsonify({
            "message": "Model initialized successfully",
            "model_path": local_model_dir,
            "task": ARCHITECTURE
        })

    except Exception as e:
        print(traceback.format_exc())
        return jsonify({"error": error_msg}), 500

@app.route('/process/<ws_session_id>', methods=['POST'])
def process_request(ws_session_id):
    global MODEL, TOKENIZER, ARCHITECTURE
    logger.info("Reached /process")
    try:
        # Check if model is initialized
        if MODEL is None or TOKENIZER is None:
            return jsonify({"error": "Model not initialized. Call /init endpoint first"}), 400
        logger.info("Model not none")
        # Get request data
        data = request.get_json()
        user_id = data['user_id']
        model_id = data['model_id']
        sess_id = data['session_id']
        prompt_timestamp = datetime.now().isoformat()
        if not data or 'prompt' not in data:
            return jsonify({"error": "Missing 'prompt' in request data"}), 400
        logger.info(f"deserailized request: {data}")

        input_text = data['prompt']
        logger.info(f"Processing input text: {input_text}")
        logger.debug(f"Processing input text: {input_text}")
        # Run inference
        try:
            result = run_inference(
                model=MODEL,
                tokenizer=TOKENIZER,
                input_text=input_text,
                task=ARCHITECTURE,
                max_new_tokens=data.get('max_new_tokens', 50)
            )
            logger.info(f"Result: {result}")
            logger.debug(f"Result: {result}")
            
            # return jsonify({
            #     "input_text": input_text,
            #     "output": result,
            #     "task": ARCHITECTURE
            # })
        
            # Redis keys and entry
            redis_key = f"{user_id}_{sess_id}_{model_id}"
            new_entry = {
                "prompt": input_text,
                "response": result,
                "prompt_timestamp": prompt_timestamp,
                "response_timestamp": datetime.now().isoformat(),
                "model_id": model_id
            }
            logger.info(f"redis_key: {redis_key}")
            logger.debug(f"redis_key: {redis_key}")
            try:
                # Store and publish results in Redis
                logger.info(f"Redis client: {redis_client}")
                redis_client.publish(ws_session_id, json.dumps(new_entry))
                logger.info(f"R publish Worked!")
                redis_client.rpush(redis_key, json.dumps(new_entry))
                logger.info(f"R push Worked!")
              
                return {
                    "redis_key": redis_key,
                    "published_channel": ws_session_id,
                    "data_stored": new_entry
                }
            except Exception as e:
                print("Error with redis", traceback.format_exc())
                logger.debug(f"Redis Error : {traceback.format_exc()}")
                logger.info(f"Redis Error : {traceback.format_exc()}")
                raise

        except Exception as e:
            print(traceback.format_exc())
            logger.debug(f"Not Redis Error : {traceback.format_exc()}")
            logger.info(f"Not Redis Error : {traceback.format_exc()}")
            return jsonify({"error": error_msg}), 500

    except Exception as e:
        print(traceback.format_exc())
        logger.debug(f"Not Not Redis Error : {traceback.format_exc()}")
        logger.info(f"Not Not Redis Error : {traceback.format_exc()}")
        return jsonify({"error": error_msg}), 500


@app.route('/')
def hello_world():
    return "Dynamic Model Loading API with AutoTokenizer and AutoConfig is running!"


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)

""")
    
    dockerfile_path = os.path.join(folder_path, "Dockerfile")
    # Create Dockerfile
    with open(dockerfile_path, 'w') as f:
        f.write(f"""
FROM python:3.8-slim

WORKDIR /app

RUN apt-get update && apt-get install -y curl openssh-client && apt-get clean

COPY app.py .
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

CMD [\"python\", \"app.py\"]
""")
    requirements_path = os.path.join(folder_path, "requirements.txt")
    # Create requirements.txt
    with open(requirements_path, 'w') as f:
        f.write("\n".join(requirements))

def build_and_push_image(app_name, ecr_repository, folder_path):
    """Build and push the Docker image to ECR."""
    image_tag = f"{ecr_repository}:{app_name}"
    subprocess.run(["docker", "build", "-t", image_tag, folder_path], check=True)

    aws_region = boto3.session.Session().region_name
    auth_token = subprocess.check_output([
        "aws", "ecr", "get-login-password", "--region", aws_region
    ]).decode('utf-8')
    subprocess.run([
        "docker", "login", "--username", "AWS", "--password-stdin", f"{ecr_repository}"
    ], input=auth_token, text=True, check=True)

    subprocess.run(["docker", "push", image_tag], check=True)
    return image_tag

def deploy_to_k8s(app_name, image_uri):
    """Deploy the application to the Kubernetes cluster."""
    config.load_kube_config(context="arn:aws:eks:us-east-1:982081088641:cluster/llm-models")

    # Get the Hugging Face token from the environment
    huggingface_token = os.environ.get("HUGGINGFACE_TOKEN")
    if not huggingface_token:
        raise ValueError("HUGGINGFACE_TOKEN is not set in the environment!")

    # Define the container specification
    container = client.V1Container(
        name=app_name,
        image=image_uri,
        image_pull_policy="Always",
        ports=[client.V1ContainerPort(container_port=5001)],
        env=[
            client.V1EnvVar(name="HUGGINGFACE_TOKEN", value=huggingface_token)
        ]
    )

    # Define the pod template
    template = client.V1PodTemplateSpec(
        metadata=client.V1ObjectMeta(labels={"app": app_name}),
        spec=client.V1PodSpec(containers=[container])
    )

    # Define the deployment specification
    spec = client.V1DeploymentSpec(
        replicas=1,
        selector=client.V1LabelSelector(match_labels={"app": app_name}),
        template=template
    )

    # Define the deployment
    deployment = client.V1Deployment(
        api_version="apps/v1",
        kind="Deployment",
        metadata=client.V1ObjectMeta(name=app_name),
        spec=spec
    )

    # Create the deployment in Kubernetes
    api_instance = client.AppsV1Api()
    api_instance.create_namespaced_deployment(namespace="default", body=deployment)
    print(f"Deployment '{app_name}' created successfully.")

    # Define the service specification
    service = client.V1Service(
        api_version="v1",
        kind="Service",
        metadata=client.V1ObjectMeta(name=app_name),
        spec=client.V1ServiceSpec(
            selector={"app": app_name},
            ports=[client.V1ServicePort(port=80, target_port=5001)],
            type="LoadBalancer"
        )
    )

    # Create the service in Kubernetes
    core_v1_api = client.CoreV1Api()
    core_v1_api.create_namespaced_service(namespace="default", body=service)
    print(f"Service '{app_name}' created successfully.")


def create_docker_and_pod(app_name, s3_bucket, s3_key):
    # Constants
    REQUIREMENTS = ["flask", "torch", "transformers", "safetensors", "huggingface_hub", "redis", "boto3"]
    ECR_REPOSITORY = "982081088641.dkr.ecr.us-east-1.amazonaws.com/model-service"

    # Create a unique folder for the container files
    folder_path = f"container_{str(uuid.uuid4())}"
    os.makedirs(folder_path, exist_ok=True)

    try:
        print(f"Generating application files for {app_name}...")
        generate_files(app_name, REQUIREMENTS, folder_path)

        print("Building and pushing Docker image...")
        image_uri = build_and_push_image(app_name, ECR_REPOSITORY, folder_path)

        print("Deploying to Kubernetes...")
        deploy_to_k8s(app_name, image_uri)

        print("Application deployed successfully!")

        # Wait for the pod to be ready
        print("Waiting for pod to be ready...")
        core_v1_api = client.CoreV1Api()
        wait_for_pod_ready(core_v1_api, app_name)

        # Get the service URL
        service = core_v1_api.read_namespaced_service(app_name, "default")
        if not service.status.load_balancer.ingress:
            raise Exception("Load balancer not ready yet")
        
        service_ip = service.status.load_balancer.ingress[0].hostname or service.status.load_balancer.ingress[0].ip
        service_url = f"http://{service_ip}"
        print("service_ip: ", service_ip)
        print("service_url: ", service_url)

        # Initialize the model by calling /init endpoint
        print("Initializing model...")
        init_url = f"{service_url}/init"
        init_data = {
            "s3_bucket": s3_bucket,
            "s3_key": s3_key,
            "task": "generation"  # You can make this configurable if needed
        }
        
        # Retry logic for initialization
        max_retries = 100
        retry_delay = 15  # seconds
        
        for attempt in range(max_retries):
            try:
                response = requests.post(init_url, json=init_data)
                response.raise_for_status()
                print("Model initialized successfully!")
                break
            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    raise Exception(f"Failed to initialize model after {max_retries} attempts: {str(e)}")
                print(f"Attempt {attempt + 1} failed, retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)

        

        return {
            "status": "success",
            "message": f"Application {app_name} deployed and initialized successfully",
            "image_uri": image_uri,
            "service_url": service_url
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Deployment failed: {str(e)}\n{traceback.format_exc()}"
        )
        
    finally:
        # Clean up the temporary folder
        if os.path.exists(folder_path):
            print(f"Cleaning up temporary files in {folder_path}")
            shutil.rmtree(folder_path)

def wait_for_pod_ready(api, app_name, timeout=300):
    """Wait for the pod to be ready."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        print("Waiting for pod to come up!!!!!")
        pods = api.list_namespaced_pod(
            namespace="default",
            label_selector=f"app={app_name}"
        )
        
        if not pods.items:
            time.sleep(5)
            continue
            
        pod = pods.items[0]
        if pod.status.phase == 'Running':
            # Check if all containers are ready
            if all(container.ready for container in pod.status.container_statuses):
                print("Pod is ready!")
                return True
        
        time.sleep(5)
    
    raise TimeoutError(f"Pod not ready after {timeout} seconds")