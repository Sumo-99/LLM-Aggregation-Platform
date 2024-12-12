from flask import Flask, request, jsonify
import logging
import json
import subprocess
import redis
from datetime import datetime
from helpers.get_pem import download_pem_from_s3
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
REDIS_PORT = 6379

def create_redis_client():
    """Set up the Redis client with SSH tunnel."""
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
    """Ping Redis to check connectivity."""
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
    """
    Run inference on the model for a given input.

    Args:
        model: Hugging Face model object.
        tokenizer: Hugging Face tokenizer object.
        input_text (str): Input text for the model.
        task (str): Task type ("generation", "classification", etc.).
        max_new_tokens (int): Maximum number of new tokens to generate.

    Returns:
        str: Generated or processed text.
    """
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
        error_msg = f"Error during initialization: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        return jsonify({"error": error_msg}), 500

@app.route('/process/<ws_session_id>', methods=['POST'])
def process_request(ws_session_id):
    global MODEL, TOKENIZER, ARCHITECTURE

    try:
        # Check if model is initialized
        if MODEL is None or TOKENIZER is None:
            return jsonify({"error": "Model not initialized. Call /init endpoint first"}), 400

        # Get request data
        data = request.get_json()
        user_id = data['user_id']
        model_id = data['model_id']
        sess_id = data['session_id']
        prompt_timestamp = datetime.now().isoformat()
        
        if not data or 'input_text' not in data:
            return jsonify({"error": "Missing 'input_text' in request data"}), 400

        input_text = data['input_text']
        logger.info(f"Processing input text: {input_text}")

        # Run inference
        try:
            result = run_inference(
                model=MODEL,
                tokenizer=TOKENIZER,
                input_text=input_text,
                task=ARCHITECTURE,
                max_new_tokens=data.get('max_new_tokens', 50)
            )
            
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

            try:
                # Store and publish results in Redis
                print(redis_client)
                redis_client.rpush(redis_key, json.dumps(new_entry))
                redis_client.publish(ws_session_id, json.dumps(new_entry))
                return {
                    "redis_key": redis_key,
                    "published_channel": ws_session_id,
                    "data_stored": new_entry
                }
            except Exception as e:
                print("Error with redis", traceback.format_exc())
                raise

        except Exception as e:
            error_msg = f"Error during inference: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            return jsonify({"error": error_msg}), 500

    except Exception as e:
        error_msg = f"Error processing request: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        return jsonify({"error": error_msg}), 500


@app.route('/')
def hello_world():
    return "Dynamic Model Loading API with AutoTokenizer and AutoConfig is running!"


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)