import os
import subprocess
import boto3
import traceback
import uuid
import shutil
from kubernetes import client, config

def generate_files(app_name, requirements, folder_path):
    
    # Create app.py
    app_path = os.path.join(folder_path, "app.py")
    with open(app_path, 'w') as f:
        f.write("""
from flask import Flask, request, jsonify
import logging
from transformers import AutoTokenizer, AutoConfig, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from huggingface_hub import login
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)

MODEL = None  # Hugging Face model object
TOKENIZER = None  # Hugging Face tokenizer object
CONFIG = None  # Hugging Face configuration object
ARCHITECTURE = None  # Store the architecture type provided during initialization


def download_model_from_s3(bucket_name, object_name):

    model_path = object_name

    s3_url = f"https://{bucket_name}.s3.amazonaws.com/{object_name}"
    logger.info(f"Constructed S3 URL: {s3_url}")

    if not os.path.exists(model_path):
        logger.info(f"Downloading model from {s3_url}...")
        os.system(f"curl -o {model_path} {s3_url}")
        logger.info("Model downloaded successfully!")
    else:
        logger.info(f"Model already exists locally: {model_path}")

    return model_path


def load_model_and_tokenizer(architecture, model_path):

    global MODEL, TOKENIZER, CONFIG
    logger.info(f"Loading model and tokenizer for architecture: {architecture}...")

    try:
        if architecture == "bart":
            model_name = "lucadiliello/bart-small"
            TOKENIZER = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
            CONFIG = AutoConfig.from_pretrained(model_name, use_auth_token=True)
            MODEL = AutoModelForSeq2SeqLM.from_pretrained(model_name, use_auth_token=True)
        elif architecture == "llama":
            logger.info(f"Using locally downloaded model path: {model_path}")
            TOKENIZER = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
            CONFIG = AutoConfig.from_pretrained(model_path, local_files_only=True)
            MODEL = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")

        logger.info(f"Model, tokenizer, and config loaded successfully for {architecture.capitalize()}!")
    except Exception as e:
        logger.error(f"Error during loading: {e}")
        raise


@app.route('/init', methods=['POST'])
def init_model():

    global ARCHITECTURE
    login(token=os.environ["HUGGINGFACE_TOKEN"])
    data = request.get_json()
    if not data or 'bucket_name' not in data or 'object_name' not in data or 'architecture' not in data:
        logger.error("Missing 'bucket_name', 'object_name', or 'architecture' in request data")
        return jsonify({"error": "Missing 'bucket_name', 'object_name', or 'architecture' in request data"}), 400

    bucket_name = data['bucket_name']
    object_name = data['object_name']
    ARCHITECTURE = data['architecture']
    logger.info(f"Initializing model with architecture: {ARCHITECTURE}")

    try:
        model_path = download_model_from_s3(bucket_name, object_name)
        load_model_and_tokenizer(ARCHITECTURE, model_path)
        return jsonify({"message": f"{ARCHITECTURE.capitalize()} model initialized successfully!"})
    except Exception as e:
        logger.error(f"Error during model initialization: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/process', methods=['POST'])
def process_request():

    data = request.get_json()
    if not data or 'input_text' not in data:
        logger.error("Missing 'input_text' in request data")
        return jsonify({"error": "Missing 'input_text' in request data"}), 400

    input_text = data['input_text']
    logger.info(f"Processing input text: {input_text}")

    try:
        if ARCHITECTURE in ["bart", "llama"]:
            # Tokenize input and generate output
            inputs = TOKENIZER.encode(input_text, return_tensors="pt")
            outputs = MODEL.generate(inputs, max_length=50, num_beams=5, early_stopping=True)
            generated_text = TOKENIZER.decode(outputs[0], skip_special_tokens=True)

            logger.info(f"Generated response: {generated_text}")
            return jsonify({"response": generated_text})
        else:
            logger.error(f"Unsupported architecture: {ARCHITECTURE}")
            return jsonify({"error": f"Unsupported architecture: {ARCHITECTURE}"}), 400
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/')
def hello_world():
    return "Dynamic Model Loading API with AutoTokenizer and AutoConfig is running!"


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
""")
    
    dockerfile_path = os.path.join(folder_path, "Dockerfile")
    # Create Dockerfile
    with open(dockerfile_path, 'w') as f:
        f.write(f"""
FROM python:3.8-slim

WORKDIR /app

RUN apt-get update && apt-get install -y curl && apt-get clean

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
        ports=[client.V1ContainerPort(container_port=5000)],
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
            ports=[client.V1ServicePort(port=80, target_port=5000)],
            type="LoadBalancer"
        )
    )

    # Create the service in Kubernetes
    core_v1_api = client.CoreV1Api()
    core_v1_api.create_namespaced_service(namespace="default", body=service)
    print(f"Service '{app_name}' created successfully.")


if __name__ == "__main__":
    APP_NAME = "hello-world-app"
    REQUIREMENTS = ["flask", "torch", "transformers", "safetensors", "huggingface_hub"]
    ECR_REPOSITORY = "982081088641.dkr.ecr.us-east-1.amazonaws.com/model-service"

    # Create a unique folder for the container files
    folder_path = f"container_{str(uuid.uuid4())}"
    os.makedirs(folder_path, exist_ok=True)

    print("Generating application files...")
    generate_files(APP_NAME, REQUIREMENTS, folder_path)  # Pass the correct folder_path here

    print("Building and pushing Docker image...")
    image_uri = build_and_push_image(APP_NAME, ECR_REPOSITORY, folder_path)  # Pass folder_path

    print("Deploying to Kubernetes...")
    try:
        deploy_to_k8s(APP_NAME, image_uri)
    except Exception as e:
        print(traceback.format_exc())

    print("Application deployed successfully!")

    # Cleanup:
    # shutil.rmtree(folder_path)


