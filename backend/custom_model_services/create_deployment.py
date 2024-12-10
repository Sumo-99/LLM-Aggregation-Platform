import os
import subprocess
import boto3
import traceback
from kubernetes import client, config

def generate_files(app_name, requirements):
    # Create app.py
    with open('app.py', 'w') as f:
        f.write("""
from flask import Flask

app = Flask(_name_)

@app.route('/')
def hello_world():
    return "Hello, World!"

if _name_ == '_main_':
    app.run(host='0.0.0.0', port=5000)
""")

    # Create Dockerfile
    with open('Dockerfile', 'w') as f:
        f.write(f"""
FROM python:3.8-slim

WORKDIR /app

COPY app.py .
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

CMD [\"python\", \"app.py\"]
""")

    # Create requirements.txt
    with open('requirements.txt', 'w') as f:
        f.write("\n".join(requirements))

def build_and_push_image(app_name, ecr_repository):
    # Build Docker image
    image_tag = f"{ecr_repository}:{app_name}"
    subprocess.run(["docker", "build", "-t", image_tag, "."], check=True)

    # Authenticate Docker to ECR
    aws_region = boto3.session.Session().region_name
    auth_token = subprocess.check_output([
        "aws", "ecr", "get-login-password", "--region", aws_region
    ]).decode('utf-8')
    subprocess.run([
        "docker", "login", "--username", "AWS", "--password-stdin", f"{ecr_repository}"
    ], input=auth_token, text=True, check=True)

    # Push Docker image to ECR
    subprocess.run(["docker", "push", image_tag], check=True)

    return image_tag

def deploy_to_k8s(app_name, image_uri):
    config.load_kube_config()

    # Define Kubernetes pod spec
    container = client.V1Container(
        name=app_name,
        image=image_uri,
        ports=[client.V1ContainerPort(container_port=5000)]
    )
    print("p1", container)

    template = client.V1PodTemplateSpec(
        metadata=client.V1ObjectMeta(labels={"app": app_name}),
        spec=client.V1PodSpec(containers=[container])
    )
    print("p2", template)

    spec = client.V1DeploymentSpec(
        replicas=1,
        selector=client.V1LabelSelector(match_labels={"app": app_name}),
        template=template
    )
    print("p3", spec)

    deployment = client.V1Deployment(
        api_version="apps/v1",
        kind="Deployment",
        metadata=client.V1ObjectMeta(name=app_name),
        spec=spec
    )
    print("p4", deployment)


    api_instance = client.AppsV1Api()
    api_instance.create_namespaced_deployment(namespace="default", body=deployment)
    print("p5", api_instance)
    # Define Kubernetes service spec
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
    print("p6", service)

    core_v1_api = client.CoreV1Api()
    core_v1_api.create_namespaced_service(namespace="default", body=service)

if __name__ == "__main__":
    APP_NAME = "hello-world-app"
    REQUIREMENTS = ["flask"]  # You can customize this input
    print("Hiiiiiiii")
    # Replace with your ECR repository URI
    ECR_REPOSITORY = "982081088641.dkr.ecr.us-east-1.amazonaws.com/model-service"

    print("Generating application files...")
    generate_files(APP_NAME, REQUIREMENTS)

    print("Building and pushing Docker image...")
    image_uri = build_and_push_image(APP_NAME, ECR_REPOSITORY)

    print("Deploying to Kubernetes...")
    try:
        deploy_to_k8s(APP_NAME, image_uri)
    except Exception as e:
        print(traceback.format_exc())


    print("Application deployed successfully!")