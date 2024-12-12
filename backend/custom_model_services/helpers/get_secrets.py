import boto3
import os
import traceback

def get_secret():
    region_name = "us-east-1"  # Change this to your AWS region

    # Initialize a session using environment variables
    session = boto3.session.Session(
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=region_name,
    )
    client = session.client("secretsmanager")

    try:
        import json
        response = client.get_secret_value(SecretId="LLM-Platform-Secrets")
        client = session.client(service_name='secretsmanager', region_name=region_name)
        get_secret_value_response = client.get_secret_value(SecretId="LLM-Platform-Secrets")
        cred_details = json.loads(get_secret_value_response['SecretString'])
        # print("Cred Details: ", cred_details)
        # print("open ai: ", cred_details["OPENAI_API_KEY"])
        return cred_details
        # if "SecretString" in response:
        #     return response["SecretString"]
        # else:
        #     raise ValueError("SecretBinary is not supported in this implementation.")
    except Exception as e:
        print(traceback.format_exc())
        print(f"Error retrieving secret LLM-Platform-Secrets: {e}")
        raise