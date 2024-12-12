import boto3
import os
import stat

def download_pem_from_s3(bucket_name, s3_key, local_path):
    """
    Downloads a PEM file from S3 to the local filesystem and sets correct permissions.

    :param bucket_name: The name of the S3 bucket.
    :param s3_key: The key (path) of the PEM file in the S3 bucket.
    :param local_path: The local path where the PEM file should be saved.
    """
    aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    region_name = os.getenv("AWS_REGION", "us-east-1")

    # Create S3 client
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
        os.chmod(local_path, stat.S_IRUSR)  # User read-only permission
        print(f"Permissions for {local_path} set to 400 (read-only for owner).")
    except Exception as e:
        print(f"Error downloading or setting permissions for file: {e}")
        raise

if __name__ == "__main__":
    # Configure these based on your requirements
    BUCKET_NAME = "llm-platform-general"
    S3_KEY = "jumper.pem"  # Replace with the key of your PEM file
    LOCAL_PATH = "/app/pem/jumper.pem"  # Relative path for the PEM file

    download_pem_from_s3(BUCKET_NAME, S3_KEY, LOCAL_PATH)
