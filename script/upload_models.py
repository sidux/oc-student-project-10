import argparse
import os

from azure.storage.blob import BlobServiceClient


def upload_models_to_blob(connection_string, container_name, model_path="project-10/models"):
    """Upload models to Azure Blob Storage"""
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    container_client = blob_service_client.get_container_client(container_name)

    # Create container if it doesn't exist
    if not container_client.exists():
        print(f"Creating container {container_name}...")
        container_client.create_container()

    # Upload all files in the models directory
    for root, dirs, files in os.walk(model_path):
        for file in files:
            file_path = os.path.join(root, file)

            # Get relative path from the model_path
            relative_path = os.path.relpath(file_path, model_path)
            blob_path = relative_path.replace("\\", "/")  # Handle Windows paths

            print(f"Uploading {file_path} to {blob_path}...")
            with open(file_path, "rb") as data:
                container_client.upload_blob(name=blob_path, data=data, overwrite=True)
            print(f"✅ Uploaded {file}")

    print("All models uploaded successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload ML models to Azure Blob Storage")
    parser.add_argument("--connection-string", required=True, help="Azure Storage Connection String")
    parser.add_argument("--container", default="models", help="Blob Container Name")
    parser.add_argument("--model-path", default="project-10/models", help="Local model directory path")

    args = parser.parse_args()

    upload_models_to_blob(args.connection_string, args.container, args.model_path)
