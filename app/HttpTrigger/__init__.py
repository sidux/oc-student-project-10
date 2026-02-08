import io
import logging
import os
import pickle

import azure.functions as func
import numpy as np
import pandas as pd
from azure.functions import AsgiMiddleware
from azure.storage.blob import BlobServiceClient

# Import model_store to hold our models
from models.model_store import model_store

# Track if we've attempted to load models yet
model_load_attempted = False


async def download_blob(blob_service_client, container_name, blob_name):
    """Download a blob from storage"""
    try:
        container_client = blob_service_client.get_container_client(container_name)
        blob_client = container_client.get_blob_client(blob_name)

        if not blob_client.exists():
            logging.error(f"Blob {blob_name} not found in container {container_name}")
            return None

        download_stream = blob_client.download_blob()
        blob_data = download_stream.readall()
        logging.info(f"Downloaded {len(blob_data)} bytes from {blob_name}")
        return blob_data
    except Exception as e:
        logging.error(f"Error downloading {blob_name}: {str(e)}")
        return None


async def load_models_if_needed():
    """Load all recommendation models from blob storage"""
    global model_load_attempted

    # Skip if we've already loaded or attempted to load
    if model_load_attempted:
        return model_store.is_ready()

    # Mark that we've attempted to load
    model_load_attempted = True

    if model_store.is_ready():
        logging.info("Models already loaded, skipping download")
        return True

    try:
        # Get connection info from environment variables
        connection_string = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
        container_name = os.environ.get("MODELS_CONTAINER_NAME", "models")

        logging.info(f"Connection string set: {connection_string is not None and len(connection_string) > 0}")
        logging.info(f"Container name: {container_name}")

        if not connection_string:
            logging.error("AZURE_STORAGE_CONNECTION_STRING not set")
            return False

        # Create the blob client
        logging.info("Creating BlobServiceClient...")
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        logging.info("BlobServiceClient created successfully")

        # 1. Load SVD model
        svd_blob = await download_blob(blob_service_client, container_name, "svd_model.pkl")
        if svd_blob:
            tmp_path = "/tmp/svd_model_load.pkl"
            try:
                # Write to temp file
                logging.info(f"SVD blob length: {len(svd_blob)}")
                with open(tmp_path, "wb") as f:
                    f.write(svd_blob)

                file_size = os.path.getsize(tmp_path)
                logging.info(f"SVD temp file size: {file_size}")

                # Load using surprise.dump
                from surprise import dump as surprise_dump

                loaded_data = surprise_dump.load(tmp_path)
                logging.info(f"SVD loaded data type: {type(loaded_data)}")

                # surprise.dump saves as tuple (predictions, algo)
                if isinstance(loaded_data, tuple) and len(loaded_data) >= 2:
                    algo = loaded_data[1]  # Algorithm is at index 1
                    logging.info(f"SVD Algorithm type: {type(algo)}")
                    if algo is not None:
                        model_store.svd_model = algo
                        model_store.loaded_models["svd_model"] = True
                        logging.info("SVD model loaded successfully")
                    else:
                        logging.error("SVD Algorithm in tuple is None")
                else:
                    # Maybe it's just the algorithm directly
                    if hasattr(loaded_data, "predict"):
                        model_store.svd_model = loaded_data
                        model_store.loaded_models["svd_model"] = True
                        logging.info("SVD model loaded successfully (direct)")
                    else:
                        logging.error(f"Unexpected SVD data format: {type(loaded_data)}")
            except Exception as e:
                logging.exception(f"Error loading SVD model: {str(e)}")
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

        # 2. Load article embeddings
        embeddings_blob = await download_blob(blob_service_client, container_name, "article_embeddings.pkl")
        if embeddings_blob:
            try:
                embeddings = pickle.loads(embeddings_blob)
                if isinstance(embeddings, np.ndarray):
                    model_store.article_embeddings = embeddings
                    model_store.loaded_models["article_embeddings"] = True
                    logging.info(f"Article embeddings loaded successfully: shape {embeddings.shape}")
            except Exception as e:
                logging.error(f"Error processing article embeddings: {str(e)}")

        # 3. Load article ID mapping
        mapping_blob = await download_blob(blob_service_client, container_name, "article_id_mapping.pkl")
        if mapping_blob:
            try:
                mapping = pickle.loads(mapping_blob)
                if isinstance(mapping, dict):
                    model_store.article_id_to_idx = mapping
                    model_store.idx_to_article_id = {v: k for k, v in mapping.items()}
                    model_store.loaded_models["article_id_mapping"] = True
                    logging.info(f"Article ID mapping loaded successfully: {len(mapping)} entries")
            except Exception as e:
                logging.error(f"Error processing article ID mapping: {str(e)}")

        # 4. Load article popularity
        popularity_blob = await download_blob(blob_service_client, container_name, "article_popularity.csv")
        if popularity_blob:
            try:
                popularity_data = io.BytesIO(popularity_blob)
                popularity_df = pd.read_csv(popularity_data)
                if "click_article_id" in popularity_df.columns:
                    model_store.article_popularity = popularity_df
                    model_store.loaded_models["article_popularity"] = True
                    logging.info(f"Article popularity loaded successfully: {len(popularity_df)} entries")
            except Exception as e:
                logging.error(f"Error processing article popularity: {str(e)}")

        # 5. Load user article ratings
        ratings_blob = await download_blob(blob_service_client, container_name, "user_article_ratings.csv")
        if ratings_blob:
            try:
                ratings_data = io.BytesIO(ratings_blob)
                ratings_df = pd.read_csv(ratings_data)
                if "user_id" in ratings_df.columns and "click_article_id" in ratings_df.columns:
                    model_store.user_article_ratings = ratings_df
                    model_store.loaded_models["user_article_ratings"] = True
                    logging.info(f"User article ratings loaded successfully: {len(ratings_df)} entries")
            except Exception as e:
                logging.error(f"Error processing user article ratings: {str(e)}")

        # 6. Load article metadata
        metadata_blob = await download_blob(blob_service_client, container_name, "articles_metadata.csv")
        if metadata_blob:
            try:
                metadata_data = io.BytesIO(metadata_blob)
                metadata_df = pd.read_csv(metadata_data)
                if "article_id" in metadata_df.columns:
                    model_store.articles_metadata = metadata_df
                    model_store.loaded_models["articles_metadata"] = True
                    logging.info(f"Articles metadata loaded successfully: {len(metadata_df)} entries")
            except Exception as e:
                logging.error(f"Error processing articles metadata: {str(e)}")

        # Update models_loaded flag if critical models are loaded
        model_store.models_loaded = model_store.is_ready()
        if model_store.models_loaded:
            logging.info("Critical models loaded successfully")
        else:
            logging.error("Failed to load critical models")

        return model_store.is_ready()

    except Exception as e:
        logging.exception(f"Error loading models: {str(e)}")
        return False


# Import the FastAPI app after model_store
from app import app  # noqa: E402

# Create ASGI middleware
middleware = AsgiMiddleware(app)


async def main(req: func.HttpRequest) -> func.HttpResponse:
    """
    Azure Function with FastAPI integration.
    Loads models once and reuses them for subsequent requests.
    """
    global model_load_attempted

    logging.info(f"Request received: {req.url}")
    logging.info(f"Model load attempted: {model_load_attempted}")
    logging.info(f"Model store ready: {model_store.is_ready()}")

    # Try to load models if needed (will only download once)
    try:
        result = await load_models_if_needed()
        logging.info(f"Model load result: {result}")
    except Exception as e:
        logging.exception(f"Error in load_models_if_needed: {str(e)}")
        model_load_attempted = False  # Reset to allow retry

    # Pass the request to FastAPI
    return await middleware.handle_async(req)
