"""
FastAPI application with recommendation endpoints.
"""

import logging

from fastapi import FastAPI, HTTPException, Path, Query

# Import data models
from models.data_models import ModelStatusResponse, RecommendationResponse
from models.model_store import model_store
from models.recommendation_utils import get_cf_recommendations, get_content_recommendations, get_hybrid_recommendations, get_popular_recommendations

# Initialize FastAPI app
app = FastAPI(title="Article Recommendation API", description="API for generating article recommendations using various strategies.", version="1.0.0")


@app.get("/", tags=["General"])
async def root():
    """Root endpoint"""
    status = "healthy" if model_store.is_ready() else "unhealthy"
    return {"message": "Article Recommendation API is running", "status": status, "available_models": model_store.get_status()}


@app.get("/health", tags=["General"])
async def health():
    """Health check endpoint"""
    status = "healthy" if model_store.is_ready() else "unhealthy"

    # Get detailed model information
    model_info = model_store.get_status()

    # Ready status for different recommendation methods
    ready_for = {
        "collaborative": model_store.is_ready(),
        "content": model_store.is_content_ready(),
        "hybrid": model_store.is_hybrid_ready(),
        "popular": model_info["article_popularity"],
    }

    # Determine message
    if status == "healthy":
        if all(ready_for.values()):
            message = "All recommendation methods are available"
        else:
            unavailable = [method for method, ready in ready_for.items() if not ready]
            message = f"Some recommendation methods are unavailable: {', '.join(unavailable)}"
    else:
        message = "Service is unhealthy - collaborative filtering model not loaded"

    return ModelStatusResponse(status=status, models_loaded=model_info, ready_for=ready_for, message=message)


@app.get("/recommendations/{user_id}", response_model=RecommendationResponse, tags=["Recommendations"])
async def get_recommendations(
    user_id: int = Path(..., description="User ID", ge=1),
    n: int = Query(5, description="Number of recommendations", ge=1, le=20),
    method: str = Query("hybrid", description="Recommendation method", enum=["collaborative", "content", "hybrid", "popular"]),
):
    """
    Generate article recommendations for a user using the specified method

    - **user_id**: The ID of the user to get recommendations for
    - **n**: Number of recommendations to return (1-20)
    - **method**: Recommendation strategy to use:
      - **collaborative**: Uses collaborative filtering with the SVD model
      - **content**: Uses content-based filtering using article embeddings
      - **hybrid**: Combines collaborative and content-based methods (default)
      - **popular**: Returns the most popular articles (fallback for new users)
    """
    # Check if required models are available for the method
    if method == "collaborative" and not model_store.is_ready():
        method = "popular"  # Fallback to popular
        logging.warning(f"Collaborative filtering not available, falling back to popular for user {user_id}")
    elif method == "content" and not model_store.is_content_ready():
        method = "popular"  # Fallback to popular
        logging.warning(f"Content-based recommendations not available, falling back to popular for user {user_id}")
    elif method == "hybrid" and not model_store.is_hybrid_ready():
        # Fallback to whichever method is available
        if model_store.is_ready():
            method = "collaborative"
            logging.warning(f"Hybrid recommendations not available, falling back to collaborative for user {user_id}")
        elif model_store.is_content_ready():
            method = "content"
            logging.warning(f"Hybrid recommendations not available, falling back to content for user {user_id}")
        else:
            method = "popular"
            logging.warning(f"Hybrid recommendations not available, falling back to popular for user {user_id}")

    try:
        recommendations = []

        # Generate recommendations using selected method
        if method == "collaborative":
            recommendations = get_cf_recommendations(user_id, n)
        elif method == "content":
            recommendations = get_content_recommendations(user_id, n)
        elif method == "hybrid":
            recommendations = get_hybrid_recommendations(user_id, n)
        else:  # popular
            recommendations = get_popular_recommendations(n)

        # If we got no recommendations, fall back to popular
        if not recommendations and method != "popular":
            logging.warning(f"No {method} recommendations found for user {user_id}, falling back to popular")
            recommendations = get_popular_recommendations(n)
            method = "popular_fallback"

        # Build response
        return RecommendationResponse(
            user_id=user_id,
            recommendations=[item_id for item_id, _ in recommendations],
            score=recommendations[0][1] if recommendations else None,
            method=method,
        )

    except Exception as e:
        logging.exception(f"Error generating recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating recommendations: {str(e)}")


# Make the app runnable directly
if __name__ == "__main__":
    import os

    import uvicorn

    # Get port from environment variable or use default
    port = int(os.environ.get("PORT", 8000))
    log_level = os.environ.get("LOG_LEVEL", "info").lower()

    uvicorn.run(app, host="0.0.0.0", port=port, log_level=log_level)
