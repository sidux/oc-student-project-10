"""
Pydantic data models for API requests and responses.
"""

from pydantic import BaseModel


class RecommendationResponse(BaseModel):
    """Response model for recommendation endpoints"""

    user_id: int
    recommendations: list[int]
    score: float | None = None
    method: str = "collaborative"


class ModelStatusResponse(BaseModel):
    """Response model for model status endpoint"""

    status: str
    models_loaded: dict[str, bool]
    ready_for: dict[str, bool]
    message: str | None = None
