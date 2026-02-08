"""
Tests for Pydantic data models.
"""

from models.data_models import ModelStatusResponse, RecommendationResponse


class TestRecommendationResponse:
    def test_create_with_all_fields(self):
        resp = RecommendationResponse(
            user_id=1,
            recommendations=[100, 101, 102],
            score=4.5,
            method="collaborative",
        )
        assert resp.user_id == 1
        assert resp.recommendations == [100, 101, 102]
        assert resp.score == 4.5
        assert resp.method == "collaborative"

    def test_default_score_is_none(self):
        resp = RecommendationResponse(user_id=1, recommendations=[])
        assert resp.score is None

    def test_default_method_is_collaborative(self):
        resp = RecommendationResponse(user_id=1, recommendations=[])
        assert resp.method == "collaborative"

    def test_empty_recommendations(self):
        resp = RecommendationResponse(user_id=1, recommendations=[])
        assert resp.recommendations == []

    def test_serialization_roundtrip(self):
        resp = RecommendationResponse(user_id=42, recommendations=[1, 2, 3], score=3.7, method="hybrid")
        data = resp.model_dump()
        restored = RecommendationResponse(**data)
        assert restored == resp


class TestModelStatusResponse:
    def test_create_with_all_fields(self):
        resp = ModelStatusResponse(
            status="healthy",
            models_loaded={"svd_model": True, "article_embeddings": False},
            ready_for={"collaborative": True, "content": False},
            message="Partial",
        )
        assert resp.status == "healthy"
        assert resp.models_loaded["svd_model"] is True
        assert resp.ready_for["content"] is False
        assert resp.message == "Partial"

    def test_default_message_is_none(self):
        resp = ModelStatusResponse(status="unhealthy", models_loaded={}, ready_for={})
        assert resp.message is None

    def test_serialization_roundtrip(self):
        resp = ModelStatusResponse(
            status="healthy",
            models_loaded={"svd": True},
            ready_for={"collaborative": True},
            message="All good",
        )
        data = resp.model_dump()
        restored = ModelStatusResponse(**data)
        assert restored == resp
