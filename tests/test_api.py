"""
Tests for FastAPI application endpoints.
"""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create a test client with mocked model_store (unhealthy)."""
    with patch("app.model_store") as mock_store:
        mock_store.is_ready.return_value = False
        mock_store.is_content_ready.return_value = False
        mock_store.is_hybrid_ready.return_value = False
        mock_store.get_status.return_value = {
            "svd_model": False,
            "article_embeddings": False,
            "article_id_mapping": False,
            "article_popularity": False,
            "user_article_ratings": False,
            "articles_metadata": False,
        }
        from app import app

        yield TestClient(app)


@pytest.fixture
def ready_client():
    """Create a test client with fully loaded models (healthy)."""
    with patch("app.model_store") as mock_store:
        mock_store.is_ready.return_value = True
        mock_store.is_content_ready.return_value = True
        mock_store.is_hybrid_ready.return_value = True
        mock_store.get_status.return_value = {
            "svd_model": True,
            "article_embeddings": True,
            "article_id_mapping": True,
            "article_popularity": True,
            "user_article_ratings": True,
            "articles_metadata": True,
        }
        from app import app

        yield TestClient(app)


# ---------------------------------------------------------------------------
# Root endpoint
# ---------------------------------------------------------------------------
class TestRootEndpoint:
    def test_root_returns_200(self, client):
        response = client.get("/")
        assert response.status_code == 200

    def test_root_contains_message(self, client):
        response = client.get("/")
        data = response.json()
        assert "message" in data
        assert "Recommendation API" in data["message"]

    def test_root_unhealthy_when_not_ready(self, client):
        response = client.get("/")
        data = response.json()
        assert data["status"] == "unhealthy"

    def test_root_healthy_when_ready(self, ready_client):
        response = ready_client.get("/")
        data = response.json()
        assert data["status"] == "healthy"


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------
class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_contains_required_fields(self, client):
        response = client.get("/health")
        data = response.json()
        assert "status" in data
        assert "models_loaded" in data
        assert "ready_for" in data
        assert "message" in data

    def test_health_unhealthy_message(self, client):
        response = client.get("/health")
        data = response.json()
        assert data["status"] == "unhealthy"
        assert "unhealthy" in data["message"].lower()

    def test_health_healthy_all_ready(self, ready_client):
        response = ready_client.get("/health")
        data = response.json()
        assert data["status"] == "healthy"
        assert data["ready_for"]["collaborative"] is True


# ---------------------------------------------------------------------------
# Recommendations endpoint
# ---------------------------------------------------------------------------
class TestRecommendationsEndpoint:
    def test_popular_fallback_when_no_models(self, client):
        with patch("app.get_popular_recommendations", return_value=[(100, 5.0), (101, 4.0)]):
            response = client.get("/recommendations/1?method=popular")
            assert response.status_code == 200
            data = response.json()
            assert data["user_id"] == 1
            assert 100 in data["recommendations"]

    def test_collaborative_falls_back_to_popular(self, client):
        with patch("app.get_popular_recommendations", return_value=[(100, 5.0)]):
            response = client.get("/recommendations/1?method=collaborative")
            assert response.status_code == 200

    def test_hybrid_method_when_ready(self, ready_client):
        with patch("app.get_hybrid_recommendations", return_value=[(100, 3.5), (101, 3.0)]):
            response = ready_client.get("/recommendations/1?method=hybrid&n=2")
            assert response.status_code == 200
            data = response.json()
            assert len(data["recommendations"]) == 2

    def test_custom_n_parameter(self, ready_client):
        with patch("app.get_cf_recommendations", return_value=[(100, 4.0), (101, 3.5), (102, 3.0)]):
            response = ready_client.get("/recommendations/1?method=collaborative&n=3")
            assert response.status_code == 200
            data = response.json()
            assert len(data["recommendations"]) == 3

    def test_invalid_user_id_returns_422(self, client):
        response = client.get("/recommendations/0")
        assert response.status_code == 422

    def test_response_includes_method(self, ready_client):
        with patch("app.get_cf_recommendations", return_value=[(100, 4.0)]):
            response = ready_client.get("/recommendations/1?method=collaborative")
            data = response.json()
            assert data["method"] == "collaborative"

    def test_empty_recommendations_falls_back_to_popular(self, ready_client):
        with patch("app.get_cf_recommendations", return_value=[]):
            with patch("app.get_popular_recommendations", return_value=[(100, 5.0)]):
                response = ready_client.get("/recommendations/1?method=collaborative")
                assert response.status_code == 200
                data = response.json()
                assert data["method"] == "popular_fallback"
