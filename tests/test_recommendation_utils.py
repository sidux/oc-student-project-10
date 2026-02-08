"""
Tests for recommendation utility functions.
"""

from unittest.mock import patch

from models.recommendation_utils import (
    get_article_metadata,
    get_cf_recommendations,
    get_content_recommendations,
    get_hybrid_recommendations,
    get_popular_recommendations,
    get_similar_articles,
    get_user_history,
)


# ---------------------------------------------------------------------------
# get_cf_recommendations
# ---------------------------------------------------------------------------
class TestGetCfRecommendations:
    def test_returns_empty_when_model_not_loaded(self):
        with patch("models.recommendation_utils.model_store") as ms:
            ms.is_ready.return_value = False
            assert get_cf_recommendations(1) == []

    def test_returns_empty_for_unknown_user(self, mock_svd_model):
        with patch("models.recommendation_utils.model_store") as ms:
            ms.is_ready.return_value = True
            ms.svd_model = mock_svd_model
            result = get_cf_recommendations(999)
            assert result == []

    def test_returns_recommendations_for_known_user(self, mock_svd_model):
        with patch("models.recommendation_utils.model_store") as ms:
            ms.is_ready.return_value = True
            ms.svd_model = mock_svd_model
            result = get_cf_recommendations(1, n=3)
            assert len(result) > 0
            # Each item is (article_id, score)
            for article_id, score in result:
                assert isinstance(score, float)

    def test_excludes_already_rated_items(self, mock_svd_model):
        with patch("models.recommendation_utils.model_store") as ms:
            ms.is_ready.return_value = True
            ms.svd_model = mock_svd_model
            result = get_cf_recommendations(1, n=10)
            article_ids = [aid for aid, _ in result]
            # User 1 rated item 100, so it should be excluded
            assert 100 not in article_ids


# ---------------------------------------------------------------------------
# get_similar_articles
# ---------------------------------------------------------------------------
class TestGetSimilarArticles:
    def test_returns_empty_when_not_content_ready(self):
        with patch("models.recommendation_utils.model_store") as ms:
            ms.is_content_ready.return_value = False
            assert get_similar_articles(100) == []

    def test_returns_empty_for_unknown_article(self, mock_article_embeddings, mock_article_id_mapping):
        with patch("models.recommendation_utils.model_store") as ms:
            ms.is_content_ready.return_value = True
            ms.article_embeddings = mock_article_embeddings
            ms.article_id_to_idx = mock_article_id_mapping[0]
            ms.idx_to_article_id = mock_article_id_mapping[1]
            assert get_similar_articles(999) == []

    def test_returns_similar_articles(self, mock_article_embeddings, mock_article_id_mapping):
        with patch("models.recommendation_utils.model_store") as ms:
            ms.is_content_ready.return_value = True
            ms.article_embeddings = mock_article_embeddings
            ms.article_id_to_idx = mock_article_id_mapping[0]
            ms.idx_to_article_id = mock_article_id_mapping[1]
            result = get_similar_articles(100, n=2)
            assert len(result) == 2
            # Article 100 is [1,0,0], so article 101 [0.9,0.1,0] should be most similar
            assert result[0][0] == 101

    def test_excludes_self(self, mock_article_embeddings, mock_article_id_mapping):
        with patch("models.recommendation_utils.model_store") as ms:
            ms.is_content_ready.return_value = True
            ms.article_embeddings = mock_article_embeddings
            ms.article_id_to_idx = mock_article_id_mapping[0]
            ms.idx_to_article_id = mock_article_id_mapping[1]
            result = get_similar_articles(100, n=10)
            article_ids = [aid for aid, _ in result]
            assert 100 not in article_ids


# ---------------------------------------------------------------------------
# get_user_history
# ---------------------------------------------------------------------------
class TestGetUserHistory:
    def test_returns_empty_when_ratings_not_loaded(self):
        with patch("models.recommendation_utils.model_store") as ms:
            ms.user_article_ratings = None
            assert get_user_history(1) == []

    def test_returns_empty_for_unknown_user(self, mock_user_article_ratings):
        with patch("models.recommendation_utils.model_store") as ms:
            ms.user_article_ratings = mock_user_article_ratings
            assert get_user_history(999) == []

    def test_returns_history_for_known_user(self, mock_user_article_ratings):
        with patch("models.recommendation_utils.model_store") as ms:
            ms.user_article_ratings = mock_user_article_ratings
            result = get_user_history(1)
            assert len(result) == 3
            article_ids = [aid for aid, _ in result]
            assert 100 in article_ids

    def test_respects_limit(self, mock_user_article_ratings):
        with patch("models.recommendation_utils.model_store") as ms:
            ms.user_article_ratings = mock_user_article_ratings
            result = get_user_history(1, limit=2)
            assert len(result) == 2


# ---------------------------------------------------------------------------
# get_popular_recommendations
# ---------------------------------------------------------------------------
class TestGetPopularRecommendations:
    def test_returns_empty_when_popularity_not_loaded(self):
        with patch("models.recommendation_utils.model_store") as ms:
            ms.article_popularity = None
            assert get_popular_recommendations() == []

    def test_returns_top_n(self, mock_article_popularity):
        with patch("models.recommendation_utils.model_store") as ms:
            ms.article_popularity = mock_article_popularity
            result = get_popular_recommendations(n=3)
            assert len(result) == 3
            # First should be most popular (rating 5.0)
            assert result[0][0] == 100
            assert result[0][1] == 5.0


# ---------------------------------------------------------------------------
# get_content_recommendations
# ---------------------------------------------------------------------------
class TestGetContentRecommendations:
    def test_returns_empty_when_not_content_ready(self):
        with patch("models.recommendation_utils.model_store") as ms:
            ms.is_content_ready.return_value = False
            ms.user_article_ratings = None
            assert get_content_recommendations(1) == []

    def test_returns_empty_for_unknown_user(self, mock_user_article_ratings, mock_article_embeddings, mock_article_id_mapping):
        with patch("models.recommendation_utils.model_store") as ms:
            ms.is_content_ready.return_value = True
            ms.user_article_ratings = mock_user_article_ratings
            ms.article_embeddings = mock_article_embeddings
            ms.article_id_to_idx = mock_article_id_mapping[0]
            ms.idx_to_article_id = mock_article_id_mapping[1]
            result = get_content_recommendations(999)
            assert result == []

    def test_returns_recommendations_for_known_user(self, mock_user_article_ratings, mock_article_embeddings, mock_article_id_mapping):
        with patch("models.recommendation_utils.model_store") as ms:
            ms.is_content_ready.return_value = True
            ms.user_article_ratings = mock_user_article_ratings
            ms.article_embeddings = mock_article_embeddings
            ms.article_id_to_idx = mock_article_id_mapping[0]
            ms.idx_to_article_id = mock_article_id_mapping[1]
            result = get_content_recommendations(1, n=2)
            assert len(result) == 2


# ---------------------------------------------------------------------------
# get_hybrid_recommendations
# ---------------------------------------------------------------------------
class TestGetHybridRecommendations:
    def test_returns_empty_when_not_hybrid_ready(self):
        with patch("models.recommendation_utils.model_store") as ms:
            ms.is_hybrid_ready.return_value = False
            assert get_hybrid_recommendations(1) == []

    def test_falls_back_to_content_when_cf_empty(self):
        with patch("models.recommendation_utils.model_store") as ms:
            ms.is_hybrid_ready.return_value = True
            with patch("models.recommendation_utils.get_cf_recommendations", return_value=[]):
                with patch("models.recommendation_utils.get_content_recommendations", return_value=[(101, 0.9), (102, 0.8)]):
                    result = get_hybrid_recommendations(1, n=2)
                    assert len(result) == 2

    def test_falls_back_to_cf_when_content_empty(self):
        with patch("models.recommendation_utils.model_store") as ms:
            ms.is_hybrid_ready.return_value = True
            with patch("models.recommendation_utils.get_cf_recommendations", return_value=[(101, 4.0), (102, 3.5)]):
                with patch("models.recommendation_utils.get_content_recommendations", return_value=[]):
                    result = get_hybrid_recommendations(1, n=2)
                    assert len(result) == 2

    def test_combines_cf_and_content(self):
        with patch("models.recommendation_utils.model_store") as ms:
            ms.is_hybrid_ready.return_value = True
            with patch("models.recommendation_utils.get_cf_recommendations", return_value=[(101, 4.0)]):
                with patch("models.recommendation_utils.get_content_recommendations", return_value=[(102, 0.9)]):
                    result = get_hybrid_recommendations(1, n=5)
                    article_ids = [aid for aid, _ in result]
                    assert 101 in article_ids
                    assert 102 in article_ids


# ---------------------------------------------------------------------------
# get_article_metadata
# ---------------------------------------------------------------------------
class TestGetArticleMetadata:
    def test_returns_none_when_metadata_not_loaded(self):
        with patch("models.recommendation_utils.model_store") as ms:
            ms.articles_metadata = None
            assert get_article_metadata(100) is None

    def test_returns_none_for_unknown_article(self, mock_articles_metadata):
        with patch("models.recommendation_utils.model_store") as ms:
            ms.articles_metadata = mock_articles_metadata
            assert get_article_metadata(999) is None

    def test_returns_metadata_dict(self, mock_articles_metadata):
        with patch("models.recommendation_utils.model_store") as ms:
            ms.articles_metadata = mock_articles_metadata
            result = get_article_metadata(100)
            assert isinstance(result, dict)
            assert result["article_id"] == 100
            assert result["title"] == "Article A"
