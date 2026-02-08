"""
Tests for ModelStore.
"""

import pandas as pd
from models.model_store import ModelStore


class TestModelStoreInit:
    def test_initial_state(self):
        store = ModelStore()
        assert store.svd_model is None
        assert store.article_embeddings is None
        assert store.article_id_to_idx is None
        assert store.idx_to_article_id is None
        assert store.article_popularity is None
        assert store.user_article_ratings is None
        assert store.articles_metadata is None
        assert store.models_loaded is False

    def test_initial_loaded_models_all_false(self):
        store = ModelStore()
        for value in store.loaded_models.values():
            assert value is False

    def test_loaded_models_keys(self):
        store = ModelStore()
        expected_keys = {
            "svd_model",
            "article_embeddings",
            "article_id_mapping",
            "article_popularity",
            "user_article_ratings",
            "articles_metadata",
        }
        assert set(store.loaded_models.keys()) == expected_keys


class TestIsReady:
    def test_not_ready_when_no_svd(self):
        store = ModelStore()
        assert store.is_ready() is False

    def test_ready_when_svd_loaded(self, mock_svd_model):
        store = ModelStore()
        store.svd_model = mock_svd_model
        assert store.is_ready() is True


class TestIsContentReady:
    def test_not_ready_when_nothing_loaded(self):
        store = ModelStore()
        assert store.is_content_ready() is False

    def test_not_ready_with_only_embeddings(self, mock_article_embeddings):
        store = ModelStore()
        store.article_embeddings = mock_article_embeddings
        assert store.is_content_ready() is False

    def test_ready_with_embeddings_and_mapping(self, mock_article_embeddings, mock_article_id_mapping):
        store = ModelStore()
        store.article_embeddings = mock_article_embeddings
        store.article_id_to_idx = mock_article_id_mapping[0]
        assert store.is_content_ready() is True


class TestIsHybridReady:
    def test_not_ready_when_nothing_loaded(self):
        store = ModelStore()
        assert store.is_hybrid_ready() is False

    def test_not_ready_with_only_svd(self, mock_svd_model):
        store = ModelStore()
        store.svd_model = mock_svd_model
        assert store.is_hybrid_ready() is False

    def test_ready_with_svd_and_content(self, mock_svd_model, mock_article_embeddings, mock_article_id_mapping):
        store = ModelStore()
        store.svd_model = mock_svd_model
        store.article_embeddings = mock_article_embeddings
        store.article_id_to_idx = mock_article_id_mapping[0]
        assert store.is_hybrid_ready() is True


class TestGetStatus:
    def test_all_false_initially(self):
        store = ModelStore()
        status = store.get_status()
        assert all(v is False for v in status.values())

    def test_reflects_loaded_models(self, mock_svd_model, mock_article_embeddings):
        store = ModelStore()
        store.svd_model = mock_svd_model
        store.article_embeddings = mock_article_embeddings
        status = store.get_status()
        assert status["svd_model"] is True
        assert status["article_embeddings"] is True
        assert status["article_id_mapping"] is False

    def test_updates_loaded_models_dict(self):
        store = ModelStore()
        store.article_popularity = pd.DataFrame({"x": [1]})
        store.get_status()
        assert store.loaded_models["article_popularity"] is True
