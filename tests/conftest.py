"""
Shared fixtures for recommendation system tests.
"""

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def mock_svd_model():
    """Mock SVD model with trainset and predict."""
    model = MagicMock()

    trainset = MagicMock()

    # Known users/items
    def _to_inner_uid(uid):
        mapping = {1: 0, 2: 1}
        if uid not in mapping:
            raise ValueError(f"User {uid} not found")
        return mapping[uid]

    trainset.to_inner_uid.side_effect = _to_inner_uid
    trainset._raw2inner_id_items = {100: 0, 101: 1, 102: 2, 103: 3, 104: 4}
    # User 1 has rated item 100 (inner id 0)
    trainset.ur = {0: [(0, 4.0)], 1: []}
    trainset.to_raw_iid.side_effect = lambda iid: {0: 100, 1: 101, 2: 102, 3: 103, 4: 104}[iid]

    model.trainset = trainset

    # Predict returns a named-tuple-like object with .est
    def mock_predict(user_id, item_id):
        pred = MagicMock()
        pred.est = 3.5 + (item_id % 10) * 0.1
        return pred

    model.predict.side_effect = mock_predict

    return model


@pytest.fixture
def mock_article_embeddings():
    """Real numpy array with 5 articles, 3-dimensional embeddings."""
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.5, 0.5, 0.0],
        ]
    )


@pytest.fixture
def mock_article_id_mapping():
    """Mapping between article IDs and embedding indices."""
    id_to_idx = {100: 0, 101: 1, 102: 2, 103: 3, 104: 4}
    idx_to_id = {v: k for k, v in id_to_idx.items()}
    return id_to_idx, idx_to_id


@pytest.fixture
def mock_article_popularity():
    """DataFrame of popular articles."""
    return pd.DataFrame(
        {
            "click_article_id": [100, 101, 102, 103, 104],
            "rating": [5.0, 4.5, 4.0, 3.5, 3.0],
        }
    )


@pytest.fixture
def mock_user_article_ratings():
    """DataFrame of user-article interactions."""
    return pd.DataFrame(
        {
            "user_id": [1, 1, 1, 2, 2],
            "click_article_id": [100, 101, 102, 103, 104],
            "rating": [5.0, 4.0, 3.0, 4.5, 3.5],
        }
    )


@pytest.fixture
def mock_articles_metadata():
    """DataFrame of article metadata."""
    return pd.DataFrame(
        {
            "article_id": [100, 101, 102, 103, 104],
            "title": ["Article A", "Article B", "Article C", "Article D", "Article E"],
            "category": ["tech", "tech", "science", "sports", "tech"],
        }
    )
