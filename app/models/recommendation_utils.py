"""
Utility functions for recommendation algorithms.
"""

import logging
import random

from sklearn.metrics.pairwise import cosine_similarity

from models.model_store import model_store


def get_cf_recommendations(user_id: int, n: int = 5) -> list[tuple[int, float]]:
    """Generate recommendations using collaborative filtering (SVD)"""
    if not model_store.is_ready():
        logging.warning("SVD model not loaded, cannot generate CF recommendations")
        return []

    try:
        # Get user and item IDs known to the model
        trainset = model_store.svd_model.trainset

        # Check if user is in the model
        try:
            inner_uid = trainset.to_inner_uid(user_id)
        except ValueError:
            # User not in model
            logging.warning(f"User {user_id} not found in SVD model")
            return []

        # Find all items the user hasn't rated
        all_items = trainset._raw2inner_id_items.keys()

        # Get the user's rated items (inner IDs)
        user_seen_items = set()
        for rating_tuple in trainset.ur[inner_uid]:
            # The structure might vary based on surprise version
            if isinstance(rating_tuple, tuple):
                user_seen_items.add(rating_tuple[0])  # Add the item ID
            else:
                user_seen_items.add(rating_tuple)

        # Convert inner item IDs to raw IDs
        user_seen_items_raw = [trainset.to_raw_iid(iid) for iid in user_seen_items]

        # Items the user hasn't rated yet
        candidate_items = [item for item in all_items if item not in user_seen_items_raw]

        # Limit candidates for efficiency
        if len(candidate_items) > 100:
            candidate_items = random.sample(candidate_items, 100)

        # Generate predictions
        predictions = []
        for item_id in candidate_items:
            try:
                pred = model_store.svd_model.predict(user_id, item_id)
                predictions.append((item_id, pred.est))
            except Exception as e:
                logging.warning(f"Error predicting for user {user_id}, item {item_id}: {str(e)}")
                continue

        # Sort by predicted rating and take top N
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:n]

    except Exception as e:
        logging.error(f"Error generating CF recommendations for user {user_id}: {str(e)}")
        return []


def get_similar_articles(article_id: int, n: int = 5) -> list[tuple[int, float]]:
    """Find similar articles using embeddings"""
    if not model_store.is_content_ready():
        logging.warning("Article embeddings not loaded, cannot find similar articles")
        return []

    try:
        # Check if article exists in our embeddings
        article_idx = model_store.article_id_to_idx.get(article_id)
        if article_idx is None:
            logging.warning(f"Article ID {article_id} not found in embeddings")
            return []

        # Get the article embedding
        article_embedding = model_store.article_embeddings[article_idx].reshape(1, -1)

        # Calculate similarity to all other articles
        similarities = cosine_similarity(article_embedding, model_store.article_embeddings)[0]

        # Create (article_id, similarity) pairs
        article_similarities = []
        for idx, sim in enumerate(similarities):
            if idx != article_idx:  # Exclude the input article itself
                other_article_id = model_store.idx_to_article_id.get(idx)
                if other_article_id is not None:
                    article_similarities.append((other_article_id, float(sim)))

        # Sort by similarity (descending) and return top N
        article_similarities.sort(key=lambda x: x[1], reverse=True)
        return article_similarities[:n]

    except Exception as e:
        logging.error(f"Error finding similar articles for article {article_id}: {str(e)}")
        return []


def get_user_history(user_id: int, limit: int = 20) -> list[tuple[int, float | None]]:
    """Get a user's rating history"""
    if model_store.user_article_ratings is None:
        logging.warning("User article ratings not loaded, cannot get user history")
        return []

    try:
        # Get all ratings for this user
        user_data = model_store.user_article_ratings[model_store.user_article_ratings["user_id"] == user_id]

        if user_data.empty:
            return []

        # Check if rating column exists
        if "rating" in user_data.columns:
            history = [(row["click_article_id"], row["rating"]) for _, row in user_data.iterrows()]
        else:
            # If no explicit ratings, just return the articles
            history = [(row["click_article_id"], None) for _, row in user_data.iterrows()]

        # Limit the number of returned items
        return history[:limit]

    except Exception as e:
        logging.error(f"Error getting history for user {user_id}: {str(e)}")
        return []


def get_popular_recommendations(n: int = 5) -> list[tuple[int, float | None]]:
    """Get the most popular articles"""
    if model_store.article_popularity is None:
        logging.warning("Article popularity not loaded, cannot get popular recommendations")
        return []

    try:
        # Get top N most popular articles
        popular = model_store.article_popularity.head(n)

        # Check if rating/popularity score column exists
        if "rating" in popular.columns:
            return [(row["click_article_id"], row["rating"]) for _, row in popular.iterrows()]
        else:
            return [(row["click_article_id"], None) for _, row in popular.iterrows()]

    except Exception as e:
        logging.error(f"Error getting popular recommendations: {str(e)}")
        return []


def get_content_recommendations(user_id: int, n: int = 5) -> list[tuple[int, float]]:
    """Generate content-based recommendations for a user"""
    if not model_store.is_content_ready() or model_store.user_article_ratings is None:
        logging.warning("Required data not loaded for content recommendations")
        return []

    try:
        # Get user's ratings
        user_data = model_store.user_article_ratings[model_store.user_article_ratings["user_id"] == user_id]

        if user_data.empty:
            logging.info(f"No ratings found for user {user_id}")
            return []

        # Get the user's most recent article
        recent_article_id = user_data["click_article_id"].values[-1]

        # Find similar articles to the most recent one
        similar_articles = get_similar_articles(recent_article_id, n)

        return similar_articles

    except Exception as e:
        logging.error(f"Error generating content recommendations for user {user_id}: {str(e)}")
        return []


def get_hybrid_recommendations(user_id: int, n: int = 5) -> list[tuple[int, float]]:
    """Generate hybrid recommendations combining CF and content-based approaches"""
    if not model_store.is_hybrid_ready():
        logging.warning("Required models not loaded for hybrid recommendations")
        return []

    try:
        # Get recommendations from both methods
        cf_recs = get_cf_recommendations(user_id, n=n * 2)  # Get more to allow for overlap
        content_recs = get_content_recommendations(user_id, n=n * 2)

        # Handle case where one method returns no results
        if not cf_recs:
            return content_recs[:n]
        if not content_recs:
            return cf_recs[:n]

        # Combine recommendations with a weighted approach
        seen_ids = set()
        hybrid_recs = []

        # Add CF recommendations with higher weight
        cf_weight = 0.7
        for article_id, score in cf_recs:
            if article_id not in seen_ids:
                hybrid_recs.append((article_id, score * cf_weight))
                seen_ids.add(article_id)

        # Add content recommendations with lower weight
        content_weight = 0.3
        for article_id, score in content_recs:
            if article_id not in seen_ids:
                hybrid_recs.append((article_id, score * content_weight))
                seen_ids.add(article_id)

        # Sort by weighted score
        hybrid_recs.sort(key=lambda x: x[1], reverse=True)

        # Return top N
        return hybrid_recs[:n]

    except Exception as e:
        logging.error(f"Error generating hybrid recommendations for user {user_id}: {str(e)}")
        # Fall back to CF recommendations
        return get_cf_recommendations(user_id, n)


def get_article_metadata(article_id: int) -> dict | None:
    """Get metadata for a specific article"""
    if model_store.articles_metadata is None:
        logging.warning("Article metadata not loaded")
        return None

    try:
        # Find the article in the metadata
        article_data = model_store.articles_metadata[model_store.articles_metadata["article_id"] == article_id]

        if article_data.empty:
            return None

        # Convert to dictionary
        return article_data.iloc[0].to_dict()

    except Exception as e:
        logging.error(f"Error getting metadata for article {article_id}: {str(e)}")
        return None
