from pathlib import Path
from typing import List, Dict, Any, Tuple
import re
import ast
import json
from concurrent.futures import ThreadPoolExecutor
import requests
import time

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from loguru import logger
from tqdm import tqdm
import typer

from zomato_prediction.config import PROCESSED_DATA_DIR

app = typer.Typer()


class ZomatoFeatureEngineer:
    """
    Advanced feature engineering pipeline that converts processed data into ML-ready features.
    Implements all sophisticated feature engineering from notebooks 04-05 (Geolocation & NLP).
    """

    def __init__(self):
        """Initialize feature engineer with required components."""
        self.sia = None  # Sentiment analyzer
        self.stop_words = None
        self.tfidf_vectorizer = None
        self.cuisine_encodings = {}
        self.location_stats = {}
        self.scaler = StandardScaler()

        # Initialize NLTK components
        self._setup_nltk()

        logger.info("✅ ZomatoFeatureEngineer initialized")

    def _setup_nltk(self):
        """Setup NLTK components for text processing."""
        try:
            # Download required NLTK data
            nltk.download("vader_lexicon", quiet=True)
            nltk.download("stopwords", quiet=True)
            nltk.download("punkt", quiet=True)

            # Initialize sentiment analyzer
            self.sia = SentimentIntensityAnalyzer()
            self.stop_words = set(stopwords.words("english"))

            logger.info("✅ NLTK components initialized")
        except Exception as e:
            logger.warning(f"NLTK setup warning: {e}")

    def extract_geolocation_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract sophisticated geolocation features from notebook 04.
        Creates location-based statistical features, clustering, and popularity metrics.
        """
        logger.info("🗺️ Extracting geolocation features...")
        df_geo = df.copy()

        # Feature 1: Location popularity (restaurant density)
        location_counts = df_geo["location"].value_counts()
        df_geo["location_restaurant_count"] = df_geo["location"].map(location_counts)

        # Feature 2: Location average rating
        location_avg_rating = df_geo.groupby("location")["rate"].mean()
        df_geo["location_avg_rating"] = df_geo["location"].map(location_avg_rating)

        # Feature 3: Location average cost
        location_avg_cost = df_geo.groupby("location")["cost_for_two"].mean()
        df_geo["location_avg_cost"] = df_geo["location"].map(location_avg_cost)

        # Feature 4: Location price category (budget/mid-range/premium)
        cost_q33 = df_geo["cost_for_two"].quantile(0.33)
        cost_q66 = df_geo["cost_for_two"].quantile(0.66)

        def categorize_location_price(avg_cost):
            if pd.isna(avg_cost):
                return "unknown"
            elif avg_cost <= cost_q33:
                return "budget_area"
            elif avg_cost <= cost_q66:
                return "mid_range_area"
            else:
                return "premium_area"

        df_geo["location_price_category"] = df_geo["location_avg_cost"].apply(
            categorize_location_price
        )

        # Feature 5: Location competition level (based on restaurant density and rating variance)
        location_rating_std = df_geo.groupby("location")["rate"].std().fillna(0)
        df_geo["location_rating_variance"] = df_geo["location"].map(location_rating_std)

        # Create competition score: high density + high variance = high competition
        df_geo["location_competition_score"] = (
            df_geo["location_restaurant_count"]
            / df_geo["location_restaurant_count"].max()
        ) * 0.6 + (
            df_geo["location_rating_variance"]
            / df_geo["location_rating_variance"].max()
        ) * 0.4

        # Feature 6: Area category encoding (one-hot for major areas)
        top_areas = df_geo["area"].value_counts().head(10).index
        for area in top_areas:
            df_geo[f'area_is_{area.lower().replace(" ", "_").replace("-", "_")}'] = (
                df_geo["area"] == area
            ).astype(int)

        logger.success(
            f"✅ Geolocation features extracted: {7 + len(top_areas)} new features"
        )
        return df_geo

    def extract_nlp_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract sophisticated NLP features from notebook 05.
        Processes restaurant names, cuisines, dishes, and reviews for sentiment and topics.
        """
        logger.info("📝 Extracting NLP features...")
        df_nlp = df.copy()

        # Feature 1: Restaurant name analysis
        df_nlp = self._analyze_restaurant_names(df_nlp)

        # Feature 2: Cuisine diversity and popularity
        df_nlp = self._analyze_cuisines(df_nlp)

        # Feature 3: Menu/dish analysis
        df_nlp = self._analyze_dishes(df_nlp)

        # Feature 4: Review sentiment analysis (if reviews_list exists)
        if "reviews_list" in df_nlp.columns:
            df_nlp = self._analyze_review_sentiment(df_nlp)

        # Feature 5: Text complexity features
        df_nlp = self._extract_text_complexity_features(df_nlp)

        logger.success("✅ NLP features extracted")
        return df_nlp

    def _analyze_restaurant_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Analyze restaurant names for patterns and popularity."""
        logger.info("🏪 Analyzing restaurant names...")

        # Chain detection (restaurants with same name)
        name_counts = df["name"].value_counts()
        df["is_chain"] = (df["name"].map(name_counts) > 1).astype(int)
        df["chain_size"] = df["name"].map(name_counts)

        # Name characteristics
        df["name_length"] = df["name"].astype(str).str.len()
        df["name_word_count"] = df["name"].astype(str).str.split().str.len()

        # Common keywords in names
        food_keywords = [
            "restaurant",
            "cafe",
            "bar",
            "grill",
            "kitchen",
            "food",
            "corner",
            "palace",
            "garden",
        ]
        for keyword in food_keywords:
            df[f"name_has_{keyword}"] = (
                df["name"]
                .astype(str)
                .str.lower()
                .str.contains(keyword, na=False)
                .astype(int)
            )

        return df

    def _analyze_cuisines(self, df: pd.DataFrame) -> pd.DataFrame:
        """Analyze cuisine patterns and create cuisine-based features."""
        logger.info("🍽️ Analyzing cuisines...")

        # Parse cuisines (they're comma-separated)
        df["cuisines_list"] = (
            df["cuisines"]
            .astype(str)
            .apply(lambda x: [c.strip() for c in x.split(",") if c.strip()])
        )
        df["cuisine_count"] = df["cuisines_list"].apply(len)

        # Most popular cuisines
        all_cuisines = []
        for cuisines in df["cuisines_list"]:
            all_cuisines.extend(cuisines)

        cuisine_popularity = pd.Series(all_cuisines).value_counts()
        top_cuisines = cuisine_popularity.head(15).index

        # Create binary features for top cuisines
        for cuisine in top_cuisines:
            df[f'serves_{cuisine.lower().replace(" ", "_").replace("-", "_")}'] = df[
                "cuisines_list"
            ].apply(lambda x: int(cuisine in x))

        # Cuisine diversity score (based on how common the combination is)
        cuisine_combo_counts = df["cuisines"].value_counts()
        df["cuisine_combo_rarity"] = df["cuisines"].map(cuisine_combo_counts)
        df["cuisine_combo_rarity_normalized"] = 1 / (
            df["cuisine_combo_rarity"] + 1
        )  # Rare combinations get higher scores

        return df

    def _analyze_dishes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Analyze dish_liked column for popular dishes and food preferences."""
        logger.info("🍛 Analyzing popular dishes...")

        if "dish_liked" not in df.columns:
            logger.warning("dish_liked column not found, skipping dish analysis")
            return df

        # Handle missing dish_liked data
        df["has_dish_info"] = (~df["dish_liked"].isna()).astype(int)
        df["dish_liked"] = df["dish_liked"].fillna("")

        # Count dishes mentioned
        df["dish_count"] = (
            df["dish_liked"].astype(str).apply(lambda x: len(x.split(",")) if x else 0)
        )

        # Popular dish keywords
        dish_keywords = [
            "biryani",
            "pizza",
            "burger",
            "pasta",
            "noodles",
            "chicken",
            "paneer",
            "fish",
            "mutton",
            "dessert",
        ]
        for keyword in dish_keywords:
            df[f"serves_{keyword}"] = (
                df["dish_liked"]
                .astype(str)
                .str.lower()
                .str.contains(keyword, na=False)
                .astype(int)
            )

        return df

    def _analyze_review_sentiment(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract sentiment features from review text."""
        logger.info("💭 Analyzing review sentiment...")

        def extract_review_sentiment(reviews_str):
            """Extract sentiment from reviews_list string."""
            if pd.isna(reviews_str) or reviews_str == "[]":
                return {"compound": 0, "pos": 0, "neu": 1, "neg": 0, "review_count": 0}

            try:
                # Parse the reviews list (it's stored as string representation of list)
                reviews = ast.literal_eval(reviews_str)

                if not reviews:
                    return {
                        "compound": 0,
                        "pos": 0,
                        "neu": 1,
                        "neg": 0,
                        "review_count": 0,
                    }

                sentiments = []
                for review in reviews:
                    if isinstance(review, (list, tuple)) and len(review) >= 2:
                        # Review format: [rating, text]
                        review_text = str(review[1])
                        if self.sia and review_text:
                            sentiment = self.sia.polarity_scores(review_text)
                            sentiments.append(sentiment)

                if sentiments:
                    # Average sentiment across all reviews
                    avg_sentiment = {
                        "compound": np.mean([s["compound"] for s in sentiments]),
                        "pos": np.mean([s["pos"] for s in sentiments]),
                        "neu": np.mean([s["neu"] for s in sentiments]),
                        "neg": np.mean([s["neg"] for s in sentiments]),
                        "review_count": len(sentiments),
                    }
                    return avg_sentiment
                else:
                    return {
                        "compound": 0,
                        "pos": 0,
                        "neu": 1,
                        "neg": 0,
                        "review_count": 0,
                    }

            except Exception as e:
                logger.debug(f"Error parsing reviews: {e}")
                return {"compound": 0, "pos": 0, "neu": 1, "neg": 0, "review_count": 0}

        # Apply sentiment analysis
        tqdm.pandas(desc="Analyzing sentiment")
        sentiment_data = df["reviews_list"].progress_apply(extract_review_sentiment)

        # Extract sentiment features
        df["review_sentiment_compound"] = [s["compound"] for s in sentiment_data]
        df["review_sentiment_positive"] = [s["pos"] for s in sentiment_data]
        df["review_sentiment_neutral"] = [s["neu"] for s in sentiment_data]
        df["review_sentiment_negative"] = [s["neg"] for s in sentiment_data]
        df["review_count"] = [s["review_count"] for s in sentiment_data]

        return df

    def _extract_text_complexity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract text complexity and descriptiveness features."""
        logger.info("📊 Extracting text complexity features...")

        # Combine all text fields for analysis
        text_fields = ["name", "rest_type", "cuisines", "address"]

        def calculate_text_metrics(row):
            combined_text = " ".join([str(row.get(field, "")) for field in text_fields])

            return {
                "total_text_length": len(combined_text),
                "total_word_count": len(combined_text.split()),
                "avg_word_length": (
                    np.mean([len(word) for word in combined_text.split()])
                    if combined_text.split()
                    else 0
                ),
                "descriptiveness_score": len(combined_text.split())
                / (len(text_fields) + 1),  # Words per field
            }

        tqdm.pandas(desc="Calculating text metrics")
        text_metrics = df.progress_apply(calculate_text_metrics, axis=1)

        # Extract metrics
        df["total_text_length"] = [m["total_text_length"] for m in text_metrics]
        df["total_word_count"] = [m["total_word_count"] for m in text_metrics]
        df["avg_word_length"] = [m["avg_word_length"] for m in text_metrics]
        df["descriptiveness_score"] = [m["descriptiveness_score"] for m in text_metrics]

        return df

    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create sophisticated interaction features between different data aspects.
        """
        logger.info("🔗 Creating interaction features...")
        df_inter = df.copy()

        # Cost vs Location interaction
        df_inter["cost_vs_location_avg"] = df_inter["cost_for_two"] / (
            df_inter["location_avg_cost"] + 1
        )
        df_inter["premium_in_budget_area"] = (
            (df_inter["cost_for_two"] > df_inter["cost_for_two"].quantile(0.7))
            & (df_inter["location_price_category"] == "budget_area")
        ).astype(int)

        # Service features interaction
        df_inter["full_service"] = (
            df_inter["online_order"] & df_inter["book_table"]
        ).astype(int)
        df_inter["convenience_score"] = (
            df_inter["online_order"] + df_inter["book_table"]
        )

        # Popularity vs Quality interaction
        df_inter["votes_per_year"] = df_inter["votes"] / (
            2023 - 2010 + 1
        )  # Assuming dataset from 2010-2023
        df_inter["rating_votes_interaction"] = df_inter["rate"] * np.log1p(
            df_inter["votes"]
        )

        # Competition vs Performance
        df_inter["outperforms_location"] = (
            df_inter["rate"] > df_inter["location_avg_rating"]
        ).astype(int)
        df_inter["competitive_advantage"] = (
            df_inter["rate"] - df_inter["location_avg_rating"]
        )

        logger.success("✅ Interaction features created")
        return df_inter

    def process_full_feature_pipeline(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Execute the complete feature engineering pipeline.
        This transforms processed data into ML-ready features with 90+ features.
        """
        logger.info("🚀 Starting full feature engineering pipeline...")

        # Step 1: Extract geolocation features
        df_features = self.extract_geolocation_features(df)

        # Step 2: Extract NLP features
        df_features = self.extract_nlp_features(df_features)

        # Step 3: Create interaction features
        df_features = self.create_interaction_features(df_features)

        # Step 4: Handle categorical encoding for remaining categoricals
        df_features = self._encode_categorical_features(df_features)

        # Step 5: Feature scaling and normalization
        df_features = self._scale_numerical_features(df_features)

        logger.success(
            f"🎉 Feature engineering completed! Final shape: {df_features.shape}"
        )
        return df_features

    def _encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode remaining categorical features using appropriate strategies."""
        logger.info("🏷️ Encoding categorical features...")

        # Target encoding for high-cardinality categoricals (location, rest_type)
        high_card_cols = ["location", "rest_type"]

        for col in high_card_cols:
            if col in df.columns:
                # Use rate as target for encoding
                target_mean = df.groupby(col)["rate"].mean()
                df[f"{col}_target_encoded"] = df[col].map(target_mean)
                df[f"{col}_target_encoded"] = df[f"{col}_target_encoded"].fillna(
                    df["rate"].mean()
                )

        # Label encoding for medium cardinality
        le = LabelEncoder()
        label_encode_cols = ["area", "category", "location_price_category"]

        for col in label_encode_cols:
            if col in df.columns:
                df[f"{col}_encoded"] = le.fit_transform(df[col].astype(str))

        return df

    def _scale_numerical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Scale numerical features appropriately."""
        logger.info("📏 Scaling numerical features...")

        # Features that should be scaled
        scale_features = [
            "cost_for_two",
            "votes",
            "location_restaurant_count",
            "location_avg_cost",
            "location_competition_score",
            "chain_size",
            "name_length",
            "cuisine_count",
            "dish_count",
            "total_text_length",
            "total_word_count",
            "descriptiveness_score",
            "votes_per_year",
            "rating_votes_interaction",
        ]

        existing_scale_features = [col for col in scale_features if col in df.columns]

        if existing_scale_features:
            df[existing_scale_features] = self.scaler.fit_transform(
                df[existing_scale_features]
            )
            logger.info(f"Scaled {len(existing_scale_features)} numerical features")

        return df


@app.command()
def main(
    input_path: Path = PROCESSED_DATA_DIR / "zomato_master_processed.parquet",
    output_path: Path = PROCESSED_DATA_DIR / "zomato_features_final.parquet",
):
    """
    Main function to engineer features from processed Zomato data.
    Creates 90+ sophisticated ML-ready features from notebooks 04-05.
    """
    logger.info(f"🔄 Engineering features from {input_path} to {output_path}")

    # Initialize feature engineer
    feature_engineer = ZomatoFeatureEngineer()

    # Load processed data
    logger.info("📖 Loading processed data...")
    df_processed = pd.read_parquet(input_path)
    logger.info(f"Processed data loaded: {df_processed.shape}")

    # Engineer features
    df_features = feature_engineer.process_full_feature_pipeline(df_processed)

    # Save featured data
    logger.info(f"💾 Saving feature-engineered data to {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_features.to_parquet(output_path, index=False)

    # Generate feature summary
    logger.info("📊 Feature Engineering Summary:")
    logger.info(f"  • Input features: {df_processed.shape[1]}")
    logger.info(f"  • Output features: {df_features.shape[1]}")
    logger.info(
        f"  • New features created: {df_features.shape[1] - df_processed.shape[1]}"
    )
    logger.info(f"  • Total records: {len(df_features):,}")

    # List feature categories
    geo_features = [
        col for col in df_features.columns if "location" in col or "area" in col
    ]
    nlp_features = [
        col
        for col in df_features.columns
        if any(x in col for x in ["name", "cuisine", "dish", "sentiment", "text"])
    ]
    interaction_features = [
        col
        for col in df_features.columns
        if any(x in col for x in ["vs", "interaction", "advantage", "score"])
    ]

    logger.info(f"  • Geolocation features: {len(geo_features)}")
    logger.info(f"  • NLP features: {len(nlp_features)}")
    logger.info(f"  • Interaction features: {len(interaction_features)}")

    logger.success("🎉 Feature engineering complete!")


if __name__ == "__main__":
    app()
