from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import pickle
import joblib
import json

import pandas as pd
import numpy as np
import mlflow
import mlflow.pyfunc
from loguru import logger
from tqdm import tqdm
import typer

from zomato_prediction.config import MODELS_DIR, PROCESSED_DATA_DIR
from zomato_prediction.dataset import ZomatoDataProcessor
from zomato_prediction.features import ZomatoFeatureEngineer

app = typer.Typer()


class ZomatoPredictionService:
    """
    Production-ready prediction service that handles the complete pipeline:
    Raw Data → Cleaning → Feature Engineering → ML Prediction → Results

    This integrates all your sophisticated preprocessing with your best trained model.
    """

    def __init__(
        self, model_path: Optional[str] = None, model_uri: Optional[str] = None
    ):
        """
        Initialize the prediction service.

        Args:
            model_path: Path to local model file (.pkl)
            model_uri: MLflow model URI (e.g., 'models:/ModelName/Production')
        """
        self.model = None
        self.data_processor = ZomatoDataProcessor()
        self.feature_engineer = ZomatoFeatureEngineer()
        self.feature_names = None
        self.model_metadata = {}

        # Load model
        if model_uri:
            self._load_mlflow_model(model_uri)
        elif model_path:
            self._load_local_model(model_path)
        else:
            logger.warning(
                "No model specified. Call load_model() before making predictions."
            )

        logger.info("✅ ZomatoPredictionService initialized")

    def _load_mlflow_model(self, model_uri: str):
        """Load model from MLflow."""
        try:
            logger.info(f"🔄 Loading MLflow model from: {model_uri}")
            self.model = mlflow.pyfunc.load_model(model_uri)

            # Try to get model metadata
            try:
                model_info = mlflow.models.get_model_info(model_uri)
                self.model_metadata = {
                    "name": (
                        model_info.name if hasattr(model_info, "name") else "Unknown"
                    ),
                    "version": (
                        model_info.version
                        if hasattr(model_info, "version")
                        else "Unknown"
                    ),
                    "uri": model_uri,
                }
            except Exception as e:
                logger.warning(f"Could not get model metadata: {e}")

            logger.success(f"✅ MLflow model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load MLflow model: {e}")
            raise

    def _load_local_model(self, model_path: str):
        """Load model from local file."""
        try:
            logger.info(f"🔄 Loading local model from: {model_path}")

            model_path = Path(model_path)
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")

            # Try different loading methods
            if model_path.suffix == ".pkl":
                with open(model_path, "rb") as f:
                    self.model = pickle.load(f)
            else:
                self.model = joblib.load(model_path)

            self.model_metadata = {
                "name": model_path.stem,
                "path": str(model_path),
                "type": "local",
            }

            logger.success(f"✅ Local model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load local model: {e}")
            raise

    def predict_single(self, restaurant_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict rating for a single restaurant.

        Args:
            restaurant_data: Dictionary with restaurant information (raw format)

        Returns:
            Dictionary with prediction results and confidence metrics
        """
        if self.model is None:
            raise ValueError("No model loaded. Please load a model first.")

        logger.info("🔮 Making single prediction...")

        try:
            # Convert to DataFrame
            df_raw = pd.DataFrame([restaurant_data])

            # Process through pipeline
            result = self._process_and_predict(df_raw)

            # Return single result
            prediction_result = {
                "predicted_rating": float(result["predictions"][0]),
                "confidence_score": (
                    float(result["confidence_scores"][0])
                    if result["confidence_scores"]
                    else None
                ),
                "input_features_count": result["feature_count"],
                "model_info": self.model_metadata,
                "processing_stats": result["processing_stats"],
            }

            logger.success("✅ Single prediction completed")
            return prediction_result

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return {"error": str(e), "predicted_rating": None, "confidence_score": None}

    def predict_batch(self, restaurants_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Predict ratings for multiple restaurants efficiently.

        Args:
            restaurants_data: List of restaurant data dictionaries

        Returns:
            Dictionary with batch prediction results
        """
        if self.model is None:
            raise ValueError("No model loaded. Please load a model first.")

        logger.info(
            f"🔮 Making batch prediction for {len(restaurants_data)} restaurants..."
        )

        try:
            # Convert to DataFrame
            df_raw = pd.DataFrame(restaurants_data)

            # Process through pipeline
            result = self._process_and_predict(df_raw)

            # Return batch results
            batch_result = {
                "predictions": [float(pred) for pred in result["predictions"]],
                "confidence_scores": (
                    [float(conf) for conf in result["confidence_scores"]]
                    if result["confidence_scores"]
                    else None
                ),
                "input_count": len(restaurants_data),
                "successful_predictions": len(result["predictions"]),
                "feature_count": result["feature_count"],
                "model_info": self.model_metadata,
                "processing_stats": result["processing_stats"],
            }

            logger.success(
                f"✅ Batch prediction completed: {len(result['predictions'])} predictions"
            )
            return batch_result

        except Exception as e:
            logger.error(f"Batch prediction failed: {e}")
            return {"error": str(e), "predictions": [], "successful_predictions": 0}

    def _process_and_predict(self, df_raw: pd.DataFrame) -> Dict[str, Any]:
        """
        Internal method to process raw data through the complete pipeline and make predictions.
        """
        processing_stats = {}

        # Step 1: Data cleaning and preprocessing
        logger.info("🧹 Processing data through cleaning pipeline...")
        df_processed = self.data_processor.process_full_pipeline(df_raw)
        processing_stats["records_after_cleaning"] = len(df_processed)

        # Step 2: Feature engineering
        logger.info("⚙️ Processing data through feature engineering pipeline...")
        df_features = self.feature_engineer.process_full_feature_pipeline(df_processed)
        processing_stats["final_feature_count"] = df_features.shape[1]

        # Step 3: Prepare features for model
        X = self._prepare_features_for_model(df_features)
        processing_stats["model_input_features"] = X.shape[1]

        # Step 4: Make predictions
        logger.info("🎯 Making predictions...")
        predictions = self.model.predict(X)

        # Step 5: Calculate confidence scores (if possible)
        confidence_scores = self._calculate_confidence_scores(X)

        return {
            "predictions": predictions,
            "confidence_scores": confidence_scores,
            "feature_count": X.shape[1],
            "processing_stats": processing_stats,
        }

    def _prepare_features_for_model(self, df_features: pd.DataFrame) -> np.ndarray:
        """
        Prepare engineered features for model input.
        Handles feature selection, missing values, and proper ordering.
        """
        logger.info("🎛️ Preparing features for model...")

        # Remove non-numeric columns that shouldn't be in model
        text_columns = [
            "name",
            "address",
            "location",
            "rest_type",
            "cuisines",
            "dish_liked",
            "reviews_list",
            "area",
            "category",
            "cuisines_list",
            "location_price_category",
        ]

        numeric_df = df_features.select_dtypes(include=[np.number])

        # Fill any remaining missing values
        numeric_df = numeric_df.fillna(0)

        # Store feature names for later reference
        self.feature_names = list(numeric_df.columns)

        logger.info(
            f"✅ Features prepared: {numeric_df.shape[1]} features for {numeric_df.shape[0]} records"
        )
        return numeric_df.values

    def _calculate_confidence_scores(self, X: np.ndarray) -> Optional[List[float]]:
        """
        Calculate prediction confidence scores if the model supports it.
        """
        try:
            # For ensemble models or models with predict_proba
            if hasattr(self.model, "predict_proba"):
                # For classification, use max probability
                probabilities = self.model.predict_proba(X)
                confidence_scores = np.max(probabilities, axis=1).tolist()
                return confidence_scores
            elif hasattr(self.model, "decision_function"):
                # For SVM-like models
                decision_scores = self.model.decision_function(X)
                # Normalize to 0-1 range
                confidence_scores = (decision_scores - decision_scores.min()) / (
                    decision_scores.max() - decision_scores.min()
                )
                return confidence_scores.tolist()
            elif hasattr(self.model, "predict") and hasattr(self.model, "estimators_"):
                # For ensemble regressors, use prediction variance as inverse confidence
                individual_predictions = np.array(
                    [estimator.predict(X) for estimator in self.model.estimators_]
                )
                prediction_variance = np.var(individual_predictions, axis=0)
                # Convert variance to confidence (lower variance = higher confidence)
                max_variance = np.max(prediction_variance)
                confidence_scores = (1 - (prediction_variance / max_variance)).tolist()
                return confidence_scores
            else:
                logger.info("Model doesn't support confidence scoring")
                return None

        except Exception as e:
            logger.warning(f"Could not calculate confidence scores: {e}")
            return None

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Get feature importance from the model if available.
        """
        try:
            if hasattr(self.model, "feature_importances_"):
                if self.feature_names:
                    importance_dict = dict(
                        zip(self.feature_names, self.model.feature_importances_)
                    )
                    # Sort by importance
                    return dict(
                        sorted(
                            importance_dict.items(), key=lambda x: x[1], reverse=True
                        )
                    )
                else:
                    return {
                        "feature_importances": self.model.feature_importances_.tolist()
                    }
            elif hasattr(self.model, "coef_"):
                if self.feature_names:
                    importance_dict = dict(
                        zip(self.feature_names, abs(self.model.coef_))
                    )
                    return dict(
                        sorted(
                            importance_dict.items(), key=lambda x: x[1], reverse=True
                        )
                    )
                else:
                    return {"coefficients": self.model.coef_.tolist()}
            else:
                logger.info("Model doesn't provide feature importance")
                return None

        except Exception as e:
            logger.warning(f"Could not get feature importance: {e}")
            return None

    def validate_input(self, restaurant_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate input data format and completeness.

        Returns:
            Dictionary with validation results and suggestions
        """
        validation_result = {
            "is_valid": True,
            "warnings": [],
            "errors": [],
            "suggestions": [],
        }

        # Required fields (minimum for meaningful prediction)
        required_fields = ["name", "location", "cuisines"]
        optional_but_important = [
            "cost_for_two",
            "rest_type",
            "online_order",
            "book_table",
        ]

        # Check required fields
        for field in required_fields:
            if (
                field not in restaurant_data
                or pd.isna(restaurant_data[field])
                or restaurant_data[field] == ""
            ):
                validation_result["errors"].append(f"Missing required field: {field}")
                validation_result["is_valid"] = False

        # Check optional but important fields
        for field in optional_but_important:
            if field not in restaurant_data or pd.isna(restaurant_data[field]):
                validation_result["warnings"].append(
                    f"Missing optional field: {field} (may affect prediction accuracy)"
                )

        # Validate data types and ranges
        if "cost_for_two" in restaurant_data:
            try:
                cost = float(str(restaurant_data["cost_for_two"]).replace(",", ""))
                if cost < 50 or cost > 10000:
                    validation_result["warnings"].append(
                        f"Cost for two seems unusual: {cost} (expected range: 50-10000)"
                    )
            except (ValueError, TypeError):
                validation_result["warnings"].append("Cost for two should be a number")

        # Provide suggestions
        if len(validation_result["warnings"]) > 0:
            validation_result["suggestions"].append(
                "Providing more complete data will improve prediction accuracy"
            )

        if validation_result["is_valid"]:
            validation_result["suggestions"].append(
                "Input data looks good for prediction"
            )

        return validation_result


@app.command()
def predict_from_file(
    input_path: Path = PROCESSED_DATA_DIR / "test_data.csv",
    model_path: Path = MODELS_DIR / "best_model.pkl",
    output_path: Path = PROCESSED_DATA_DIR / "predictions.csv",
):
    """
    Make predictions from a CSV file using the trained model.
    """
    logger.info(f"🔄 Making predictions from {input_path} using {model_path}")

    # Initialize prediction service
    service = ZomatoPredictionService(model_path=str(model_path))

    # Load data
    logger.info("📖 Loading test data...")
    df_test = pd.read_csv(input_path)
    test_data = df_test.to_dict("records")

    # Make batch predictions
    results = service.predict_batch(test_data)

    if "error" not in results:
        # Save predictions
        df_results = df_test.copy()
        df_results["predicted_rating"] = results["predictions"]
        if results["confidence_scores"]:
            df_results["confidence_score"] = results["confidence_scores"]

        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_results.to_csv(output_path, index=False)

        logger.success(f"✅ Predictions saved to {output_path}")
        logger.info(f"📊 Prediction Summary:")
        logger.info(f"  • Total records processed: {results['input_count']}")
        logger.info(f"  • Successful predictions: {results['successful_predictions']}")
        logger.info(f"  • Features used: {results['feature_count']}")
    else:
        logger.error(f"Prediction failed: {results['error']}")


@app.command()
def test_single_prediction(
    name: str = "Test Restaurant",
    location: str = "Koramangala",
    cuisines: str = "North Indian, Chinese",
    cost_for_two: int = 800,
    model_path: Path = MODELS_DIR / "best_model.pkl",
):
    """
    Test single prediction with sample data.
    """
    logger.info("🧪 Testing single prediction...")

    # Initialize prediction service
    service = ZomatoPredictionService(model_path=str(model_path))

    # Sample restaurant data
    restaurant_data = {
        "name": name,
        "location": location,
        "cuisines": cuisines,
        "cost_for_two": cost_for_two,
        "online_order": "Yes",
        "book_table": "No",
        "rest_type": "Casual Dining",
        "votes": 100,
        "area": location,
        "category": "Dine-out",
    }

    # Validate input
    validation = service.validate_input(restaurant_data)
    logger.info(f"Validation result: {validation}")

    # Make prediction
    if validation["is_valid"]:
        result = service.predict_single(restaurant_data)

        logger.success("🎉 Prediction completed!")
        logger.info(f"📊 Results:")
        logger.info(
            f"  • Predicted Rating: {result.get('predicted_rating', 'N/A'):.2f}"
            if result.get("predicted_rating")
            else "  • Predicted Rating: N/A"
        )
        logger.info(
            f"  • Confidence Score: {result.get('confidence_score', 'N/A'):.2f}"
            if result.get("confidence_score")
            else "  • Confidence Score: N/A"
        )
        logger.info(f"  • Features Used: {result.get('input_features_count', 'N/A')}")
    else:
        logger.error("❌ Validation failed. Cannot make prediction.")


if __name__ == "__main__":
    app()
