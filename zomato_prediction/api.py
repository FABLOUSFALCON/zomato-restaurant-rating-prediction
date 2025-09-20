"""
FastAPI Application for Zomato Restaurant Rating Prediction
===========================================================

This module provides REST API endpoints for the Zomato rating prediction system.
"""

import asyncio
import time
from typing import Dict, List, Optional, Any
from datetime import datetime

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel, Field, validator
import json
from loguru import logger

from .prediction_service import ZomatoPredictionService

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional
import asyncio
import time
from datetime import datetime
import json
import uuid

from loguru import logger
import pandas as pd

from zomato_prediction.modeling.predict import ZomatoPredictionService
from zomato_prediction.config import MODELS_DIR


# Pydantic models for request/response validation
class RestaurantInput(BaseModel):
    """Input model for single restaurant prediction."""

    name: str = Field(..., description="Restaurant name", example="Cafe Coffee Day")
    location: str = Field(
        ..., description="Restaurant location/area", example="Koramangala"
    )
    cuisines: str = Field(
        ...,
        description="Cuisines served (comma-separated)",
        example="North Indian, Chinese, Continental",
    )
    cost_for_two: Optional[float] = Field(
        None, description="Approximate cost for two people", example=800.0
    )
    rest_type: Optional[str] = Field(
        None, description="Restaurant type", example="Casual Dining"
    )
    online_order: Optional[str] = Field(
        "Yes", description="Online ordering available", example="Yes"
    )
    book_table: Optional[str] = Field(
        "No", description="Table booking available", example="No"
    )
    votes: Optional[int] = Field(
        100, description="Number of votes/reviews", example=150
    )
    area: Optional[str] = Field(
        None, description="Broader area classification", example="Koramangala"
    )
    category: Optional[str] = Field(
        "Dine-out", description="Restaurant category", example="Dine-out"
    )
    address: Optional[str] = Field(
        None, description="Full address", example="123 Main Street, Koramangala"
    )
    dish_liked: Optional[str] = Field(
        None, description="Popular dishes", example="Biryani, Pizza"
    )
    reviews_list: Optional[str] = Field("[]", description="Reviews data", example="[]")

    @validator("cost_for_two")
    def validate_cost(cls, v):
        if v is not None and (v < 50 or v > 10000):
            raise ValueError("Cost for two should be between 50 and 10000")
        return v

    @validator("votes")
    def validate_votes(cls, v):
        if v is not None and v < 0:
            raise ValueError("Votes cannot be negative")
        return v


class BatchRestaurantInput(BaseModel):
    """Input model for batch restaurant predictions."""

    restaurants: List[RestaurantInput] = Field(
        ..., description="List of restaurants to predict"
    )

    @validator("restaurants")
    def validate_batch_size(cls, v):
        if len(v) > 100:
            raise ValueError("Batch size cannot exceed 100 restaurants")
        return v


class PredictionResponse(BaseModel):
    """Response model for single prediction."""

    predicted_rating: float = Field(
        ..., description="Predicted restaurant rating (0-5)"
    )
    confidence_score: Optional[float] = Field(
        None, description="Prediction confidence (0-1)"
    )
    input_features_count: int = Field(..., description="Number of features used")
    model_info: Dict[str, Any] = Field(..., description="Model metadata")
    processing_stats: Dict[str, Any] = Field(..., description="Processing statistics")
    prediction_id: str = Field(..., description="Unique prediction ID")
    timestamp: datetime = Field(..., description="Prediction timestamp")


class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions."""

    predictions: List[float] = Field(..., description="List of predicted ratings")
    confidence_scores: Optional[List[float]] = Field(
        None, description="List of confidence scores"
    )
    input_count: int = Field(..., description="Number of input restaurants")
    successful_predictions: int = Field(
        ..., description="Number of successful predictions"
    )
    feature_count: int = Field(..., description="Number of features used")
    model_info: Dict[str, Any] = Field(..., description="Model metadata")
    processing_stats: Dict[str, Any] = Field(..., description="Processing statistics")
    batch_id: str = Field(..., description="Unique batch ID")
    timestamp: datetime = Field(..., description="Prediction timestamp")


class ValidationResponse(BaseModel):
    """Response model for input validation."""

    is_valid: bool = Field(..., description="Whether input is valid")
    warnings: List[str] = Field(..., description="Validation warnings")
    errors: List[str] = Field(..., description="Validation errors")
    suggestions: List[str] = Field(..., description="Improvement suggestions")


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    model_info: Dict[str, Any] = Field(..., description="Model information")
    uptime_seconds: float = Field(..., description="Service uptime in seconds")
    timestamp: datetime = Field(..., description="Health check timestamp")


# Global prediction service instance
prediction_service: Optional[ZomatoPredictionService] = None
app_start_time = time.time()


def get_prediction_service() -> ZomatoPredictionService:
    """Dependency to get prediction service instance."""
    global prediction_service
    if prediction_service is None:
        raise HTTPException(
            status_code=503, detail="Prediction service not initialized"
        )
    return prediction_service


# FastAPI app initialization
app = FastAPI(
    title="Zomato Restaurant Rating Prediction API",
    description="""
    🍽️ **Production-ready ML API for predicting Zomato restaurant ratings**
    
    This API provides sophisticated restaurant rating predictions using advanced machine learning 
    models with comprehensive feature engineering including:
    
    - **Text Analysis**: Restaurant names, cuisines, and reviews
    - **Geolocation Features**: Location-based popularity and competition metrics  
    - **Business Intelligence**: Cost analysis, service features, and market positioning
    - **90+ Engineered Features**: From your original 30 features to 90+ sophisticated ML features
    
    Built with production-grade architecture including:
    - Real-time single predictions
    - Efficient batch processing
    - Input validation and error handling
    - Comprehensive logging and monitoring
    - Model metadata and confidence scoring
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS configuration for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Initialize the prediction service on startup."""
    global prediction_service

    try:
        logger.info("🚀 Starting Zomato Prediction API...")

        # Try to find the best model from MLflow or use a default
        model_path = MODELS_DIR / "best_model.pkl"

        # For production, you might want to load from MLflow:
        # prediction_service = ZomatoPredictionService(model_uri="models:/ZomatoRatingPredictor/Production")

        # For now, use local model (you'll need to save your best model here)
        if model_path.exists():
            prediction_service = ZomatoPredictionService(model_path=str(model_path))
            logger.success("✅ Prediction service initialized with local model")
        else:
            logger.warning("⚠️ No model found, service will be in limited mode")
            prediction_service = ZomatoPredictionService()

    except Exception as e:
        logger.error(f"❌ Failed to initialize prediction service: {e}")
        # Don't fail startup, but log the error
        prediction_service = None


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "🍽️ Zomato Restaurant Rating Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "status": "operational",
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Comprehensive health check endpoint."""
    global prediction_service, app_start_time

    uptime = time.time() - app_start_time

    if prediction_service is None:
        return HealthResponse(
            status="degraded",
            model_loaded=False,
            model_info={},
            uptime_seconds=uptime,
            timestamp=datetime.now(),
        )

    return HealthResponse(
        status="healthy",
        model_loaded=prediction_service.model is not None,
        model_info=prediction_service.model_metadata,
        uptime_seconds=uptime,
        timestamp=datetime.now(),
    )


@app.post("/validate", response_model=ValidationResponse)
async def validate_input(
    restaurant: RestaurantInput,
    service: ZomatoPredictionService = Depends(get_prediction_service),
):
    """Validate restaurant input data."""
    try:
        restaurant_dict = restaurant.dict()
        validation_result = service.validate_input(restaurant_dict)

        return ValidationResponse(**validation_result)

    except Exception as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=f"Validation failed: {str(e)}")


@app.post("/predict", response_model=PredictionResponse)
async def predict_single(
    restaurant: RestaurantInput,
    service: ZomatoPredictionService = Depends(get_prediction_service),
):
    """
    Predict rating for a single restaurant.

    Processes the restaurant data through the complete ML pipeline:
    1. Data cleaning and standardization
    2. Advanced feature engineering (90+ features)
    3. ML model prediction with confidence scoring
    """
    try:
        logger.info(f"🔮 Processing prediction request for: {restaurant.name}")

        # Convert to dictionary
        restaurant_dict = restaurant.dict()

        # Make prediction
        result = service.predict_single(restaurant_dict)

        if "error" in result:
            raise HTTPException(
                status_code=500, detail=f"Prediction failed: {result['error']}"
            )

        # Add metadata
        result["prediction_id"] = str(uuid.uuid4())
        result["timestamp"] = datetime.now()

        logger.success(f"✅ Prediction completed: {result['predicted_rating']:.2f}")
        return PredictionResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(
    batch_input: BatchRestaurantInput,
    background_tasks: BackgroundTasks,
    service: ZomatoPredictionService = Depends(get_prediction_service),
):
    """
    Predict ratings for multiple restaurants efficiently.

    Optimized for batch processing with parallel feature engineering
    and efficient model inference.
    """
    try:
        logger.info(
            f"🔮 Processing batch prediction for {len(batch_input.restaurants)} restaurants"
        )

        # Convert to list of dictionaries
        restaurants_data = [restaurant.dict() for restaurant in batch_input.restaurants]

        # Make batch prediction
        result = service.predict_batch(restaurants_data)

        if "error" in result:
            raise HTTPException(
                status_code=500, detail=f"Batch prediction failed: {result['error']}"
            )

        # Add metadata
        result["batch_id"] = str(uuid.uuid4())
        result["timestamp"] = datetime.now()

        logger.success(
            f"✅ Batch prediction completed: {result['successful_predictions']} predictions"
        )
        return BatchPredictionResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(
            status_code=500, detail=f"Batch prediction failed: {str(e)}"
        )


@app.get("/model/info")
async def get_model_info(
    service: ZomatoPredictionService = Depends(get_prediction_service),
):
    """Get detailed model information and feature importance."""
    try:
        model_info = service.model_metadata.copy()

        # Add feature importance if available
        feature_importance = service.get_feature_importance()
        if feature_importance:
            model_info["feature_importance"] = feature_importance

        # Add feature names
        if service.feature_names:
            model_info["feature_names"] = service.feature_names
            model_info["feature_count"] = len(service.feature_names)

        return model_info

    except Exception as e:
        logger.error(f"Model info error: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get model info: {str(e)}"
        )


@app.get("/metrics")
async def get_metrics():
    """Get API metrics and statistics."""
    global app_start_time

    uptime = time.time() - app_start_time

    # In production, you'd collect these from a metrics store
    metrics = {
        "uptime_seconds": uptime,
        "uptime_human": f"{uptime/3600:.1f} hours",
        "model_loaded": prediction_service is not None
        and prediction_service.model is not None,
        "api_version": "1.0.0",
        "features_engineered": "90+",
        "model_type": (
            prediction_service.model_metadata.get("name", "Unknown")
            if prediction_service
            else "None"
        ),
    }

    return metrics


@app.post("/feedback")
async def submit_feedback(feedback: Dict[str, Any], background_tasks: BackgroundTasks):
    """
    Submit prediction feedback for model improvement.

    In production, this would store feedback for model retraining.
    """
    try:
        # Add timestamp
        feedback["timestamp"] = datetime.now().isoformat()
        feedback["feedback_id"] = str(uuid.uuid4())

        # In production, save to database or feedback store
        logger.info(f"📝 Feedback received: {feedback.get('feedback_id')}")

        return {
            "message": "Feedback received successfully",
            "feedback_id": feedback["feedback_id"],
            "status": "stored",
        }

    except Exception as e:
        logger.error(f"Feedback error: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to store feedback: {str(e)}"
        )


# Error handlers
@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    """Handle validation errors."""
    logger.warning(f"Validation error: {exc}")
    return JSONResponse(
        status_code=400, content={"detail": f"Validation error: {str(exc)}"}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    logger.error(f"Unexpected error: {exc}")
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


if __name__ == "__main__":
    import uvicorn

    logger.info("🚀 Starting Zomato Prediction API server...")
    uvicorn.run(
        "zomato_prediction.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
