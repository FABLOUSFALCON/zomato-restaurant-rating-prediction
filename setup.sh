#!/bin/bash

# Zomato Prediction Project Setup Script
# This script sets up the complete environment for the Zomato Restaurant Rating Prediction project

set -e  # Exit on any error

echo "🍽️  Zomato Restaurant Rating Prediction - Environment Setup"
echo "=========================================================="

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    print_error "Conda is not installed. Please install Anaconda or Miniconda first."
    exit 1
fi

print_status "Conda found: $(conda --version)"

# Ask user which environment to create
echo ""
echo "⚠️  IMPORTANT: This project requires the complete newAge environment"
echo "Choose environment setup option:"
echo "1. newAge environment (RECOMMENDED - battle-tested with 639 dependencies)"
echo "2. Custom name environment (same dependencies, different name)"
read -p "Enter your choice (1-2): " choice

case $choice in
    1)
        ENV_FILE="environment.yml"
        ENV_NAME="newAge"
        print_status "Setting up complete newAge environment (639 dependencies)..."
        ;;
    2)
        ENV_FILE="environment.yml"
        read -p "Enter custom environment name: " ENV_NAME
        print_status "Setting up complete environment as '$ENV_NAME'..."
        ;;
    *)
        print_error "Invalid choice. Exiting."
        exit 1
        ;;
esac

# Check if environment file exists
if [ ! -f "$ENV_FILE" ]; then
    print_error "Environment file $ENV_FILE not found!"
    exit 1
fi

# Remove existing environment if it exists
if conda env list | grep -q "^$ENV_NAME "; then
    print_warning "Environment $ENV_NAME already exists. Removing it..."
    conda env remove -n "$ENV_NAME" -y
fi

# Create new environment
print_status "Creating conda environment from $ENV_FILE..."
conda env create -n "$ENV_NAME" -f "$ENV_FILE"

print_status "Environment created successfully!"

# Activate environment and install additional pip packages if needed
print_status "Activating environment and installing additional packages..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# Install NLTK data
print_status "Downloading NLTK data..."
python -c "
import nltk
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    print('NLTK data downloaded successfully')
except Exception as e:
    print(f'Warning: Could not download NLTK data: {e}')
"

# Create necessary directories
print_status "Creating project directories..."
mkdir -p data/raw data/interim data/processed data/external
mkdir -p models logs notebooks/outputs
mkdir -p tests/unit tests/integration

# Set up MLflow
print_status "Initializing MLflow..."
export MLFLOW_TRACKING_URI="sqlite:///mlflow.db"
python -c "
try:
    import mlflow
    print('MLflow initialized successfully')
except ImportError:
    print('Warning: MLflow not available')
"

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    print_status "Creating .env file..."
    cat > .env << EOF
# Environment Configuration
DEBUG=true
LOG_LEVEL=INFO

# MLflow Configuration
MLFLOW_TRACKING_URI=sqlite:///mlflow.db
MLFLOW_ARTIFACT_ROOT=./mlruns

# Redis Configuration (optional)
REDIS_URL=redis://localhost:6379
REDIS_PASSWORD=

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000

# Model Configuration
MODEL_NAME=xgboost_optimized
MODEL_VERSION=latest
EOF
    print_status ".env file created with default configuration"
else
    print_status ".env file already exists"
fi

echo ""
echo -e "${GREEN}🎉 Environment setup complete!${NC}"
echo ""
echo "Next steps:"
echo "1. Activate the environment:"
echo "   conda activate $ENV_NAME"
echo ""
echo "2. Run the project:"
echo "   make serve    # Start FastAPI server"
echo "   make mlflow   # Start MLflow UI"
echo "   make train    # Train models"
echo ""
echo "3. Run tests:"
echo "   make test     # Run unit tests"
echo "   make lint     # Check code quality"
echo ""
echo "4. View documentation:"
echo "   Open http://localhost:8000/docs (after starting the server)"
echo ""
echo -e "${YELLOW}Note:${NC} Make sure to activate the environment before running any commands!"
