#!/bin/bash

# ===================================================================
# Zomato Restaurant Rating Prediction - Production Deployment Script
# ===================================================================

set -e  # Exit on any error

echo "🍽️ Starting Zomato Restaurant Rating Prediction Deployment"
echo "============================================================"

# Configuration
PROJECT_NAME="zomato_prediction"
API_PORT=8000
FRONTEND_PORT=80
ENVIRONMENT=${ENVIRONMENT:-production}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check requirements
check_requirements() {
    print_status "Checking system requirements..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    # Check if ports are available
    if lsof -Pi :$API_PORT -sTCP:LISTEN -t >/dev/null ; then
        print_warning "Port $API_PORT is already in use. Please stop the service using this port."
    fi
    
    if lsof -Pi :$FRONTEND_PORT -sTCP:LISTEN -t >/dev/null ; then
        print_warning "Port $FRONTEND_PORT is already in use. Please stop the service using this port."
    fi
    
    print_success "System requirements check completed"
}

# Setup environment
setup_environment() {
    print_status "Setting up environment..."
    
    # Create necessary directories
    mkdir -p logs temp data/processed models
    
    # Set permissions
    chmod 755 logs temp
    
    # Create environment file if it doesn't exist
    if [ ! -f .env ]; then
        cat > .env << EOF
# Zomato Prediction API Configuration
ENVIRONMENT=production
LOG_LEVEL=info
API_HOST=0.0.0.0
API_PORT=8000

# Database Configuration (optional)
DATABASE_URL=postgresql://zomato_user:zomato_password@postgres:5432/zomato_db

# Redis Configuration
REDIS_URL=redis://redis:6379/0

# Model Configuration
MODEL_PATH=/app/models/best_model.pkl
MLFLOW_TRACKING_URI=http://localhost:5000

# Security (change in production)
SECRET_KEY=your-secret-key-change-this-in-production
API_KEY_HEADER=X-API-Key

# Monitoring
ENABLE_METRICS=true
SENTRY_DSN=your-sentry-dsn-here
EOF
        print_success "Created .env file with default configuration"
        print_warning "Please review and update .env file with your specific configuration"
    fi
}

# Prepare model
prepare_model() {
    print_status "Preparing ML model..."
    
    # Check if model exists
    if [ ! -f "models/best_model.pkl" ]; then
        print_warning "No trained model found at models/best_model.pkl"
        print_status "You need to train your model first. Options:"
        echo "  1. Run your training notebook and save the best model to models/best_model.pkl"
        echo "  2. Copy your trained model from MLflow artifacts to models/best_model.pkl"
        echo "  3. Use the following command to find your best model in MLflow:"
        echo "     find notebooks/mlruns -name '*.pkl' | head -5"
        
        # Try to find a model automatically
        MODEL_FILE=$(find notebooks/mlruns -name "*.pkl" | head -1)
        if [ ! -z "$MODEL_FILE" ]; then
            print_status "Found model file: $MODEL_FILE"
            read -p "Copy this model to models/best_model.pkl? (y/n): " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                cp "$MODEL_FILE" models/best_model.pkl
                print_success "Model copied successfully"
            fi
        fi
    else
        print_success "Model found at models/best_model.pkl"
    fi
}

# Prepare data
prepare_data() {
    print_status "Preparing processed data..."
    
    # Check for processed data files
    PROCESSED_FILES=(
        "data/processed/zomato_master_processed.parquet"
        "data/processed/zomato_features_final.parquet"
    )
    
    for file in "${PROCESSED_FILES[@]}"; do
        if [ ! -f "$file" ]; then
            print_warning "Processed data file not found: $file"
            print_status "Run your data processing pipeline first:"
            echo "  python -m zomato_prediction.dataset"
            echo "  python -m zomato_prediction.features"
        else
            print_success "Found: $file"
        fi
    done
}

# Build Docker images
build_images() {
    print_status "Building Docker images..."
    
    # Build the main application image
    docker-compose build zomato-api
    
    print_success "Docker images built successfully"
}

# Deploy services
deploy_services() {
    print_status "Deploying services..."
    
    # Start services with Docker Compose
    docker-compose up -d
    
    # Wait for services to be ready
    print_status "Waiting for services to start..."
    sleep 10
    
    # Check API health
    for i in {1..30}; do
        if curl -f http://localhost:$API_PORT/health >/dev/null 2>&1; then
            print_success "API is healthy and responding"
            break
        fi
        if [ $i -eq 30 ]; then
            print_error "API failed to start after 30 attempts"
            print_status "Check logs with: docker-compose logs zomato-api"
            exit 1
        fi
        sleep 2
    done
}

# Run tests
run_tests() {
    print_status "Running basic functionality tests..."
    
    # Test API endpoints
    echo "Testing API endpoints..."
    
    # Health check
    if curl -f http://localhost:$API_PORT/health >/dev/null 2>&1; then
        print_success "✓ Health check endpoint working"
    else
        print_error "✗ Health check endpoint failed"
    fi
    
    # Test prediction endpoint with sample data
    SAMPLE_DATA='{
        "name": "Test Restaurant",
        "location": "Koramangala",
        "cuisines": "North Indian, Chinese",
        "cost_for_two": 800,
        "online_order": "Yes",
        "book_table": "No"
    }'
    
    if curl -f -X POST "http://localhost:$API_PORT/predict" \
        -H "Content-Type: application/json" \
        -d "$SAMPLE_DATA" >/dev/null 2>&1; then
        print_success "✓ Prediction endpoint working"
    else
        print_warning "✗ Prediction endpoint test failed (this might be expected if no model is loaded)"
    fi
    
    # Test frontend
    if curl -f http://localhost:$FRONTEND_PORT >/dev/null 2>&1; then
        print_success "✓ Frontend is accessible"
    else
        print_error "✗ Frontend is not accessible"
    fi
}

# Show deployment info
show_deployment_info() {
    print_success "🎉 Deployment completed successfully!"
    echo ""
    echo "========================================"
    echo "🍽️ Zomato Prediction System is Ready!"
    echo "========================================"
    echo ""
    echo "📱 Frontend URL:     http://localhost:$FRONTEND_PORT"
    echo "🔌 API URL:          http://localhost:$API_PORT"
    echo "📖 API Docs:         http://localhost:$API_PORT/docs"
    echo "📊 Health Check:     http://localhost:$API_PORT/health"
    echo ""
    echo "🐳 Docker Services:"
    echo "   - zomato-api      (Main ML API)"
    echo "   - nginx           (Web server & reverse proxy)"
    echo "   - redis           (Caching)"
    echo "   - postgres        (Database)"
    echo ""
    echo "🛠️  Management Commands:"
    echo "   View logs:        docker-compose logs -f"
    echo "   Stop services:    docker-compose down"
    echo "   Restart:          docker-compose restart"
    echo "   Update:           git pull && docker-compose up --build -d"
    echo ""
    echo "📊 Monitoring (if enabled):"
    echo "   Prometheus:       http://localhost:9090"
    echo "   Grafana:          http://localhost:3000 (admin/admin)"
    echo ""
    print_warning "Remember to:"
    echo "  - Update your .env file with production secrets"
    echo "  - Configure SSL certificates for HTTPS"
    echo "  - Set up proper backup procedures"
    echo "  - Monitor system performance and logs"
}

# Main deployment function
main() {
    echo "Starting deployment process..."
    
    check_requirements
    setup_environment
    prepare_model
    prepare_data
    build_images
    deploy_services
    run_tests
    show_deployment_info
    
    print_success "Deployment script completed successfully! 🚀"
}

# Handle script interruption
cleanup() {
    print_warning "Deployment interrupted. Cleaning up..."
    docker-compose down >/dev/null 2>&1 || true
    exit 1
}

trap cleanup INT TERM

# Run main function
main "$@"
