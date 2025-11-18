#!/bin/bash
# Setup script for Novelty LLM System

set -e

echo "🚀 Setting up Novelty LLM System..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if Python 3.11+ is installed
echo "📋 Checking Python version..."
if ! command -v python3.11 &> /dev/null; then
    echo -e "${RED}❌ Python 3.11+ is required but not found${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Python 3.11+ found${NC}"

# Check if Docker is installed
echo "📋 Checking Docker..."
if ! command -v docker &> /dev/null; then
    echo -e "${YELLOW}⚠️  Docker not found. Some features will not be available.${NC}"
else
    echo -e "${GREEN}✓ Docker found${NC}"
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo -e "${YELLOW}⚠️  Docker Compose not found. Some features will not be available.${NC}"
else
    echo -e "${GREEN}✓ Docker Compose found${NC}"
fi

# Create virtual environment
echo "📦 Creating virtual environment..."
python3.11 -m venv venv
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Install development dependencies
read -p "Install development dependencies? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    pip install -r requirements-dev.txt
    echo -e "${GREEN}✓ Development dependencies installed${NC}"
fi

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "⚙️  Creating .env file..."
    cp .env.example .env
    echo -e "${YELLOW}⚠️  Please edit .env file with your configuration${NC}"
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p logs
mkdir -p data
mkdir -p models
mkdir -p cache

# Download embedding model
echo "🤖 Downloading embedding model..."
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
echo -e "${GREEN}✓ Embedding model downloaded${NC}"

# Setup pre-commit hooks (if dev dependencies installed)
if [ -f "venv/bin/pre-commit" ]; then
    echo "🔧 Setting up pre-commit hooks..."
    pre-commit install
    echo -e "${GREEN}✓ Pre-commit hooks installed${NC}"
fi

echo ""
echo -e "${GREEN}✅ Setup complete!${NC}"
echo ""
echo "Next steps:"
echo "1. Activate virtual environment: source venv/bin/activate"
echo "2. Edit .env file with your configuration"
echo "3. Start services: make docker-up"
echo "4. Run the application: make run-dev"
echo ""
echo "For more information, see README.md"
