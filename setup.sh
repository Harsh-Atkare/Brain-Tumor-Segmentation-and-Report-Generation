echo "Setting up Brain Tumor Segmentation Application..."

echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt
echo "Creating directories..."
mkdir -p data/uploads
mkdir -p data/outputs  
mkdir -p logs
touch data/uploads/.gitkeep
touch data/outputs/.gitkeep
touch data/models/.gitkeep

# Copy environment file
if [ ! -f .env ]; then
    cp .env.example .env
    echo "Created .env file from example. Please update with your settings."
fi

echo "Setup complete!"
echo ""
echo "Next steps:"
echo "1. Place your trained model (ckpt.tar) in data/models/"
echo "2. Update .env file with your settings"
echo "3. Run: uvicorn app.main:app --reload"

# run_dev.sh - Development server script
#!/bin/bash

echo "Starting Brain Tumor Segmentation API in development mode..."

# Activate virtual environment
source venv/bin/activate

# Set development environment variables
export ENVIRONMENT=development
export LOG_LEVEL=debug

# Start the server with hot reload
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload --log-level debug