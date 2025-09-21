# setup.sh - Setup script for the application
#!/bin/bash

echo "Setting up Brain Tumor Segmentation Application with AI Reporting..."

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Create necessary directories
echo "Creating directories..."
mkdir -p data/uploads
mkdir -p data/outputs  
mkdir -p data/models
mkdir -p data/reports
mkdir -p static/css
mkdir -p static/js
mkdir -p static/images
mkdir -p templates
mkdir -p logs

# Create .gitkeep files
touch data/uploads/.gitkeep
touch data/outputs/.gitkeep
touch data/models/.gitkeep
touch data/reports/.gitkeep

# Copy environment file
if [ ! -f .env ]; then
    cp .env.example .env
    echo "Created .env file from example. Please update with your settings."
fi

echo "Setup complete!"
echo ""
echo "Next steps:"
echo "1. Place your trained model (ckpt.tar) in data/models/"
echo "2. Update .env file with your Hugging Face API key and other settings"
echo "3. Run: uvicorn app.main:app --reload"
