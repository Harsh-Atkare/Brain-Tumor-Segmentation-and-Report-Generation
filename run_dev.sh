# run_dev.sh - Development server script
#!/bin/bash

echo "Starting Brain Tumor Segmentation API in development mode..."

# Activate virtual environment
source venv/bin/activate

# Set development environment variables
export ENVIRONMENT=development
export LOG_LEVEL=debug# cleanup.py - Cleanup utility for old files

# Start the server with hot reload
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload --log-level debug