# run_prod.sh - Production server script  
#!/bin/bash

echo "Starting Brain Tumor Segmentation API in production mode..."

# Activate virtual environment
source venv/bin/activate

# Set production environment variables
export ENVIRONMENT=production
export LOG_LEVEL=info

# Start the server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4