import sys
from pathlib import Path

def check_project_structure():
    """Check if all required files exist."""
    required_files = [
        "app/main.py",
        "app/core/config.py", 
        "app/models/unet3d.py",
        "app/models/schemas.py",
        "app/services/segmentation_service.py",
        "app/services/feature_extraction_service.py", 
        "app/services/file_service.py",
        "app/utils/preprocessing.py",
        "app/utils/postprocessing.py",
        "app/api/routes/upload.py",
        "app/api/routes/segmentation.py",
        "app/api/routes/features.py",
        "app/api/dependencies.py",
        "templates/index.html",
        "data/models/ckpt.tar",
        ".env"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print("Missing files:")
        for file in missing_files:
            print(f"  - {file}")
        return False
    else:
        print("All required files present!")
        return True

if __name__ == "__main__":
    if check_project_structure():
        print("Ready to run the application!")
    else:
        print("Please create the missing files before running.")
