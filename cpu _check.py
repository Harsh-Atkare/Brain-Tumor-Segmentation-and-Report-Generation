#!/usr/bin/env python3
"""
Verification script for Brain Tumor Segmentation CPU setup
"""

import sys
import torch
from pathlib import Path

def check_torch_cpu():
    """Verify PyTorch is using CPU."""
    print("🔍 Checking PyTorch setup...")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    # Test tensor creation
    x = torch.randn(2, 3)
    print(f"Default tensor device: {x.device}")
    
    if torch.cuda.is_available():
        print("⚠️  CUDA is available but we'll force CPU usage")
    else:
        print("✅ Using CPU (CUDA not available)")

def check_file_structure():
    """Check project structure."""
    print("\n📁 Checking file structure...")
    
    required_files = {
        "app/main.py": "Main FastAPI application",
        "app/core/config.py": "Configuration settings",
        "app/models/unet3d.py": "UNet3D model",
        "app/models/schemas.py": "Pydantic schemas",
        "app/services/segmentation_service.py": "Segmentation service",
        "app/services/feature_extraction_service.py": "Feature extraction service",
        "app/services/file_service.py": "File handling service",
        "app/utils/preprocessing.py": "Preprocessing utilities",
        "app/utils/postprocessing.py": "Postprocessing utilities",
        "templates/index.html": "Web interface template",
        ".env": "Environment configuration"
    }
    
    missing_files = []
    for file_path, description in required_files.items():
        if Path(file_path).exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - {description}")
            missing_files.append(file_path)
    
    return len(missing_files) == 0

def check_model_file():
    """Check if model file exists."""
    print("\n🤖 Checking model file...")
    model_path = Path("data/models/ckpt.tar")
    
    if model_path.exists():
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"✅ Model file found: {size_mb:.1f} MB")
        return True
    else:
        print("❌ Model file not found: data/models/ckpt.tar")
        print("   Please copy your model file to this location")
        return False

def check_dependencies():
    """Check required packages."""
    print("\n📦 Checking dependencies...")
    
    required_packages = [
        "fastapi", "uvicorn", "torch", "nibabel", 
        "sklearn", "pandas", "numpy", "scipy"
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing_packages.append(package)
    
    return len(missing_packages) == 0

def run_verification():
    """Run all verification checks."""
    print("🧠 Brain Tumor Segmentation - CPU Setup Verification")
    print("=" * 55)
    
    checks = [
        ("PyTorch CPU", check_torch_cpu),
        ("File Structure", check_file_structure), 
        ("Model File", check_model_file),
        ("Dependencies", check_dependencies)
    ]
    
    all_passed = True
    for name, check_func in checks:
        try:
            result = check_func()
            if result is False:
                all_passed = False
        except Exception as e:
            print(f"❌ {name} check failed: {e}")
            all_passed = False
    
    print("\n" + "=" * 55)
    if all_passed:
        print("🎉 All checks passed! Ready to run the application.")
        print("\nTo start the server:")
        print("uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload")
    else:
        print("❌ Some checks failed. Please fix the issues above.")
        
    return all_passed

if __name__ == "__main__":
    success = run_verification()
    sys.exit(0 if success else 1)