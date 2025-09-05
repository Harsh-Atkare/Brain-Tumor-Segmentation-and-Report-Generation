# deployment_check.py - Pre-deployment checks
import os
import sys
from pathlib import Path
import subprocess

class DeploymentChecker:
    def __init__(self):
        self.errors = []
        self.warnings = []
        
    def check_environment(self):
        """Check environment setup."""
        print("Checking environment...")
        
        # Check Python version
        if sys.version_info < (3, 8):
            self.errors.append("Python 3.8+ required")
            
        # Check required directories
        required_dirs = ['data/uploads', 'data/outputs', 'data/models', 'templates']
        for dir_path in required_dirs:
            if not Path(dir_path).exists():
                self.errors.append(f"Missing directory: {dir_path}")
                
        # Check model file
        model_path = Path('data/models/ckpt.tar')
        if not model_path.exists():
            self.warnings.append("Model file not found: data/models/ckpt.tar")
            
        # Check environment file
        if not Path('.env').exists():
            self.warnings.append(".env file not found")
            
        print("✓ Environment check complete")
        
    def check_dependencies(self):
        """Check Python dependencies."""
        print("Checking dependencies...")
        
        try:
            import torch
            import fastapi
            import nibabel
            import sklearn
            print("✓ Core dependencies installed")
        except ImportError as e:
            self.errors.append(f"Missing dependency: {e}")
            
    def check_ports(self):
        """Check if required ports are available."""
        print("Checking ports...")
        
        try:
            import socket
            
            # Check port 8000
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                result = s.connect_ex(('localhost', 8000))
                if result == 0:
                    self.warnings.append("Port 8000 is already in use")
                    
        except Exception as e:
            self.warnings.append(f"Could not check ports: {e}")
            
        print("✓ Port check complete")
        
    def check_disk_space(self):
        """Check available disk space."""
        print("Checking disk space...")
        
        try:
            import shutil
            total, used, free = shutil.disk_usage("/")
            free_gb = free // (1024**3)
            
            if free_gb < 5:
                self.errors.append(f"Low disk space: {free_gb}GB available")
            elif free_gb < 10:
                self.warnings.append(f"Limited disk space: {free_gb}GB available")
                
        except Exception as e:
            self.warnings.append(f"Could not check disk space: {e}")
            
        print("✓ Disk space check complete")
        
    def run_all_checks(self):
        """Run all deployment checks."""
        print("Running deployment checks...\n")
        
        self.check_environment()
        self.check_dependencies()
        self.check_ports()
        self.check_disk_space()
        
        print("\n" + "="*50)
        print("DEPLOYMENT CHECK RESULTS")
        print("="*50)
        
        if self.errors:
            print("ERRORS (must fix before deployment):")
            for error in self.errors:
                print(f"  - {error}")
                
        if self.warnings:
            print("WARNINGS (recommended to fix):")
            for warning in self.warnings:
                print(f"  - {warning}")
                
        if not self.errors and not self.warnings:
            print("All checks passed! Ready for deployment.")
            
        return len(self.errors) == 0

if __name__ == "__main__":
    checker = DeploymentChecker()
    is_ready = checker.run_all_checks()
    sys.exit(0 if is_ready else 1)