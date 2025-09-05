from pydantic_settings import BaseSettings
from pathlib import Path
import os

class Settings(BaseSettings):
    PROJECT_NAME: str = "Brain Tumor Segmentation API"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api"
    
    # Directories
    BASE_DIR: Path = Path(__file__).parent.parent.parent
    UPLOAD_DIR: Path = BASE_DIR / "data" / "uploads"
    OUTPUT_DIR: Path = BASE_DIR / "data" / "outputs" 
    MODEL_DIR: Path = BASE_DIR / "data" / "models"
    
    # Model settings - FORCE CPU
    MODEL_PATH: str = str(MODEL_DIR / "ckpt.tar")
    DEVICE: str = "cpu"  # Force CPU usage
    
    # File settings
    MAX_FILE_SIZE: int = 500 * 1024 * 1024  # 500MB
    ALLOWED_EXTENSIONS: set = {".nii", ".nii.gz"}
    
    # Security
    SECRET_KEY: str = "your-secret-key-here"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # CORS
    BACKEND_CORS_ORIGINS: list = ["http://localhost:3000", "http://localhost:8080", "http://localhost:8000"]
    
    class Config:
        env_file = ".env"

settings = Settings()

settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
settings.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
settings.MODEL_DIR.mkdir(parents=True, exist_ok=True)

print(f"Using device: {settings.DEVICE}")