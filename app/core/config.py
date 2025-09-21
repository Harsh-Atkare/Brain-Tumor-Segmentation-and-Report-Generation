# app/core/config.py
from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    PROJECT_NAME: str = "Brain Tumor Segmentation API"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api"

    BASE_DIR: Path = Path(__file__).parent.parent.parent
    UPLOAD_DIR: Path = BASE_DIR / "data" / "uploads"
    OUTPUT_DIR: Path = BASE_DIR / "data" / "outputs"
    MODEL_DIR: Path = BASE_DIR / "data" / "models"
    REPORTS_DIR: Path = BASE_DIR / "data" / "reports"

    MODEL_PATH: str
    DEVICE: str 

    MAX_FILE_SIZE: int = 500 * 1024 * 1024
    ALLOWED_EXTENSIONS: set = {".nii", ".nii.gz"}


    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30


    BACKEND_CORS_ORIGINS: list[str]


    HUGGINGFACEHUB_ACCESS_TOKEN: str
    MODEL_NAME: str

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()

settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
settings.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
settings.MODEL_DIR.mkdir(parents=True, exist_ok=True)
settings.REPORTS_DIR.mkdir(parents=True, exist_ok=True)

print(f"Using device: {settings.DEVICE}")
