# app/api/routes/upload.py
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from typing import Dict
from app.models.schemas import FileUploadResponse
from app.services.file_service import FileService
from app.api.dependencies import get_file_service
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/", response_model=FileUploadResponse)
async def upload_files(
    flair: UploadFile = File(..., description="FLAIR MRI image"),
    t1ce: UploadFile = File(..., description="T1CE MRI image"),
    t2: UploadFile = File(..., description="T2 MRI image"),
    file_service: FileService = Depends(get_file_service)
):

    try:
        files = {"flair": flair, "t1ce": t1ce, "t2": t2}
        upload_id, saved_files = await file_service.save_uploaded_files(files)
        
        logger.info(f"Files uploaded successfully with ID: {upload_id}")
        
        return FileUploadResponse(
            upload_id=upload_id,
            message="Files uploaded successfully",
            files_received={k: str(v) for k, v in saved_files.items()}
        )
    except Exception as e:
        logger.error(f"File upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
