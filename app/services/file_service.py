import aiofiles
import uuid
from pathlib import Path
from typing import Dict, Optional
from fastapi import UploadFile, HTTPException
from app.core.config import settings
import shutil

class FileService:
    def __init__(self):
        self.upload_dir = settings.UPLOAD_DIR
        self.output_dir = settings.OUTPUT_DIR
    
    async def save_uploaded_files(self, files: Dict[str, UploadFile]) -> Dict[str, Path]:
        upload_id = str(uuid.uuid4())
        upload_path = self.upload_dir / upload_id
        upload_path.mkdir(exist_ok=True)
        
        saved_files = {}
        
        for file_type, file in files.items():
            if not self._validate_file(file):
                raise HTTPException(status_code=400, detail=f"Invalid file format for {file_type}")
            
            file_path = upload_path / f"{file_type}.nii.gz"
            
            async with aiofiles.open(file_path, 'wb') as f:
                content = await file.read()
                await f.write(content)
            
            saved_files[file_type] = file_path
        
        return upload_id, saved_files
    
    def _validate_file(self, file: UploadFile) -> bool:
        if file.size > settings.MAX_FILE_SIZE:
            return False
        
        filename = file.filename.lower()
        return any(filename.endswith(ext) for ext in settings.ALLOWED_EXTENSIONS)
    
    def get_upload_files(self, upload_id: str) -> Optional[Dict[str, Path]]:
        upload_path = self.upload_dir / upload_id
        if not upload_path.exists():
            return None
        
        files = {}
        for file_type in ['flair', 't1ce', 't2']:
            file_path = upload_path / f"{file_type}.nii.gz"
            if file_path.exists():
                files[file_type] = file_path
        
        return files if len(files) == 3 else None
    
    def cleanup_upload(self, upload_id: str):
        upload_path = self.upload_dir / upload_id
        if upload_path.exists():
            shutil.rmtree(upload_path)