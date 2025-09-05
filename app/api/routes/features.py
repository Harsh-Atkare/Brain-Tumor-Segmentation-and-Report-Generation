from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from app.models.schemas import (
    FeatureExtractionRequest, FeatureExtractionResponse, 
    TaskStatusResponse, TaskStatus
)
from app.services.feature_extraction_service import FeatureExtractionService
from app.services.file_service import FileService
from app.services.task_service import TaskService
from app.api.dependencies import (
    get_feature_extraction_service, get_file_service, get_task_service
)
from fastapi.responses import FileResponse
from app.core.config import settings
from pathlib import Path
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

def run_feature_extraction_task(task_id: str, segmentation_task_id: str,
                               feature_service: FeatureExtractionService,
                               file_service: FileService,
                               task_service: TaskService):
    """Background task for feature extraction."""
    try:
        task_service.update_task(task_id, TaskStatus.PROCESSING, 0.1,
                               "Starting feature extraction...")
        
        seg_task = task_service.get_task(segmentation_task_id)
        if not seg_task or seg_task.status != TaskStatus.COMPLETED:
            raise Exception("Segmentation task not completed")
        
        upload_id = seg_task.result.get('upload_id') if seg_task.result else None
        if not upload_id:
            seg_task_data = task_service.tasks.get(segmentation_task_id, {})
            upload_id = seg_task_data.get('upload_id')
        
        if not upload_id:
            raise Exception("Cannot find original upload files")
        
        file_paths = file_service.get_upload_files(upload_id)
        if not file_paths:
            raise Exception("Original files not found")
        
        segmentation_path = Path(seg_task.result['output_path'])
        
        task_service.update_task(task_id, progress=0.5,
                               message="Extracting features...")
        features = feature_service.extract_features(
            file_paths, segmentation_path, f"case_{task_id}"
        )

        output_path = settings.OUTPUT_DIR / f"{task_id}_features.csv"
        feature_service.save_features_to_csv(features, output_path)
        
        task_service.update_task(task_id, TaskStatus.COMPLETED, 1.0,
                               "Feature extraction completed successfully",
                               {"features": features, "output_path": str(output_path)})
        
    except Exception as e:
        logger.error(f"Feature extraction task {task_id} failed: {e}")
        task_service.update_task(task_id, TaskStatus.FAILED,
                               message=f"Feature extraction failed: {str(e)}")

@router.post("/extract", response_model=FeatureExtractionResponse)
async def extract_features(
    request: FeatureExtractionRequest,
    background_tasks: BackgroundTasks,
    feature_service: FeatureExtractionService = Depends(get_feature_extraction_service),
    file_service: FileService = Depends(get_file_service),
    task_service: TaskService = Depends(get_task_service)
):
    """Extract clinical features from segmentation."""
    try:
        seg_task = task_service.get_task(request.task_id)
        if not seg_task:
            raise HTTPException(status_code=404, 
                              detail="Segmentation task not found")
        
        if seg_task.status != TaskStatus.COMPLETED:
            raise HTTPException(status_code=400,
                              detail="Segmentation task not completed yet")
        
        task_id = task_service.create_task("feature_extraction",
                                         segmentation_task_id=request.task_id)
        background_tasks.add_task(
            run_feature_extraction_task, task_id, request.task_id,
            feature_service, file_service, task_service
        )
        
        return FeatureExtractionResponse(
            task_id=task_id,
            status=TaskStatus.PENDING,
            message="Feature extraction task started"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start feature extraction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status/{task_id}", response_model=TaskStatusResponse)
async def get_feature_extraction_status(
    task_id: str,
    task_service: TaskService = Depends(get_task_service)
):
    """Get feature extraction task status."""
    task = task_service.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return task

@router.get("/download/{task_id}")
async def download_features(
    task_id: str,
    task_service: TaskService = Depends(get_task_service)
):
    """Download feature extraction results."""
    task = task_service.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    if task.status != TaskStatus.COMPLETED:
        raise HTTPException(status_code=400,
                          detail="Task not completed yet")
    
    if not task.result or 'output_path' not in task.result:
        raise HTTPException(status_code=500,
                          detail="No output file available")
    
    output_path = task.result['output_path']
    return FileResponse(
        path=output_path,
        filename=f"clinical_features_{task_id}.csv",
        media_type="text/csv"
    )
