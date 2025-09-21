from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from app.models.schemas import (
    SegmentationRequest, SegmentationResponse, TaskStatusResponse, TaskStatus
)
from app.services.segmentation_service import SegmentationService
from app.services.file_service import FileService
from app.services.task_service import TaskService
from app.services.visualization_service import VisualizationService
from app.api.dependencies import (
    get_segmentation_service, get_file_service, get_task_service
)
from fastapi.responses import FileResponse
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

def run_segmentation_task(task_id: str, file_paths: dict, 
                         segmentation_service: SegmentationService,
                         task_service: TaskService):

    try:
        task_service.update_task(task_id, TaskStatus.PROCESSING, 0.1, 
                               "Starting segmentation...")
        

        task_service.update_task(task_id, progress=0.4, 
                               message="Running model prediction...")
        
        output_path = segmentation_service.predict(file_paths, task_id)
        
        task_service.update_task(task_id, progress=0.7,
                               message="Generating visualizations...")
        
        visualization_service = VisualizationService()
        visualizations = visualization_service.create_all_modality_overlays(
            file_paths, output_path
        )
        
        volume_viz = visualization_service.create_3d_volume_visualization(output_path)
        if volume_viz:
            visualizations['3d_volume'] = volume_viz
        
        task_service.update_task(task_id, TaskStatus.COMPLETED, 1.0,
                               "Segmentation completed successfully",
                               {
                                   "output_path": str(output_path),
                                   "visualizations": visualizations,
                                   "upload_id": task_service.tasks[task_id].get('upload_id')
                               })
        
    except Exception as e:
        logger.error(f"Segmentation task {task_id} failed: {e}")
        task_service.update_task(task_id, TaskStatus.FAILED, 
                               message=f"Segmentation failed: {str(e)}")

@router.post("/predict", response_model=SegmentationResponse)
async def predict_segmentation(
    request: SegmentationRequest,
    background_tasks: BackgroundTasks,
    segmentation_service: SegmentationService = Depends(get_segmentation_service),
    file_service: FileService = Depends(get_file_service),
    task_service: TaskService = Depends(get_task_service)
):
    
    try:

        file_paths = file_service.get_upload_files(request.upload_id)
        if not file_paths:
            raise HTTPException(status_code=404, 
                              detail="Upload not found or incomplete")
        
        # Create task
        task_id = task_service.create_task("segmentation", 
                                         upload_id=request.upload_id)
        

        background_tasks.add_task(
            run_segmentation_task, task_id, file_paths,
            segmentation_service, task_service
        )
        
        return SegmentationResponse(
            task_id=task_id,
            status=TaskStatus.PENDING,
            message="Segmentation task started"
        )
        
    except Exception as e:
        logger.error(f"Failed to start segmentation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status/{task_id}", response_model=TaskStatusResponse)
async def get_segmentation_status(
    task_id: str,
    task_service: TaskService = Depends(get_task_service)
):

    task = task_service.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return task

@router.get("/download/{task_id}")
async def download_segmentation(
    task_id: str,
    task_service: TaskService = Depends(get_task_service)
):

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
        filename=f"segmentation_{task_id}.nii.gz",
        media_type="application/gzip"
    )

@router.get("/visualizations/{task_id}")
async def get_segmentation_visualizations(
    task_id: str,
    task_service: TaskService = Depends(get_task_service)
):

    task = task_service.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    if task.status != TaskStatus.COMPLETED:
        raise HTTPException(status_code=400, 
                          detail="Task not completed yet")
    
    visualizations = task.result.get('visualizations', {}) if task.result else {}
    
    return {
        "task_id": task_id,
        "visualizations": visualizations
    }