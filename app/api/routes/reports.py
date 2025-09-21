from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from app.models.schemas import (
    ReportGenerationRequest, ReportGenerationResponse, 
    TaskStatusResponse, TaskStatus
)
from app.services.report_service import ReportService
from app.services.file_service import FileService
from app.services.task_service import TaskService
from app.api.dependencies import get_report_service, get_task_service, get_file_service
from fastapi.responses import FileResponse
from app.core.config import settings
from pathlib import Path
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

def run_report_generation_task(task_id: str, features_task_id: str, patient_info: dict,
                              report_service: ReportService, task_service: TaskService,
                              file_service: FileService):
    """Background task for comprehensive report generation."""
    try:
        task_service.update_task(task_id, TaskStatus.PROCESSING, 0.1,
                               "Starting comprehensive report generation...")
        
        # Get features task
        features_task = task_service.get_task(features_task_id)
        if not features_task or features_task.status != TaskStatus.COMPLETED:
            raise Exception("Feature extraction task not completed")
        
        if not features_task.result or 'features' not in features_task.result:
            raise Exception("No features found in task result")
        
        features = features_task.result['features']
        
        segmentation_task_id = features_task.result.get('segmentation_task_id') or task_service.tasks[features_task_id].get('segmentation_task_id')
        segmentation_task = task_service.get_task(segmentation_task_id) if segmentation_task_id else None

        file_paths = None
        segmentation_path = None
        
        if segmentation_task and segmentation_task.result:
            upload_id = segmentation_task.result.get('upload_id') or task_service.tasks[segmentation_task_id].get('upload_id')
            if upload_id:
                file_paths = file_service.get_upload_files(upload_id)
            segmentation_path = Path(segmentation_task.result['output_path']) if 'output_path' in segmentation_task.result else None
        
        task_service.update_task(task_id, progress=0.3,
                               message="Generating AI report with visualizations...")

        report_data = report_service.generate_report(
            features, patient_info, task_id, file_paths, segmentation_path
        )
        
        task_service.update_task(task_id, progress=0.7,
                               message="Creating enhanced PDF report...")

        pdf_path = settings.REPORTS_DIR / f"{task_id}_comprehensive_report.pdf"
        report_service.generate_pdf_report(report_data, pdf_path)
        
        task_service.update_task(task_id, TaskStatus.COMPLETED, 1.0,
                               "Comprehensive report generation completed successfully",
                               {
                                   "report_data": report_data,
                                   "pdf_path": str(pdf_path)
                               })
        
    except Exception as e:
        logger.error(f"Report generation task {task_id} failed: {e}")
        task_service.update_task(task_id, TaskStatus.FAILED,
                               message=f"Report generation failed: {str(e)}")

@router.post("/generate", response_model=ReportGenerationResponse)
async def generate_report(
    request: ReportGenerationRequest,
    background_tasks: BackgroundTasks,
    report_service: ReportService = Depends(get_report_service),
    task_service: TaskService = Depends(get_task_service),
    file_service: FileService = Depends(get_file_service)
):

    try:

        features_task = task_service.get_task(request.features_task_id)
        if not features_task:
            raise HTTPException(status_code=404, 
                              detail="Features task not found")
        
        if features_task.status != TaskStatus.COMPLETED:
            raise HTTPException(status_code=400,
                              detail="Feature extraction not completed yet")

        task_id = task_service.create_task("report_generation",
                                         features_task_id=request.features_task_id,
                                         patient_info=request.patient_info)

        background_tasks.add_task(
            run_report_generation_task, task_id, request.features_task_id,
            request.patient_info or {}, report_service, task_service, file_service
        )
        
        return ReportGenerationResponse(
            task_id=task_id,
            status=TaskStatus.PENDING,
            message="Comprehensive report generation task started"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start report generation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status/{task_id}", response_model=TaskStatusResponse)
async def get_report_status(
    task_id: str,
    task_service: TaskService = Depends(get_task_service)
):

    task = task_service.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return task

@router.get("/download/{task_id}")
async def download_report(
    task_id: str,
    task_service: TaskService = Depends(get_task_service)
):

    task = task_service.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    if task.status != TaskStatus.COMPLETED:
        raise HTTPException(status_code=400,
                          detail="Task not completed yet")
    
    if not task.result or 'pdf_path' not in task.result:
        raise HTTPException(status_code=500,
                          detail="No report file available")
    
    pdf_path = task.result['pdf_path']
    return FileResponse(
        path=pdf_path,
        filename=f"brain_tumor_comprehensive_report_{task_id}.pdf",
        media_type="application/pdf"
    )

@router.get("/preview/{task_id}")
async def get_report_preview(
    task_id: str,
    task_service: TaskService = Depends(get_task_service)
):

    task = task_service.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    if task.status != TaskStatus.COMPLETED:
        raise HTTPException(status_code=400,
                          detail="Task not completed yet")
    
    report_data = task.result.get('report_data', {}) if task.result else {}
    
    return {
        "task_id": task_id,
        "report_text": report_data.get('report_text', ''),
        "visualizations": report_data.get('visualizations', {}),
        "features": report_data.get('features', {}),
        "generated_at": report_data.get('generated_at', ''),
        "model_used": report_data.get('model_used', '')
    }