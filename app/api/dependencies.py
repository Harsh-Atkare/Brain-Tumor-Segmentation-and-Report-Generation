from fastapi import Depends, HTTPException, status
from app.services.file_service import FileService
from app.services.segmentation_service import SegmentationService
from app.services.feature_extraction_service import FeatureExtractionService

from app.services.report_service import ReportService

from app.services.task_service import task_service, TaskService

def get_file_service() -> FileService:
    return FileService()

def get_segmentation_service() -> SegmentationService:
    return SegmentationService()

def get_feature_extraction_service() -> FeatureExtractionService:
    return FeatureExtractionService()

def get_report_service() -> ReportService:
    return ReportService()

def get_task_service() -> TaskService:
    return task_service