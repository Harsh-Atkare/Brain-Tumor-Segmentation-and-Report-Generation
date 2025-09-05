from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from datetime import datetime
from enum import Enum

class TaskStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class FileUploadResponse(BaseModel):
    upload_id: str
    message: str
    files_received: Dict[str, str]

class SegmentationRequest(BaseModel):
    upload_id: str

class SegmentationResponse(BaseModel):
    task_id: str
    status: TaskStatus
    message: str

class TaskStatusResponse(BaseModel):
    task_id: str
    status: TaskStatus
    progress: Optional[float] = None
    message: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    created_at: datetime
    updated_at: datetime

class FeatureExtractionRequest(BaseModel):
    task_id: str 

class FeatureExtractionResponse(BaseModel):
    task_id: str
    status: TaskStatus
    message: str

class ClinicalFeatures(BaseModel):
    case_id: str
    voxel_spacing_mm: str
    whole_tumor_volume_cm3: float
    whole_tumor_diameter_mm: float
    tumor_core_volume_cm3: float
    tumor_core_diameter_mm: float
    enhancing_volume_cm3: float
    enhancing_diameter_mm: float
    non_enhancing_volume_cm3: float
    necrotic_volume_cm3: float
    edema_volume_cm3: float
    enhancing_percentage: float
    necrotic_percentage: float
    edema_percentage: float
    hemisphere: str
    anatomical_location: str
    centroid_coordinates: str
    enhancement_mean_intensity: float
    enhancement_max_intensity: float
    has_enhancement: str
    has_necrosis: str
    has_edema: str
    tumor_size_category: str
    enhancement_pattern: str
    necrosis_extent: str