# app/services/segmentation_service.py
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
from app.models.unet3d import UNet3D
from app.utils.preprocessing import ImagePreprocessor
from app.utils.postprocessing import PostProcessor
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)

class SegmentationService:
    def __init__(self):
        self.device = torch.device(settings.DEVICE)
        self.model = None
        self.preprocessor = ImagePreprocessor()
        self.postprocessor = PostProcessor()
        self._load_model()
    
    def _load_model(self):

        try:
            self.model = UNet3D(in_channels=3, out_channels=4)
            checkpoint = torch.load(settings.MODEL_PATH, map_location=self.device)
            self.model.load_state_dict(checkpoint['model'])
            self.model.to(self.device)
            self.model.eval()
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def predict(self, file_paths: Dict[str, Path], task_id: str) -> Path:
        try:
   
            input_tensor, original_shape, reference_nifti = self.preprocessor.preprocess_input(
                file_paths['flair'], file_paths['t1ce'], file_paths['t2']
            )

            with torch.no_grad():
                input_tensor = input_tensor.to(self.device)
                output_logits = self.model(input_tensor)
                pred_mask = torch.argmax(output_logits, dim=1)
                pred_mask_np = pred_mask.squeeze(0).cpu().numpy()

            output_path = settings.OUTPUT_DIR / f"{task_id}_segmentation.nii.gz"
            self.postprocessor.save_segmentation_mask(
                pred_mask_np, original_shape, reference_nifti, output_path
            )
            
            logger.info(f"Segmentation completed for task {task_id}")
            return output_path
            
        except Exception as e:
            logger.error(f"Segmentation failed for task {task_id}: {e}")
            raise
