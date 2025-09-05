import numpy as np
import nibabel as nib
from scipy import ndimage
from skimage import measure
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class FeatureExtractionService:
    def __init__(self):
        pass
    
    def extract_features(self, file_paths: Dict[str, Path], 
                        segmentation_path: Path, case_id: str) -> Dict[str, Any]:
        try:
  
            t1_img = nib.load(file_paths['flair']).get_fdata() 
            t1ce_img = nib.load(file_paths['t1ce']).get_fdata()
            seg_img = nib.load(segmentation_path).get_fdata()

            header = nib.load(file_paths['flair']).header
            voxel_size = header.get_zooms()
            
            enhancing_tumor = (seg_img == 3).astype(int)
            necrotic_core = (seg_img == 1).astype(int)
            peritumoral_edema = (seg_img == 2).astype(int)
            tumor_core = ((seg_img == 1) | (seg_img == 3)).astype(int)
            whole_tumor = (seg_img > 0).astype(int)
            
            whole_voxels, whole_mm3 = self._calculate_volume_mm3(whole_tumor, voxel_size)
            core_voxels, core_mm3 = self._calculate_volume_mm3(tumor_core, voxel_size)
            enhancing_voxels, enhancing_mm3 = self._calculate_volume_mm3(enhancing_tumor, voxel_size)
            necrotic_voxels, necrotic_mm3 = self._calculate_volume_mm3(necrotic_core, voxel_size)
            edema_voxels, edema_mm3 = self._calculate_volume_mm3(peritumoral_edema, voxel_size)
            whole_diameter = self._calculate_max_diameter(whole_tumor, voxel_size)
            core_diameter = self._calculate_max_diameter(tumor_core, voxel_size)
            enhancing_diameter = self._calculate_max_diameter(enhancing_tumor, voxel_size)
            hemisphere, location, cent_x, cent_y, cent_z = self._get_location_info(whole_tumor)
            enhancing_ratio = enhancing_mm3 / whole_mm3 if whole_mm3 > 0 else 0
            necrotic_ratio = necrotic_mm3 / whole_mm3 if whole_mm3 > 0 else 0
            edema_ratio = edema_mm3 / whole_mm3 if whole_mm3 > 0 else 0
            if enhancing_voxels > 0:
                t1ce_enhancing = t1ce_img[enhancing_tumor > 0]
                enhancement_mean = np.mean(t1ce_enhancing)
                enhancement_max = np.max(t1ce_enhancing)
            else:
                enhancement_mean = 0
                enhancement_max = 0
            
            features = {
                'case_id': case_id,
                'voxel_spacing_mm': f"{voxel_size[0]:.1f}x{voxel_size[1]:.1f}x{voxel_size[2]:.1f}",
                'whole_tumor_volume_cm3': whole_mm3 / 1000,
                'whole_tumor_diameter_mm': whole_diameter,
                'tumor_core_volume_cm3': core_mm3 / 1000,
                'tumor_core_diameter_mm': core_diameter,
                'enhancing_volume_cm3': enhancing_mm3 / 1000,
                'enhancing_diameter_mm': enhancing_diameter,
                'non_enhancing_volume_cm3': necrotic_mm3 / 1000,
                'necrotic_volume_cm3': necrotic_mm3 / 1000,
                'edema_volume_cm3': edema_mm3 / 1000,
                'enhancing_percentage': enhancing_ratio * 100,
                'necrotic_percentage': necrotic_ratio * 100,
                'edema_percentage': edema_ratio * 100,
                'hemisphere': hemisphere,
                'anatomical_location': location,
                'centroid_coordinates': f"({cent_x:.0f}, {cent_y:.0f}, {cent_z:.0f})",
                'enhancement_mean_intensity': enhancement_mean,
                'enhancement_max_intensity': enhancement_max,
                'has_enhancement': 'yes' if enhancing_voxels > 0 else 'no',
                'has_necrosis': 'yes' if necrotic_voxels > 0 else 'no',
                'has_edema': 'yes' if edema_voxels > 0 else 'no',
                'tumor_size_category': self._categorize_size(whole_mm3),
                'enhancement_pattern': self._categorize_enhancement(enhancing_ratio),
                'necrosis_extent': self._categorize_necrosis(necrotic_ratio)
            }
            
            logger.info(f"Features extracted successfully for case {case_id}")
            return features
            
        except Exception as e:
            logger.error(f"Feature extraction failed for case {case_id}: {e}")
            raise
    
    def _calculate_volume_mm3(self, mask, voxel_size):
        """Calculate volume in voxels and mm³."""
        volume_voxels = np.sum(mask)
        volume_mm3 = volume_voxels * np.prod(voxel_size)
        return volume_voxels, volume_mm3
    
    def _calculate_max_diameter(self, mask, voxel_size):
        """Calculate maximum diameter of region."""
        if np.sum(mask) == 0:
            return 0
        props = measure.regionprops(mask.astype(int))[0]
        bbox = props.bbox
        bbox_dims = np.array([bbox[3]-bbox[0], bbox[4]-bbox[1], bbox[5]-bbox[2]]) * voxel_size
        return np.max(bbox_dims)
    
    def _get_location_info(self, mask):
        """Get anatomical location information."""
        if np.sum(mask) == 0:
            return 'none', 'none', 0, 0, 0
        props = measure.regionprops(mask.astype(int))[0]
        centroid = np.array(props.centroid)
        hemisphere = 'left' if centroid[0] < mask.shape[0]/2 else 'right'
        z_rel = centroid[2] / mask.shape[2]
        y_rel = centroid[1] / mask.shape[1]
        if z_rel < 0.3:
            location = 'inferior'
        elif z_rel > 0.7:
            location = 'superior'
        elif y_rel < 0.4:
            location = 'posterior'
        elif y_rel > 0.6:
            location = 'anterior'
        else:
            location = 'central'
        return hemisphere, location, centroid[0], centroid[1], centroid[2]
    
    def _categorize_size(self, volume_mm3):
        """Categorize tumor size."""
        volume_cm3 = volume_mm3 / 1000
        if volume_cm3 < 1:
            return 'small (<1 cm³)'
        elif volume_cm3 < 5:
            return 'medium (1-5 cm³)'
        elif volume_cm3 < 15:
            return 'large (5-15 cm³)'
        else:
            return 'very_large (>15 cm³)'
    
    def _categorize_enhancement(self, enhancing_ratio):
        """Categorize enhancement pattern."""
        if enhancing_ratio == 0:
            return 'none'
        elif enhancing_ratio < 0.1:
            return 'minimal (<10%)'
        elif enhancing_ratio < 0.3:
            return 'moderate (10-30%)'
        elif enhancing_ratio < 0.7:
            return 'significant (30-70%)'
        else:
            return 'extensive (>70%)'
    
    def _categorize_necrosis(self, necrotic_ratio):
        """Categorize necrosis extent."""
        if necrotic_ratio == 0:
            return 'none'
        elif necrotic_ratio < 0.1:
            return 'minimal (<10%)'
        elif necrotic_ratio < 0.3:
            return 'moderate (10-30%)'
        else:
            return 'extensive (>30%)'
    
    def save_features_to_csv(self, features: Dict[str, Any], output_path: Path) -> Path:
        """Save features to CSV file."""
        df = pd.DataFrame([features])
        df.to_csv(output_path, index=False)
        return output_path
