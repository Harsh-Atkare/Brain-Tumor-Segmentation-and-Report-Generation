# app/utils/postprocessing.py
import numpy as np
import nibabel as nib
from pathlib import Path

class PostProcessor:
    @staticmethod
    def save_segmentation_mask(pred_mask: np.ndarray, original_shape: tuple, 
                             reference_nifti, output_path: Path):
        """
        Save segmentation mask as NIfTI file.
        
        Args:
            pred_mask: Predicted segmentation mask
            original_shape: Original image shape
            reference_nifti: Reference NIfTI image for affine matrix
            output_path: Output file path
        """
        # Transpose mask to original orientation
        pred_mask_transposed = pred_mask.transpose(1, 2, 0)
        
        # Create full-size mask
        full_size_mask = np.zeros(original_shape, dtype=np.uint8)
        full_size_mask[56:184, 56:184, 13:141] = pred_mask_transposed
        
        # Create NIfTI image with original affine
        affine = reference_nifti.affine
        output_nifti = nib.Nifti1Image(full_size_mask, affine)
        
        # Save
        nib.save(output_nifti, output_path)
        
        return output_path
