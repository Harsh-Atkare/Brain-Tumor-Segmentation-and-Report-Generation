import numpy as np
import nibabel as nib
from pathlib import Path

class PostProcessor:
    @staticmethod
    def save_segmentation_mask(pred_mask: np.ndarray, original_shape: tuple, 
                             reference_nifti, output_path: Path):
        pred_mask_transposed = pred_mask.transpose(1, 2, 0)

        full_size_mask = np.zeros(original_shape, dtype=np.uint8)
        full_size_mask[56:184, 56:184, 13:141] = pred_mask_transposed
        affine = reference_nifti.affine
        output_nifti = nib.Nifti1Image(full_size_mask, affine)
        nib.save(output_nifti, output_path)
        
        return output_path
