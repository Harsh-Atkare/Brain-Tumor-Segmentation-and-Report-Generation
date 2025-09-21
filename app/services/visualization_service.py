# app/services/visualization_service.py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import nibabel as nib
from pathlib import Path
from typing import Dict, List, Tuple
import base64
import io
from scipy import ndimage
import logging

logger = logging.getLogger(__name__)

class VisualizationService:
    def __init__(self):
        self.tumor_colors = {
            0: (0, 0, 0, 0),      
            1: (1, 0, 0, 0.7),    
            2: (0, 1, 0, 0.7),    
            3: (0, 0, 1, 0.7),    
        }
        
        self.tumor_labels = {
            0: "Background",
            1: "Necrotic Core",
            2: "Peritumoral Edema", 
            3: "Enhancing Tumor"
        }
    
    def create_segmentation_overlay(self, original_path: Path, segmentation_path: Path, 
                                  file_type: str, num_slices: int = 5) -> str:

        try:

            original_img = nib.load(original_path).get_fdata()
            seg_img = nib.load(segmentation_path).get_fdata()

            original_normalized = self._normalize_image(original_img)

            tumor_slices = self._find_tumor_slices(seg_img, num_slices)

            fig, axes = plt.subplots(1, num_slices, figsize=(4*num_slices, 6))
            if num_slices == 1:
                axes = [axes]
                
            fig.suptitle(f'{file_type.upper()} with Segmentation Overlay', 
                        fontsize=16, fontweight='bold')
            
            for i, slice_idx in enumerate(tumor_slices):
                ax = axes[i]

                ax.imshow(original_normalized[:, :, slice_idx], 
                         cmap='gray', alpha=0.8)
                
                overlay = np.zeros((*original_normalized.shape[:2], 4))
                
                for label, color in self.tumor_colors.items():
                    if label == 0:
                        continue
                    mask = seg_img[:, :, slice_idx] == label
                    if np.any(mask):
                        overlay[mask] = color

                ax.imshow(overlay, alpha=0.6)
                
                ax.set_title(f'Slice {slice_idx}', fontsize=12)
                ax.axis('off')

            legend_elements = []
            for label, color in self.tumor_colors.items():
                if label == 0:
                    continue
                legend_elements.append(
                    plt.Rectangle((0,0),1,1, facecolor=color[:3], 
                                alpha=color[3], label=self.tumor_labels[label])
                )
            
            if legend_elements:
                fig.legend(handles=legend_elements, loc='lower center', 
                          bbox_to_anchor=(0.5, -0.05), ncol=3)
            
            plt.tight_layout()

            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            buffer.seek(0)
            img_str = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return img_str
            
        except Exception as e:
            logger.error(f"Error creating segmentation overlay for {file_type}: {e}")
            return ""
    
    def create_all_modality_overlays(self, file_paths: Dict[str, Path], 
                                   segmentation_path: Path) -> Dict[str, str]:

        overlays = {}
        
        for modality, path in file_paths.items():
            overlay = self.create_segmentation_overlay(
                path, segmentation_path, modality, num_slices=3
            )
            if overlay:
                overlays[modality] = overlay
        
        return overlays
    
    def create_3d_volume_visualization(self, segmentation_path: Path) -> str:

        try:
            seg_img = nib.load(segmentation_path).get_fdata()
            
            fig = plt.figure(figsize=(12, 10))

            ax1 = plt.subplot(2, 2, 1)
            ax2 = plt.subplot(2, 2, 2) 
            ax3 = plt.subplot(2, 2, 3)
            ax4 = plt.subplot(2, 2, 4)

            sagittal_slice = seg_img.shape[0] // 2
            ax1.imshow(seg_img[sagittal_slice, :, :].T, cmap='viridis', origin='lower')
            ax1.set_title('Sagittal View')
            ax1.axis('off')

            coronal_slice = seg_img.shape[1] // 2
            ax2.imshow(seg_img[:, coronal_slice, :].T, cmap='viridis', origin='lower')
            ax2.set_title('Coronal View')
            ax2.axis('off')

            axial_slice = seg_img.shape[2] // 2  
            ax3.imshow(seg_img[:, :, axial_slice], cmap='viridis')
            ax3.set_title('Axial View')
            ax3.axis('off')

            volumes = {}
            for label in [1, 2, 3]:
                volumes[self.tumor_labels[label]] = np.sum(seg_img == label)
            
            ax4.bar(volumes.keys(), volumes.values(), 
                   color=['red', 'green', 'blue'], alpha=0.7)
            ax4.set_title('Tumor Component Volumes (voxels)')
            ax4.set_ylabel('Volume (voxels)')
            plt.xticks(rotation=45)
            
            plt.suptitle('3D Tumor Segmentation Views', fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            img_str = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return img_str
            
        except Exception as e:
            logger.error(f"Error creating 3D visualization: {e}")
            return ""
    
    def _normalize_image(self, img: np.ndarray) -> np.ndarray:

        img_norm = img.copy().astype(float)
        

        p1, p99 = np.percentile(img_norm[img_norm > 0], [1, 99])
        img_norm = np.clip(img_norm, p1, p99)

        img_norm = (img_norm - p1) / (p99 - p1)
        
        return img_norm
    
    def _find_tumor_slices(self, seg_img: np.ndarray, num_slices: int) -> List[int]:
        slice_tumor_content = []
        for i in range(seg_img.shape[2]):
            tumor_voxels = np.sum(seg_img[:, :, i] > 0)
            slice_tumor_content.append((i, tumor_voxels))
        slice_tumor_content.sort(key=lambda x: x[1], reverse=True)
        top_slices = [x[0] for x in slice_tumor_content[:min(num_slices*2, len(slice_tumor_content))]]
        top_slices.sort()

        if len(top_slices) >= num_slices:
            indices = np.linspace(0, len(top_slices)-1, num_slices, dtype=int)
            selected_slices = [top_slices[i] for i in indices]
        else:
            middle = seg_img.shape[2] // 2
            selected_slices = list(range(middle - num_slices//2, middle + num_slices//2 + 1))
            selected_slices = [max(0, min(seg_img.shape[2]-1, s)) for s in selected_slices]
        
        return selected_slices[:num_slices]