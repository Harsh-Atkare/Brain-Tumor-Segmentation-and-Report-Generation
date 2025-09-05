import numpy as np
import nibabel as nib
import torch
from sklearn.preprocessing import MinMaxScaler
import torchvision.transforms as transforms
from pathlib import Path

class ImagePreprocessor:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.normalizer = transforms.Normalize(mean=[0.5], std=[0.5])
    
    def load_nifti(self, file_path: Path) -> np.ndarray:
        """Load NIfTI file and return data array."""
        nifti_img = nib.load(file_path)
        return nifti_img.get_fdata(), nifti_img
    
    def preprocess_modality(self, img_data: np.ndarray) -> np.ndarray:
        """Preprocess a single modality using MinMax scaling."""
        img_flat = img_data.reshape(-1, img_data.shape[-1])
        img_scaled = self.scaler.fit_transform(img_flat)
        return img_scaled.reshape(img_data.shape)
    
    def preprocess_input(self, flair_path: Path, t1ce_path: Path, t2_path: Path):
        flair_img, flair_nifti = self.load_nifti(flair_path)
        t1ce_img, _ = self.load_nifti(t1ce_path)
        t2_img, _ = self.load_nifti(t2_path)
        
        # Scale each modality
        flair_scaled = self.preprocess_modality(flair_img)
        t1ce_scaled = self.preprocess_modality(t1ce_img)
        t2_scaled = self.preprocess_modality(t2_img)
        
        # Stack modalities
        combined_img = np.stack([flair_scaled, t1ce_scaled, t2_scaled], axis=3)
        
        # Crop to expected size
        cropped_img = combined_img[56:184, 56:184, 13:141]
        
        # Convert to tensor
        img_tensor = torch.from_numpy(cropped_img).permute(3, 2, 0, 1)
        img_tensor = self.normalizer(img_tensor.float())
        img_tensor = img_tensor.unsqueeze(0)
        
        return img_tensor, flair_img.shape, flair_nifti