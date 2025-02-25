import numpy as np
import nibabel as nib
from typing import Tuple, List, Optional
from scipy.ndimage import zoom

def load_nifti(file_path: str) -> np.ndarray:
    """
    Load a NIfTI file and return its data as a numpy array
    
    Args:
        file_path: Path to the .nii.gz file
        
    Returns:
        Numpy array containing the image data
    """
    try:
        nifti_img = nib.load(file_path)
        return nifti_img.get_fdata()
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def normalize_volume(volume: np.ndarray) -> np.ndarray:
    """
    Normalize the volume to [0,1] range
    
    Args:
        volume: Input volume data
        
    Returns:
        Normalized volume
    """
    min_val = np.min(volume)
    max_val = np.max(volume)
    if max_val - min_val == 0:
        return volume
    return (volume - min_val) / (max_val - min_val)

def load_patient_data(patient_dir: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Load all modalities and cancer mask for a patient
    
    Args:
        patient_dir: Directory containing patient's data
        
    Returns:
        Tuple of (modalities_data, cancer_mask)
        modalities_data: (3, H, W, D) array containing ADC, DWI, and T2
        cancer_mask: (H, W, D) binary mask or None if healthy
    """
    # Load modalities
    modalities = []
    shapes = []
    for modality in ['adc.nii.gz', 'dwi.nii.gz', 't2.nii.gz']:
        data = load_nifti(f"{patient_dir}/{modality}")
        if data is None:
            return None, None
        shapes.append(data.shape)
        modalities.append(normalize_volume(data))
    
    # Ensure all modalities have the same shape
    if len(set(str(shape) for shape in shapes)) > 1:
        # Resample all images to the smallest shape
        min_shape = np.min(shapes, axis=0)
        resampled_modalities = []
        for img in modalities:
            if img.shape != tuple(min_shape):
                # 使用最近邻插值进行重采样
                scale_factors = min_shape / np.array(img.shape)
                resampled_img = zoom(img, scale_factors, order=0)
                resampled_modalities.append(resampled_img)
            else:
                resampled_modalities.append(img)
        modalities = resampled_modalities
    
    # Load cancer mask
    try:
        cancer_mask = load_nifti(f"{patient_dir}/l_a1.nii.gz")
        if cancer_mask is not None:
            cancer_mask = (cancer_mask > 0.5).astype(np.float32)
            # Ensure cancer mask has the same shape as modalities
            if cancer_mask.shape != modalities[0].shape:
                scale_factors = np.array(modalities[0].shape) / np.array(cancer_mask.shape)
                cancer_mask = zoom(cancer_mask, scale_factors, order=0)  # order=0 for nearest neighbor interpolation
    except:
        # 健康患者，没有癌症掩码
        cancer_mask = None
    
    # Stack modalities
    modalities_data = np.stack(modalities, axis=0)
    
    return modalities_data, cancer_mask

def calculate_entropy(predictions: np.ndarray) -> np.ndarray:
    """
    Calculate entropy for each voxel based on model predictions
    
    Args:
        predictions: Model predictions with shape (N, C, H, W, D)
        where C is number of classes (usually 2 for binary)
        
    Returns:
        Entropy map with shape (H, W, D)
    """
    # Ensure predictions are probabilities
    predictions = np.clip(predictions, 1e-10, 1.0)
    
    # Calculate entropy
    entropy = -np.sum(predictions * np.log(predictions), axis=1)
    return entropy

def create_region_mask(
    center: Tuple[int, int, int],
    volume_shape: Tuple[int, int, int],
    radius: int = 3
) -> np.ndarray:
    """
    Create a spherical mask around a center point
    
    Args:
        center: (z, y, x) coordinates of center point
        volume_shape: Shape of the volume
        radius: Radius of the sphere
        
    Returns:
        Binary mask of the region
    """
    z, y, x = np.ogrid[-center[0]:volume_shape[0]-center[0],
                       -center[1]:volume_shape[1]-center[1],
                       -center[2]:volume_shape[2]-center[2]]
    mask = (z*z + y*y + x*x) <= radius*radius
    return mask.astype(np.float32) 