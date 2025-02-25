import os
import numpy as np
import nibabel as nib
from typing import Tuple, Optional, Dict
from scipy.ndimage import zoom
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MRIDataset:
    """Dataset class for loading MRI data and masks"""
    
    def __init__(self, data_dir: str):
        """
        Initialize dataset
        
        Args:
            data_dir: Directory containing patient data
        """
        self.data_dir = data_dir
        self.patient_dirs = []
        self.patient_status = {}  # 记录每个患者的状态（健康/患病）
        
        # 扫描数据目录
        for d in os.listdir(data_dir):
            patient_dir = os.path.join(data_dir, d)
            if not os.path.isdir(patient_dir):
                continue
                
            # 检查是否所有必需的模态都存在
            required_modalities = ['adc.nii.gz', 'dwi.nii.gz', 't2.nii.gz']
            has_all_modalities = all(
                os.path.exists(os.path.join(patient_dir, modality))
                for modality in required_modalities
            )
            
            if has_all_modalities:
                self.patient_dirs.append(d)
                # 检查是否存在病灶标记
                has_cancer = os.path.exists(os.path.join(patient_dir, 'l_a1.nii.gz'))
                self.patient_status[d] = 'diseased' if has_cancer else 'healthy'
        
        # 记录数据集统计信息
        num_diseased = sum(1 for status in self.patient_status.values() if status == 'diseased')
        num_healthy = len(self.patient_status) - num_diseased
        
        logger.info(f"Found {len(self.patient_dirs)} valid patients")
        logger.info(f"Diseased patients: {num_diseased}")
        logger.info(f"Healthy patients: {num_healthy}")
        
    def _load_and_preprocess(self, patient_id: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Load and preprocess patient data
        
        Args:
            patient_id: Patient directory name
            
        Returns:
            Tuple of (volume, mask)
            volume: Array of shape (3, D, H, W) containing the three modalities
            mask: Array of shape (D, H, W) containing the cancer mask, or None for healthy patients
        """
        patient_dir = os.path.join(self.data_dir, patient_id)
        
        # 加载所有模态
        modalities = []
        shapes = []
        for modality in ['adc.nii.gz', 'dwi.nii.gz', 't2.nii.gz']:
            try:
                img_path = os.path.join(patient_dir, modality)
                img = nib.load(img_path).get_fdata()
                shapes.append(img.shape)
                modalities.append(img)
            except Exception as e:
                logger.error(f"Error loading {modality} for patient {patient_id}: {str(e)}")
                raise
        
        # 确保所有模态具有相同的形状
        if len(set(str(shape) for shape in shapes)) > 1:
            # 将所有图像重采样到最小形状
            min_shape = np.min(shapes, axis=0)
            resampled_modalities = []
            for img in modalities:
                if img.shape != tuple(min_shape):
                    scale_factors = min_shape / np.array(img.shape)
                    resampled = zoom(img, scale_factors, order=1)
                    resampled_modalities.append(resampled)
                else:
                    resampled_modalities.append(img)
            modalities = resampled_modalities
        
        # 标准化每个模态
        normalized_modalities = []
        for mod in modalities:
            # 使用鲁棒的标准化方法
            p1, p99 = np.percentile(mod, (1, 99))
            mod_norm = np.clip(mod, p1, p99)
            mod_norm = (mod_norm - mod_norm.mean()) / (mod_norm.std() + 1e-8)
            normalized_modalities.append(mod_norm)
        
        # 堆叠模态
        volume = np.stack(normalized_modalities, axis=0)
        
        # 加载mask（如果存在）
        mask = None
        if self.patient_status[patient_id] == 'diseased':
            try:
                mask_path = os.path.join(patient_dir, 'l_a1.nii.gz')
                mask = nib.load(mask_path).get_fdata()
                
                # 如果mask形状与volume不匹配，进行重采样
                if mask.shape != volume.shape[1:]:
                    scale_factors = np.array(volume.shape[1:]) / np.array(mask.shape)
                    mask = zoom(mask, scale_factors, order=0)  # order=0 for nearest neighbor
                
                # 确保mask是二值的
                mask = (mask > 0.5).astype(np.float32)
                
            except Exception as e:
                logger.error(f"Error loading mask for patient {patient_id}: {str(e)}")
                # 对于加载失败的mask，返回全零mask
                mask = np.zeros(volume.shape[1:], dtype=np.float32)
        else:
            # 对于健康患者，返回全零mask
            mask = np.zeros(volume.shape[1:], dtype=np.float32)
            
        return volume, mask
        
    def get_random_patient(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        随机获取一个患者的数据
        
        Returns:
            Tuple of (volume, mask)
        """
        patient_id = np.random.choice(self.patient_dirs)
        try:
            return self._load_and_preprocess(patient_id)
        except Exception as e:
            logger.error(f"Error loading patient {patient_id}: {str(e)}")
            # 如果加载失败，递归尝试另一个患者
            return self.get_random_patient()
            
    def get_patient_by_id(self, patient_id: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取指定患者的数据
        
        Args:
            patient_id: Patient directory name
            
        Returns:
            Tuple of (volume, mask)
        """
        if patient_id not in self.patient_dirs:
            raise ValueError(f"Patient {patient_id} not found in dataset")
            
        return self._load_and_preprocess(patient_id)
        
    def get_patient_status(self, patient_id: str) -> str:
        """
        获取患者的状态
        
        Args:
            patient_id: Patient directory name
            
        Returns:
            'diseased' or 'healthy'
        """
        return self.patient_status.get(patient_id, 'unknown')
        
    def __len__(self) -> int:
        """获取数据集中的患者数量"""
        return len(self.patient_dirs) 