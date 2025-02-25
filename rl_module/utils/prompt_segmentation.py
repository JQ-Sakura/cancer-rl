import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple, Dict, Optional
from scipy.ndimage import distance_transform_edt

class PromptSegmentation:
    """Prompt-based segmentation utility"""
    
    def __init__(
        self,
        distance_threshold: float = 0.2,
        temperature: float = 0.1,
        min_prob: float = 0.1,
        device: str = 'cuda'
    ):
        """
        Initialize prompt segmentation
        
        Args:
            distance_threshold: Distance threshold for prompt influence
            temperature: Temperature for softmax
            min_prob: Minimum probability for segmentation
            device: Device to run on
        """
        self.distance_threshold = distance_threshold
        self.temperature = temperature
        self.min_prob = min_prob
        self.device = device
        
    def generate_prompt_mask(
        self,
        volume_shape: Tuple[int, int, int],
        gt_mask: np.ndarray,
        num_points: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate prompt points and mask
        
        Args:
            volume_shape: Shape of the volume
            gt_mask: Ground truth mask
            num_points: Number of prompt points to generate
            
        Returns:
            prompt_points: Array of prompt points
            prompt_mask: Distance-based prompt mask
        """
        # 在真实标签中随机选择点
        valid_points = np.argwhere(gt_mask > 0)
        if len(valid_points) == 0:
            return None, np.zeros(volume_shape)
            
        # 随机选择指定数量的点
        selected_indices = np.random.choice(len(valid_points), size=num_points, replace=False)
        prompt_points = valid_points[selected_indices]
        
        # 生成距离图
        prompt_mask = np.zeros(volume_shape)
        for point in prompt_points:
            prompt_mask[tuple(point)] = 1
            
        # 计算距离变换
        distance_map = distance_transform_edt(1 - prompt_mask)
        
        # 将距离转换为概率
        max_distance = np.max(distance_map)
        distance_map = distance_map / max_distance
        prompt_mask = np.exp(-distance_map / self.temperature)
        
        # 应用阈值
        prompt_mask[distance_map > self.distance_threshold] = 0
        prompt_mask[prompt_mask < self.min_prob] = 0
        
        return prompt_points, prompt_mask
    
    def apply_prompt_guidance(
        self,
        features: torch.Tensor,
        prompt_mask: np.ndarray,
        alpha: float = 0.5
    ) -> torch.Tensor:
        """
        Apply prompt guidance to features
        
        Args:
            features: Feature tensor from model
            prompt_mask: Prompt mask
            alpha: Mixing coefficient
            
        Returns:
            Guided features
        """
        # 将prompt_mask转换为tensor
        prompt_tensor = torch.FloatTensor(prompt_mask).to(self.device)
        
        # 扩展维度以匹配特征图
        while len(prompt_tensor.shape) < len(features.shape):
            prompt_tensor = prompt_tensor.unsqueeze(0)
        prompt_tensor = prompt_tensor.expand_as(features)
        
        # 混合特征和prompt guidance
        guided_features = features * (1 + alpha * prompt_tensor)
        
        return guided_features
    
    def get_prompt_loss(
        self,
        pred_mask: torch.Tensor,
        prompt_mask: np.ndarray,
        weight: float = 0.1
    ) -> torch.Tensor:
        """
        Calculate prompt-based loss
        
        Args:
            pred_mask: Predicted segmentation mask
            prompt_mask: Prompt mask
            weight: Loss weight
            
        Returns:
            Prompt loss
        """
        # 确保输入是正确的tensor格式
        if isinstance(pred_mask, np.ndarray):
            pred_mask = torch.FloatTensor(pred_mask)
        if not isinstance(pred_mask, torch.Tensor):
            pred_mask = torch.FloatTensor([pred_mask])
            
        prompt_tensor = torch.FloatTensor(prompt_mask)
        
        # 确保维度匹配
        if len(pred_mask.shape) != len(prompt_tensor.shape):
            if len(pred_mask.shape) > len(prompt_tensor.shape):
                prompt_tensor = prompt_tensor.unsqueeze(0)
            else:
                pred_mask = pred_mask.unsqueeze(0)
                
        # 移动到正确的设备
        pred_mask = pred_mask.to(self.device)
        prompt_tensor = prompt_tensor.to(self.device)
        
        # 计算预测掩码在prompt区域的准确性
        prompt_loss = F.binary_cross_entropy_with_logits(
            pred_mask.float(),
            prompt_tensor.float(),
            reduction='none'
        )
        
        # 加权prompt区域
        prompt_loss = prompt_loss * prompt_tensor
        
        return weight * prompt_loss.mean() 