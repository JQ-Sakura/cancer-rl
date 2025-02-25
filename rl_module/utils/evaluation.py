import torch
import numpy as np
from typing import Dict, Optional, List, Tuple
from .post_processing import PostProcessor
import logging
from tqdm import tqdm
import wandb

logger = logging.getLogger(__name__)

class Evaluator:
    """评估器类，包含后处理和指标计算"""
    
    def __init__(
        self,
        post_processor: Optional[PostProcessor] = None,
        metrics: Optional[List[str]] = None,
        device: str = 'cuda'
    ):
        """
        初始化评估器
        
        Args:
            post_processor: 后处理器实例
            metrics: 要计算的指标列表
            device: 设备
        """
        self.post_processor = post_processor or PostProcessor()
        self.metrics = metrics or ['dice', 'iou', 'boundary_dice']
        self.device = device
        
    def evaluate(
        self,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        threshold: float = 0.5
    ) -> Dict[str, float]:
        """
        评估模型性能
        
        Args:
            model: 要评估的模型
            dataloader: 数据加载器
            threshold: 二值化阈值
            
        Returns:
            包含各项指标的字典
        """
        model.eval()
        metrics_sum = {metric: 0.0 for metric in self.metrics}
        metrics_count = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc='Evaluating'):
                # 获取数据
                volumes = batch['volume'].to(self.device)
                masks = batch['mask'].to(self.device)
                
                # 模型预测
                predictions = model(volumes)
                if isinstance(predictions, list):  # 处理深度监督的情况
                    predictions = predictions[0]
                
                # 对每个样本进行处理
                for i in range(volumes.size(0)):
                    pred = predictions[i:i+1]
                    mask = masks[i:i+1]
                    volume = volumes[i:i+1]
                    
                    # 应用后处理
                    processed_pred = self.post_processor(
                        pred,
                        volume,
                        threshold=threshold
                    )
                    
                    # 计算指标
                    batch_metrics = self._calculate_metrics(
                        processed_pred,
                        mask.cpu().numpy(),
                        self.metrics
                    )
                    
                    # 累加指标
                    for metric, value in batch_metrics.items():
                        metrics_sum[metric] += value
                    metrics_count += 1
                    
                    # 记录示例图像（每个epoch的第一个batch的第一个样本）
                    if metrics_count == 1 and wandb.run is not None:
                        self._log_example(volume, mask, processed_pred)
        
        # 计算平均值
        metrics_avg = {
            metric: metrics_sum[metric] / metrics_count
            for metric in self.metrics
        }
        
        return metrics_avg
        
    def _calculate_metrics(
        self,
        pred: np.ndarray,
        target: np.ndarray,
        metrics: List[str]
    ) -> Dict[str, float]:
        """计算各项指标"""
        results = {}
        
        for metric in metrics:
            if metric == 'dice':
                score = self._calculate_dice(pred, target)
            elif metric == 'iou':
                score = self._calculate_iou(pred, target)
            elif metric == 'boundary_dice':
                score = self._calculate_boundary_dice(pred, target)
            else:
                logger.warning(f"Unknown metric: {metric}")
                continue
                
            results[metric] = score
            
        # 添加边界相关指标
        boundary_metrics = self.post_processor.calculate_boundary_metrics(
            pred,
            target
        )
        results.update(boundary_metrics)
        
        # 添加区域分析
        region_analysis = self.post_processor.analyze_regions(pred)
        results['num_regions'] = region_analysis['num_regions']
        results['avg_region_area'] = region_analysis['avg_area']
        
        return results
        
    def _calculate_dice(self, pred: np.ndarray, target: np.ndarray) -> float:
        """计算Dice系数"""
        smooth = 1e-5
        intersection = np.sum(pred * target)
        union = np.sum(pred) + np.sum(target)
        return (2.0 * intersection + smooth) / (union + smooth)
        
    def _calculate_iou(self, pred: np.ndarray, target: np.ndarray) -> float:
        """计算IoU"""
        smooth = 1e-5
        intersection = np.sum(pred * target)
        union = np.sum(pred) + np.sum(target) - intersection
        return (intersection + smooth) / (union + smooth)
        
    def _calculate_boundary_dice(
        self,
        pred: np.ndarray,
        target: np.ndarray,
        tolerance: int = 2
    ) -> float:
        """计算边界Dice系数"""
        from scipy.ndimage import distance_transform_edt
        
        # 提取边界
        pred_boundary = self._extract_boundary(pred)
        target_boundary = self._extract_boundary(target)
        
        # 计算距离变换
        pred_dist = distance_transform_edt(~pred_boundary)
        target_dist = distance_transform_edt(~target_boundary)
        
        # 在容差范围内的边界点
        pred_mask = pred_dist <= tolerance
        target_mask = target_dist <= tolerance
        
        # 计算Dice
        intersection = np.sum(pred_mask * target_mask)
        union = np.sum(pred_mask) + np.sum(target_mask)
        return 2.0 * intersection / (union + 1e-5)
        
    def _extract_boundary(self, mask: np.ndarray) -> np.ndarray:
        """提取边界"""
        from scipy.ndimage import binary_dilation
        dilated = binary_dilation(mask, iterations=1)
        return dilated & ~mask
        
    def _log_example(
        self,
        volume: torch.Tensor,
        mask: torch.Tensor,
        pred: np.ndarray
    ):
        """记录示例图像到wandb"""
        if wandb.run is not None:
            # 选择中间切片
            slice_idx = volume.shape[2] // 2
            
            # 准备图像
            volume_slice = volume[0, :, slice_idx].cpu().numpy()
            mask_slice = mask[0, 0, slice_idx].cpu().numpy()
            pred_slice = pred[slice_idx]
            
            # 创建可视化
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(2, 2, figsize=(10, 10))
            
            # 显示原始图像
            axes[0, 0].imshow(volume_slice[0], cmap='gray')
            axes[0, 0].set_title('Original Image')
            
            # 显示真实掩码
            axes[0, 1].imshow(mask_slice, cmap='red')
            axes[0, 1].set_title('Ground Truth')
            
            # 显示预测掩码
            axes[1, 0].imshow(pred_slice, cmap='red')
            axes[1, 0].set_title('Prediction')
            
            # 显示叠加结果
            overlay = np.zeros((*volume_slice[0].shape, 3))
            overlay[..., 0] = volume_slice[0]
            overlay[..., 1] = pred_slice
            axes[1, 1].imshow(overlay)
            axes[1, 1].set_title('Overlay')
            
            # 记录到wandb
            wandb.log({
                "examples": wandb.Image(plt),
                "slice_idx": slice_idx
            })
            
            plt.close() 