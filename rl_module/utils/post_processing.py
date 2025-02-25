import numpy as np
import torch
from typing import Tuple, Optional, Dict, Union
from scipy import ndimage
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax
import cv2
from skimage import morphology
from skimage.measure import label, regionprops
import logging

logger = logging.getLogger(__name__)

class PostProcessor:
    """后处理类，包含多种后处理方法"""
    
    def __init__(
        self,
        min_size: int = 64,
        apply_crf: bool = True,
        crf_iterations: int = 5,
        crf_config: Optional[Dict] = None,
        use_adaptive_params: bool = True
    ):
        """
        初始化后处理器
        
        Args:
            min_size: 最小区域大小
            apply_crf: 是否应用CRF
            crf_iterations: CRF迭代次数
            crf_config: CRF配置参数
            use_adaptive_params: 是否使用自适应参数调整
        """
        self.min_size = min_size
        self.apply_crf = apply_crf
        self.crf_iterations = crf_iterations
        self.use_adaptive_params = use_adaptive_params
        
        # 增强的CRF默认配置
        self.crf_config = {
            'bilateral_sxy': (3, 3),      # 减小空间sigma以保留更多细节
            'bilateral_srgb': (5, 5, 5),  # 减小颜色sigma以增强边界敏感度
            'bilateral_compat': 15,        # 增加兼容性权重
            'gaussian_sxy': (2, 2),       # 减小高斯核以保留细节
            'gaussian_compat': 5          # 增加高斯兼容性权重
        }
        
        # 形态学操作参数
        self.morph_params = {
            'closing_iterations': 2,
            'opening_iterations': 1,
            'smoothing_iterations': 1,
            'hole_size': 64,
            'boundary_size': 1
        }
        
        # 更新配置
        if crf_config is not None:
            self.crf_config.update(crf_config)
            
        # 性能统计
        self.performance_history = []
        
    def _adapt_parameters(self, image: np.ndarray, current_mask: np.ndarray) -> None:
        """自适应调整参数"""
        try:
            if not self.use_adaptive_params:
                return
                
            # 计算图像统计信息
            if image.ndim == 4:  # (C, D, H, W)
                image_stats = np.stack([np.percentile(image[i], (2, 98)) for i in range(image.shape[0])])
                contrast = np.mean(image_stats[:, 1] - image_stats[:, 0])
            else:
                intensity_range = np.percentile(image, (2, 98))
                contrast = intensity_range[1] - intensity_range[0]
            
            # 计算区域特征
            if current_mask is not None:
                regions = self.analyze_regions(current_mask)
                avg_region_size = regions['avg_area']
                num_regions = regions['num_regions']
            else:
                avg_region_size = 0
                num_regions = 0
                
            # 根据图像对比度调整CRF参数
            if contrast > 0.5:  # 高对比度
                self.crf_config['bilateral_srgb'] = (3, 3, 3)  # 减小颜色敏感度
                self.crf_config['bilateral_compat'] = 20       # 增加边界保持
                self.crf_config['gaussian_compat'] = 7         # 增加平滑度
            else:  # 低对比度
                self.crf_config['bilateral_srgb'] = (7, 7, 7)  # 增加颜色容忍度
                self.crf_config['bilateral_compat'] = 10       # 减少边界保持
                self.crf_config['gaussian_compat'] = 3         # 减少平滑度
                
            # 根据区域大小调整形态学参数
            if avg_region_size > 1000:  # 大区域
                self.morph_params['closing_iterations'] = 3
                self.morph_params['hole_size'] = 128
                self.morph_params['boundary_size'] = 2
            elif avg_region_size > 500:  # 中等区域
                self.morph_params['closing_iterations'] = 2
                self.morph_params['hole_size'] = 64
                self.morph_params['boundary_size'] = 1
            else:  # 小区域
                self.morph_params['closing_iterations'] = 1
                self.morph_params['hole_size'] = 32
                self.morph_params['boundary_size'] = 1
                
            # 根据区域数量调整最小区域大小
            if num_regions > 5:
                self.min_size = max(32, self.min_size // 2)  # 减小最小区域要求
            else:
                self.min_size = min(128, self.min_size * 2)  # 增加最小区域要求
                
            # 记录性能统计
            self.performance_history.append({
                'contrast': float(contrast),
                'avg_region_size': float(avg_region_size),
                'num_regions': int(num_regions),
                'min_size': int(self.min_size)
            })
            
        except Exception as e:
            logger.warning(f"参数自适应调整失败: {str(e)}")
        
    def __call__(
        self,
        pred: Union[np.ndarray, torch.Tensor],
        image: Union[np.ndarray, torch.Tensor],
        threshold: float = 0.5
    ) -> np.ndarray:
        """改进的3D后处理流程"""
        try:
            # 转换为numpy数组
            if isinstance(pred, torch.Tensor):
                pred = pred.detach().cpu().numpy()
            if isinstance(image, torch.Tensor):
                image = image.detach().cpu().numpy()
                
            # 确保维度正确
            if pred.ndim == 4:
                pred = pred[0]  # 移除batch维度
                
            # 确保数据类型正确
            pred = pred.astype(np.float32)
            image = image.astype(np.float32)
                
            # 自适应调整参数
            self._adapt_parameters(image, pred)
            
            # 1. 应用阈值并确保bool类型
            binary_mask = (pred > threshold).astype(bool)
            
            # 2. 移除小区域
            binary_mask = self._remove_small_regions(binary_mask)
            binary_mask = binary_mask.astype(bool)  # 确保bool类型
            
            # 3. 应用3D形态学处理
            binary_mask = self._morphological_processing(binary_mask)
            binary_mask = binary_mask.astype(bool)  # 确保bool类型
            
            # 4. 最终的3D清理
            binary_mask = self._final_cleaning(binary_mask)
            
            return binary_mask.astype(np.float32)
            
        except Exception as e:
            logger.error(f"后处理失败: {str(e)}")
            return pred.astype(np.float32)
        
    def _remove_small_regions(self, mask: np.ndarray, min_size: Optional[int] = None) -> np.ndarray:
        """3D移除小区域"""
        try:
            # 使用指定的min_size或默认值
            size_threshold = min_size if min_size is not None else self.min_size
            
            # 确保mask是bool类型
            mask = mask.astype(bool)
            
            # 标记3D连通区域
            labeled_mask, num_features = ndimage.label(mask, structure=ndimage.generate_binary_structure(3, 2))
            
            # 计算每个区域的大小
            component_sizes = np.bincount(labeled_mask.ravel())
            
            # 创建掩码，标记要保留的区域
            too_small = component_sizes < size_threshold
            too_small[0] = False  # 保留背景
            
            # 移除小区域
            mask = ~too_small[labeled_mask]
            
            return mask.astype(np.float32)
            
        except Exception as e:
            logger.warning(f"3D移除小区域失败: {str(e)}")
            return mask.astype(np.float32)
        
    def _morphological_processing(self, mask: np.ndarray) -> np.ndarray:
        """直接在3D图像上进行形态学处理"""
        try:
            # 确保mask是bool类型
            mask = mask.astype(bool)
            
            # 生成3D结构元素
            struct = ndimage.generate_binary_structure(3, 2)
            
            # 1. 3D闭运算填充小孔
            mask = ndimage.binary_closing(
                mask,
                structure=struct,
                iterations=self.morph_params['closing_iterations']
            )
            mask = mask.astype(bool)
            
            # 2. 3D填充孔洞
            filled_mask = ndimage.binary_fill_holes(mask)
            filled_mask = filled_mask.astype(bool)
            
            # 3. 移除小于指定大小的3D孔洞
            # 使用逻辑运算代替 ~ 操作
            holes = np.logical_and(
                np.logical_not(mask),
                filled_mask
            )
            holes = holes.astype(bool)
            small_holes = self._remove_small_regions(
                holes,
                min_size=self.morph_params['hole_size']
            )
            small_holes = small_holes.astype(bool)
            mask = np.logical_or(mask, small_holes)
            mask = mask.astype(bool)
            
            # 4. 3D开运算去除噪点
            mask = ndimage.binary_opening(
                mask,
                structure=struct,
                iterations=self.morph_params['opening_iterations']
            )
            mask = mask.astype(bool)
            
            # 5. 3D边界平滑
            mask = ndimage.binary_dilation(
                mask,
                structure=struct,
                iterations=self.morph_params['boundary_size']
            )
            mask = ndimage.binary_erosion(
                mask,
                structure=struct,
                iterations=self.morph_params['boundary_size']
            )
            mask = mask.astype(bool)
            
            return mask.astype(np.float32)
            
        except Exception as e:
            logger.warning(f"3D形态学处理失败: {str(e)}")
            return mask.astype(np.float32)
        
    def _final_cleaning(self, mask: np.ndarray) -> np.ndarray:
        """3D最终清理处理"""
        try:
            # 确保mask是bool类型
            mask = mask.astype(bool)
            
            # 1. 再次移除小区域
            mask = self._remove_small_regions(mask)
            mask = mask.astype(bool)
            
            # 2. 3D填充孔洞
            mask = ndimage.binary_fill_holes(mask)
            mask = mask.astype(bool)
            
            # 3. 3D平滑边界
            struct = ndimage.generate_binary_structure(3, 2)
            mask = ndimage.binary_closing(mask, structure=struct)
            
            return mask.astype(np.float32)
            
        except Exception as e:
            logger.warning(f"3D最终清理失败: {str(e)}")
            return mask.astype(np.float32)
        
    @staticmethod
    def calculate_boundary_metrics(
        pred: np.ndarray,
        target: np.ndarray,
        tolerance: int = 2
    ) -> Dict[str, float]:
        """
        计算边界相关的指标
        
        Args:
            pred: 预测掩码
            target: 目标掩码
            tolerance: 边界容差（像素）
            
        Returns:
            包含边界指标的字典
        """
        try:
            # 确保输入是bool类型
            pred = pred.astype(bool)
            target = target.astype(bool)
            
            # 提取边界
            pred_dilated = morphology.binary_dilation(pred, morphology.ball(1))
            pred_dilated = pred_dilated.astype(bool)
            pred_boundary = np.logical_and(
                pred_dilated,
                np.logical_not(pred)
            )
            
            target_dilated = morphology.binary_dilation(target, morphology.ball(1))
            target_dilated = target_dilated.astype(bool)
            target_boundary = np.logical_and(
                target_dilated,
                np.logical_not(target)
            )
            
            # 计算距离变换
            pred_dist = ndimage.distance_transform_edt(np.logical_not(pred_boundary))
            target_dist = ndimage.distance_transform_edt(np.logical_not(target_boundary))
            
            # 计算边界准确率
            pred_correct = np.logical_and(pred_boundary, target_dist <= tolerance).sum()
            pred_total = pred_boundary.sum()
            
            # 计算边界召回率
            target_correct = np.logical_and(target_boundary, pred_dist <= tolerance).sum()
            target_total = target_boundary.sum()
            
            # 计算指标
            precision = pred_correct / (pred_total + 1e-8)
            recall = target_correct / (target_total + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            
            return {
                'boundary_precision': float(precision),
                'boundary_recall': float(recall),
                'boundary_f1': float(f1)
            }
            
        except Exception as e:
            logger.warning(f"边界指标计算失败: {str(e)}")
            return {
                'boundary_precision': 0.0,
                'boundary_recall': 0.0,
                'boundary_f1': 0.0
            }
        
    @staticmethod
    def analyze_regions(
        mask: np.ndarray,
        min_size: int = 100
    ) -> Dict[str, Union[int, float, list]]:
        """
        分析分割区域的特征
        
        Args:
            mask: 分割掩码
            min_size: 最小区域大小
            
        Returns:
            区域分析结果
        """
        try:
            # 标记连通区域
            labeled_mask = label(mask)
            props = regionprops(labeled_mask)
            
            # 收集区域统计信息
            regions_info = []
            total_area = 0
            valid_regions = 0
            
            for prop in props:
                if prop.area >= min_size:
                    # 只使用3D图像支持的属性
                    region_info = {
                        'area': int(prop.area),
                        'centroid': tuple(map(float, prop.centroid)),
                        'bbox': tuple(map(int, prop.bbox)),
                        'extent': float(prop.extent),  # 区域填充率
                        'equivalent_diameter': float(prop.equivalent_diameter)  # 等效直径
                    }
                    regions_info.append(region_info)
                    total_area += prop.area
                    valid_regions += 1
                    
            return {
                'num_regions': valid_regions,
                'total_area': int(total_area),
                'avg_area': float(total_area / (valid_regions + 1e-8)),
                'regions': regions_info
            }
            
        except Exception as e:
            logger.warning(f"区域分析失败: {str(e)}")
            # 返回默认值
            return {
                'num_regions': 0,
                'total_area': 0,
                'avg_area': 0,
                'regions': []
            } 