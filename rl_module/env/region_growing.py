import numpy as np
from typing import Tuple, List, Set
from collections import deque
from scipy.ndimage import generate_binary_structure
from scipy.ndimage import binary_dilation

class RegionGrower:
    """3D Region growing algorithm for multi-modal MRI data"""
    
    def __init__(
        self,
        similarity_threshold: float = 0.3,  # 增加阈值
        min_size: int = 8,       # 降低最小区域要求
        max_size: int = 2000,    # 增加最大区域大小
        connectivity: int = 26,    # 26-connectivity for better region growing
        modality_weights: List[float] = None  # 添加模态权重
    ):
        """
        Initialize region grower
        
        Args:
            similarity_threshold: Threshold for intensity similarity
            min_size: Minimum size of the region
            max_size: Maximum size of the region
            connectivity: 26-connectivity for better region growing
            modality_weights: List of weights for each modality
        """
        self.similarity_threshold = similarity_threshold
        self.min_size = min_size
        self.max_size = max_size
        self.connectivity = connectivity
        self.modality_weights = modality_weights or [0.4, 0.3, 0.3]
        
        # Define neighborhood patterns
        self.neighbors_6 = [
            (-1,0,0), (1,0,0), (0,-1,0), 
            (0,1,0), (0,0,-1), (0,0,1)
        ]
        
        self.neighbors_26 = [
            (x,y,z) 
            for x in [-1,0,1] 
            for y in [-1,0,1] 
            for z in [-1,0,1] 
            if not (x == 0 and y == 0 and z == 0)
        ]
    
    def _get_neighbors(
        self, 
        point: Tuple[int, int, int], 
        volume_shape: Tuple[int, int, int]
    ) -> List[Tuple[int, int, int]]:
        """Get valid neighbors for a point"""
        neighbors = []
        patterns = self.neighbors_26 if self.connectivity == 26 else self.neighbors_6
        
        for dz, dy, dx in patterns:
            z, y, x = point[0] + dz, point[1] + dy, point[2] + dx
            
            # Check bounds
            if (0 <= z < volume_shape[0] and 
                0 <= y < volume_shape[1] and 
                0 <= x < volume_shape[2]):
                neighbors.append((z, y, x))
                
        return neighbors
    
    def _calculate_similarity(
        self,
        volume: np.ndarray,
        point: Tuple[int, int, int],
        mean_intensities: np.ndarray,
        std_intensities: np.ndarray = None,
        gradient_magnitude: np.ndarray = None
    ) -> float:
        """Enhanced similarity calculation with disease-specific features"""
        point_intensities = volume[:, point[0], point[1], point[2]]
        
        # 1. 基本相似度计算（考虑模态权重）
        diff = np.abs(point_intensities - mean_intensities)
        weighted_diff = diff * np.array(self.modality_weights)
        basic_similarity = np.mean(weighted_diff)
        
        # 2. 考虑局部变异性
        if std_intensities is not None:
            normalized_diff = diff / (std_intensities + 1e-6)
            variation_factor = np.mean(normalized_diff)
            basic_similarity *= variation_factor
            
        # 3. 考虑梯度信息
        if gradient_magnitude is not None:
            # 修复：正确访问梯度值
            gradient_value = gradient_magnitude[point[0], point[1], point[2]]
            gradient_factor = gradient_value / (np.max(gradient_magnitude) + 1e-6)
            basic_similarity *= (1 + gradient_factor)
            
        # 4. 考虑疾病特征
        intensity_factor = np.mean(point_intensities) / (np.mean(mean_intensities) + 1e-6)
        disease_factor = np.clip(intensity_factor, 0.5, 2.0)
        
        return basic_similarity / disease_factor
    
    def grow_region(
        self,
        volume: np.ndarray,
        seed_point: Tuple[int, int, int],
        entropy_map: np.ndarray = None
    ) -> Tuple[np.ndarray, float]:
        """
        Grow region from seed point with enhanced features and safety checks
        """
        try:
            volume_shape = volume.shape[1:]
            region_mask = np.zeros(volume_shape, dtype=np.bool_)
            visited = np.zeros(volume_shape, dtype=np.bool_)
            
            if not (0 <= seed_point[0] < volume_shape[0] and 
                   0 <= seed_point[1] < volume_shape[1] and 
                   0 <= seed_point[2] < volume_shape[2]):
                return np.zeros(volume_shape, dtype=np.bool_), 0.0
            
            z, y, x = seed_point
            region_mask[z, y, x] = True
            visited[z, y, x] = True
            
            mean_intensities = volume[:, z, y, x]
            region_size = 1
            
            queue = deque([(z, y, x)])
            max_iterations = min(10000, np.prod(volume_shape))
            iterations = 0
            
            while queue and region_size < self.max_size and iterations < max_iterations:
                iterations += 1
                current = queue.popleft()
                cz, cy, cx = current
                
                for dz, dy, dx in self.neighbors_26:
                    nz, ny, nx = cz + dz, cy + dy, cx + dx
                    
                    if not (0 <= nz < volume_shape[0] and 
                           0 <= ny < volume_shape[1] and 
                           0 <= nx < volume_shape[2]):
                        continue
                    
                    if visited[nz, ny, nx]:
                        continue
                        
                    visited[nz, ny, nx] = True
                    
                    point_intensities = volume[:, nz, ny, nx]
                    diff = np.abs(point_intensities - mean_intensities)
                    similarity = np.mean(diff * np.array(self.modality_weights))
                    
                    threshold = self.similarity_threshold
                    if entropy_map is not None:
                        threshold *= (1 + entropy_map[nz, ny, nx])
                    
                    if similarity <= threshold:
                        region_mask[nz, ny, nx] = True
                        queue.append((nz, ny, nx))
                        region_size += 1
                        mean_intensities = (mean_intensities * (region_size - 1) + point_intensities) / region_size
                        
                        if region_size >= self.max_size:
                            break
            
            if region_size < self.min_size:
                return np.zeros(volume_shape, dtype=np.bool_), 0.0
            
            region_score = min(1.0, region_size / self.max_size)
            if entropy_map is not None:
                entropy_score = np.mean(entropy_map[region_mask])
                region_score = 0.7 * region_score + 0.3 * entropy_score
            
            return region_mask, float(region_score)
            
        except Exception as e:
            return np.zeros(volume_shape, dtype=np.bool_), 0.0

    def grow(
        self,
        volume: np.ndarray,
        seed_point: Tuple[int, int, int],
        threshold: float = None
    ) -> np.ndarray:
        """
        Region growing from seed point
        
        Args:
            volume: Input volume
            seed_point: Starting point (z, y, x)
            threshold: Optional intensity similarity threshold
            
        Returns:
            Binary mask of grown region
        """
        if threshold is None:
            threshold = self.similarity_threshold
            
        # 计算局部梯度
        gradients = np.gradient(volume)
        gradient_magnitude = np.sqrt(sum(g**2 for g in gradients))
        
        mask = np.zeros_like(volume, dtype=np.float32)
        seed_value = volume[seed_point]
        
        # 获取邻域结构
        structure = generate_binary_structure(3, self.connectivity)
        
        # Initialize queue with seed point
        queue = deque([seed_point])
        mask[seed_point] = 1
        region_size = 1
        
        # 计算种子点局部统计信息
        local_mean = seed_value
        local_std = 0
        
        while queue and region_size < self.max_size:
            current = queue.popleft()
            z, y, x = current
            
            # 获取所有邻居点
            for dz, dy, dx in self.neighbors_26:
                nz, ny, nx = z + dz, y + dy, x + dx
                
                # Check bounds
                if not (0 <= nz < volume.shape[0] and
                       0 <= ny < volume.shape[1] and
                       0 <= nx < volume.shape[2]):
                    continue
                
                # Check if already in region
                if mask[nz, ny, nx]:
                    continue
                
                # 自适应阈值
                local_threshold = threshold
                
                # 考虑梯度信息
                gradient_factor = gradient_magnitude[nz, ny, nx] / gradient_magnitude.max()
                local_threshold *= (1 + gradient_factor)
                
                # 考虑局部统计信息
                if region_size > 1:
                    local_threshold *= (1 + local_std / (np.abs(local_mean) + 1e-6))
                
                # Check intensity similarity
                neighbor_value = volume[nz, ny, nx]
                if abs(neighbor_value - local_mean) <= local_threshold:
                    mask[nz, ny, nx] = 1
                    queue.append((nz, ny, nx))
                    region_size += 1
                    
                    # 更新局部统计信息
                    old_mean = local_mean
                    local_mean = (local_mean * (region_size-1) + neighbor_value) / region_size
                    if region_size > 1:
                        delta = neighbor_value - old_mean
                        local_std = np.sqrt(
                            ((region_size-2) * local_std**2 + delta * (neighbor_value - local_mean)) / (region_size-1)
                        )
        
        # 如果区域太小，返回空掩码
        if region_size < self.min_size:
            return np.zeros_like(mask)
            
        return mask 

    def _calculate_disease_features(self, volume: np.ndarray) -> np.ndarray:
        """Calculate disease-specific features"""
        # 1. 计算每个模态的统计特征
        means = np.mean(volume, axis=(1,2,3))
        stds = np.std(volume, axis=(1,2,3))
        
        # 2. 计算异常分数
        features = np.zeros(volume.shape[1:])
        for i in range(volume.shape[0]):
            # Z-score normalization
            normalized = (volume[i] - means[i]) / (stds[i] + 1e-6)
            # 高信号区域更可能是病变
            features += np.clip(normalized, 0, None)
            
        # 3. 应用空间平滑
        from scipy.ndimage import gaussian_filter
        features = gaussian_filter(features, sigma=1.0)
        
        # 4. 归一化到 [0,1]
        features = (features - features.min()) / (features.max() - features.min() + 1e-6)
        
        return features 