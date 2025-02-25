import numpy as np
import torch
from typing import Dict, List, Tuple, Any
import gc

class Memory:
    """Memory buffer for storing trajectories with optimized memory management"""
    
    def __init__(self, max_batch_size: int = 16):
        self.states: List[Dict] = []
        self.actions: List[np.ndarray] = []
        self.rewards: List[float] = []
        self.values: List[float] = []
        self.log_probs: List[float] = []
        self.dones: List[bool] = []
        self.max_batch_size = max_batch_size
        self._cached_tensors = {}
        
    def push(
        self,
        state: Any,
        action: np.ndarray,
        reward: float,
        value: float,
        log_prob: float,
        done: bool
    ):
        """Add a transition to memory with memory optimization"""
        try:
            # 处理状态
            if isinstance(state, tuple):
                state = state[0]
                
            # 确保所有状态数据都是numpy数组，并使用float32减少内存使用
            state_np = {}
            
            # 处理volume，确保使用float32和3个通道
            if 'volume' in state:
                if torch.is_tensor(state['volume']):
                    volume = state['volume'].detach().cpu().numpy().astype(np.float32)
                else:
                    volume = np.asarray(state['volume'], dtype=np.float32)
                
                # 确保volume是5D的 (B, C, H, W, D)
                if volume.ndim == 3:  # (H, W, D)
                    volume = volume[None, None, ...]  # 添加batch和channel维度
                elif volume.ndim == 4:  # (B, H, W, D) or (C, H, W, D)
                    volume = volume[None, ...]  # 添加batch维度
                elif volume.ndim == 5:  # (B, C, H, W, D)
                    pass
                elif volume.ndim == 6 and volume.shape[1] == 1:  # (B, 1, C, H, W, D)
                    volume = np.squeeze(volume, axis=1)
                else:
                    raise ValueError(f"Invalid volume dimensions: {volume.shape}")
                
                # 确保通道数正确
                if volume.shape[1] == 1:
                    volume = np.repeat(volume, 3, axis=1)  # 复制单通道到3个通道
                elif volume.shape[1] != 3:
                    # 如果第二个维度不是通道维度，尝试转置
                    if volume.shape[0] == 1 or volume.shape[0] == 3:
                        volume = np.transpose(volume, (0, 4, 1, 2, 3))
                    else:
                        raise ValueError(f"Volume must have 1 or 3 channels, got {volume.shape[1]}")
                
                state_np['volume'] = volume
                    
            # 处理masks，确保使用float32和正确的维度
            masks = []
            for mask_name in ['current_mask', 'entropy_map', 'history_mask']:
                if mask_name in state:
                    if torch.is_tensor(state[mask_name]):
                        mask = state[mask_name].detach().cpu().numpy().astype(np.float32)
                    else:
                        mask = np.asarray(state[mask_name], dtype=np.float32)
                    
                    # 确保mask是5D的 (B, C, H, W, D)
                    if mask.ndim == 3:  # (H, W, D)
                        mask = mask[None, None, ...]  # 添加batch和channel维度
                    elif mask.ndim == 4:  # (B, H, W, D)
                        mask = mask[:, None, ...]  # 添加channel维度
                    elif mask.ndim == 5:  # (B, C, H, W, D)
                        pass
                    elif mask.ndim == 6 and mask.shape[1] == 1:  # (B, 1, C, H, W, D)
                        mask = np.squeeze(mask, axis=1)
                    else:
                        raise ValueError(f"Invalid mask dimensions for {mask_name}: {mask.shape}")
                    
                    masks.append(mask)
                else:
                    # 如果缺少某个mask，创建全零mask
                    zero_shape = (1, 1) + volume.shape[2:]  # (B, C, H, W, D)
                    masks.append(np.zeros(zero_shape, dtype=np.float32))
            
            # 将masks堆叠在一起
            state_np['masks'] = np.concatenate(masks, axis=1)  # 在通道维度上连接
            
            self.states.append(state_np)
            self.actions.append(np.asarray(action, dtype=np.float32))
            self.rewards.append(float(reward))
            self.values.append(float(value))
            self.log_probs.append(float(log_prob))
            self.dones.append(bool(done))
            
            # 如果缓存中有数据，清除它们
            self._cached_tensors.clear()
            
            # 主动进行内存清理
            if len(self.states) % 100 == 0:  # 每100个样本清理一次
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Warning: Failed to push to memory: {str(e)}")
            # 确保在出错时不会留下不完整的数据
            self._remove_last_incomplete()
            raise
            
    def _remove_last_incomplete(self):
        """移除最后一个不完整的transition"""
        if len(self.states) > len(self.actions):
            self.states.pop()
        if len(self.actions) > len(self.rewards):
            self.actions.pop()
        if len(self.rewards) > len(self.values):
            self.rewards.pop()
        if len(self.values) > len(self.log_probs):
            self.values.pop()
        if len(self.log_probs) > len(self.dones):
            self.log_probs.pop()
            
    def get_batch_indices(self, indices: np.ndarray, device: str) -> Tuple[Dict, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get a specific batch of transitions by indices"""
        try:
            if len(indices) == 0:
                raise ValueError("Empty indices array")
                
            # Convert states to tensors
            states = {}
            
            # Process volume for specific indices
            volumes = [self.states[i]['volume'] for i in indices]
            # 确保volume的维度是 (B, C, H, W, D)
            volumes = np.stack(volumes)  # 现在是 (B, 1, C, H, W, D)
            if volumes.shape[1] == 1:
                volumes = np.squeeze(volumes, axis=1)  # 移除多余的维度
            states['volume'] = torch.from_numpy(volumes).to(device, dtype=torch.float32, non_blocking=True)
            
            # Process masks for specific indices
            masks = [self.states[i]['masks'] for i in indices]
            # 确保masks的维度是 (B, C, H, W, D)
            masks = np.stack(masks)  # 现在是 (B, 1, C, H, W, D)
            if masks.shape[1] == 1:
                masks = np.squeeze(masks, axis=1)  # 移除多余的维度
            states['masks'] = torch.from_numpy(masks).to(device, dtype=torch.float32, non_blocking=True)
            
            # Convert other data to tensors for specific indices
            actions = torch.from_numpy(np.stack([self.actions[i] for i in indices])).to(device, dtype=torch.float32, non_blocking=True)
            rewards = torch.tensor([self.rewards[i] for i in indices], device=device, dtype=torch.float32)
            values = torch.tensor([self.values[i] for i in indices], device=device, dtype=torch.float32)
            log_probs = torch.tensor([self.log_probs[i] for i in indices], device=device, dtype=torch.float32)
            dones = torch.tensor([self.dones[i] for i in indices], device=device, dtype=torch.float32)
            
            return states, actions, rewards, values, log_probs, dones
            
        except Exception as e:
            print(f"Warning: Failed to get batch indices: {str(e)}")
            # 清理可能的内存泄漏
            torch.cuda.empty_cache()
            gc.collect()
            raise
            
    def get_batch(self, device: str) -> Tuple[Dict, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get all stored transitions as a batch with memory optimization"""
        if len(self) == 0:
            raise ValueError("Memory buffer is empty")
            
        try:
            # 如果数据量太大，分批处理
            if len(self) > self.max_batch_size:
                # 将数据分成较小的批次
                batch_size = min(self.max_batch_size, len(self) // 2)
                indices = np.random.choice(len(self), batch_size, replace=False)
                return self.get_batch_indices(indices, device)
            
            # 对于小数据量，使用缓存
            cache_key = f"full_batch_{device}"
            if cache_key in self._cached_tensors:
                return self._cached_tensors[cache_key]
            
            # 如果没有缓存，创建新的tensors
            states = {}
            
            # Process volume
            volumes = [s['volume'] for s in self.states]
            # 确保volume的维度是 (B, C, H, W, D)
            volumes = np.stack(volumes)  # 现在是 (B, 1, C, H, W, D)
            if volumes.shape[1] == 1:
                volumes = np.squeeze(volumes, axis=1)  # 移除多余的维度
            states['volume'] = torch.from_numpy(volumes).to(device, dtype=torch.float32, non_blocking=True)
            
            # Process masks
            masks = [s['masks'] for s in self.states]
            # 确保masks的维度是 (B, C, H, W, D)
            masks = np.stack(masks)  # 现在是 (B, 1, C, H, W, D)
            if masks.shape[1] == 1:
                masks = np.squeeze(masks, axis=1)  # 移除多余的维度
            states['masks'] = torch.from_numpy(masks).to(device, dtype=torch.float32, non_blocking=True)
            
            # Convert other data to tensors
            actions = torch.from_numpy(np.stack(self.actions)).to(device, dtype=torch.float32, non_blocking=True)
            rewards = torch.tensor(self.rewards, device=device, dtype=torch.float32)
            values = torch.tensor(self.values, device=device, dtype=torch.float32)
            log_probs = torch.tensor(self.log_probs, device=device, dtype=torch.float32)
            dones = torch.tensor(self.dones, device=device, dtype=torch.float32)
            
            # 缓存结果
            self._cached_tensors[cache_key] = (states, actions, rewards, values, log_probs, dones)
            
            return self._cached_tensors[cache_key]
            
        except Exception as e:
            print(f"Warning: Failed to get batch: {str(e)}")
            # 清理可能的内存泄漏
            torch.cuda.empty_cache()
            gc.collect()
            raise
            
    def clear(self):
        """Clear memory and cached tensors"""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()
        self._cached_tensors.clear()
        
        # 强制进行垃圾回收
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
    def __len__(self) -> int:
        """Get current size of memory"""
        return len(self.states) 