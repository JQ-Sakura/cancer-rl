import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, Any
from rl_module.agents.memory import Memory
import torch.cuda.amp as amp
from contextlib import nullcontext
import gc

class CNNEncoder(nn.Module):
    """Enhanced CNN encoder with attention and residual connections"""
    
    def __init__(self, in_channels: int = 6):
        """
        Args:
            in_channels: Number of input channels (volume + masks)
        """
        super().__init__()
        
        # 定义通道注意力模块
        class ChannelAttention(nn.Module):
            def __init__(self, channels: int, reduction: int = 16):
                super().__init__()
                self.avg_pool = nn.AdaptiveAvgPool3d(1)
                self.max_pool = nn.AdaptiveMaxPool3d(1)
                self.fc = nn.Sequential(
                    nn.Linear(channels, channels // reduction),
                    nn.ReLU(),
                    nn.Linear(channels // reduction, channels)
                )
                
            def forward(self, x):
                avg_out = self.fc(self.avg_pool(x).view(x.size(0), -1))
                max_out = self.fc(self.max_pool(x).view(x.size(0), -1))
                out = torch.sigmoid(avg_out + max_out).view(x.size(0), x.size(1), 1, 1, 1)
                return x * out
        
        # 定义空间注意力模块
        class SpatialAttention(nn.Module):
            def __init__(self, kernel_size: int = 7):
                super().__init__()
                self.conv = nn.Conv3d(2, 1, kernel_size, padding=kernel_size//2)
                
            def forward(self, x):
                avg_out = torch.mean(x, dim=1, keepdim=True)
                max_out, _ = torch.max(x, dim=1, keepdim=True)
                x_cat = torch.cat([avg_out, max_out], dim=1)
                out = torch.sigmoid(self.conv(x_cat))
                return x * out
        
        # 定义残差块
        class ResBlock(nn.Module):
            def __init__(self, channels: int):
                super().__init__()
                self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
                self.bn1 = nn.BatchNorm3d(channels)
                self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
                self.bn2 = nn.BatchNorm3d(channels)
                self.ca = ChannelAttention(channels)
                self.sa = SpatialAttention()
                
            def forward(self, x):
                residual = x
                out = F.relu(self.bn1(self.conv1(x)))
                out = self.bn2(self.conv2(out))
                out = self.ca(out)
                out = self.sa(out)
                out += residual
                return F.relu(out)
        
        # 主干网络
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(2)
        )
        
        # 编码器阶段
        self.encoder_stages = nn.ModuleList([
            nn.Sequential(
                ResBlock(32),
                nn.Conv3d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm3d(64),
                nn.ReLU(),
                nn.MaxPool3d(2)
            ),
            nn.Sequential(
                ResBlock(64),
                nn.Conv3d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm3d(128),
                nn.ReLU(),
                nn.MaxPool3d(2)
            ),
            nn.Sequential(
                ResBlock(128),
                nn.Conv3d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm3d(256),
                nn.ReLU(),
                nn.MaxPool3d(2)
            ),
            nn.Sequential(
                ResBlock(256),
                nn.Conv3d(256, 512, kernel_size=3, padding=1),
                nn.BatchNorm3d(512),
                nn.ReLU(),
                nn.AdaptiveAvgPool3d((2, 2, 2))
            )
        ])
        
        # 特征融合
        self.fusion = nn.Sequential(
            nn.Linear(512 * 2 * 2 * 2, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 主干特征
        x = self.stem(x)
        
        # 多尺度特征提取
        features = []
        for stage in self.encoder_stages:
            x = stage(x)
            features.append(x)
        
        # 特征融合
        x = features[-1]
        x = x.view(x.size(0), -1)
        x = self.fusion(x)
        
        return x

class ActorCritic(nn.Module):
    """Combined actor-critic network"""
    
    def __init__(self, state_dim: Dict, action_dim: Any = 3):
        super().__init__()
        
        # Get action dimension
        if isinstance(action_dim, (tuple, list, np.ndarray)):
            action_dim = 3  # Default to 3 for (x, y, z) coordinates
        elif hasattr(action_dim, 'shape'):
            action_dim = 3  # Default to 3 for Box space
        
        # Encoder for processing state
        self.encoder = CNNEncoder(in_channels=6)  # 3 modalities + 3 masks
        
        # 增加更深的网络结构和残差连接
        self.actor_hidden = nn.Sequential(
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # 使用更保守的初始化
        self.actor_mean = nn.Sequential(
            nn.Linear(256, action_dim),
            nn.Tanh()  # 限制输出范围
        )
        
        # 使用可学习的log_std，但有更严格的范围限制
        log_std_init = -1.0 * torch.ones(1, action_dim)
        self.actor_log_std = nn.Parameter(log_std_init)
        self.log_std_min = -10
        self.log_std_max = 1
        
        # 增加critic网络的深度
        self.critic_hidden = nn.Sequential(
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.critic = nn.Linear(256, 1)
        
        # 添加辅助任务：预测下一个状态的value
        self.aux_value_pred = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # 初始化权重
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """使用更保守的初始化策略"""
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight.data, gain=0.1)
            if module.bias is not None:
                module.bias.data.zero_()
                
    def forward(self, state: Dict) -> Tuple[torch.distributions.Normal, torch.Tensor]:
        """Forward pass with improved stability"""
        try:
            # 输入检查和预处理
            if not isinstance(state, dict):
                raise ValueError(f"Expected state to be a dict, got {type(state)}")
            
            if 'volume' not in state:
                raise ValueError("State must contain 'volume' key")
                
            if 'masks' not in state:
                if isinstance(state['volume'], torch.Tensor):
                    state['masks'] = torch.zeros_like(state['volume'])
                else:
                    raise ValueError("Volume must be a torch.Tensor")
            
            # 维度检查
            if state['volume'].dim() != 5:
                raise ValueError(f"Volume must be 5D (B,C,H,W,D), got shape {state['volume'].shape}")
            if state['masks'].dim() != 5:
                raise ValueError(f"Masks must be 5D (B,C,H,W,D), got shape {state['masks'].shape}")
            
            # 连接volume和masks
            x = torch.cat([state['volume'], state['masks']], dim=1)
            
            # 特征提取
            features = self.encoder(x)
            
            # Actor路径
            actor_features = self.actor_hidden(features)
            mean = self.actor_mean(actor_features)
            
            # 限制log_std的范围
            log_std = torch.clamp(
                self.actor_log_std,
                self.log_std_min,
                self.log_std_max
            )
            std = torch.exp(log_std) + 1e-6
            
            # 使用更稳定的正态分布
            dist = torch.distributions.Normal(mean, std)
            
            # Critic路径
            critic_features = self.critic_hidden(features)
            value = self.critic(critic_features)
            
            # 辅助任务：预测下一个状态的value
            aux_value = self.aux_value_pred(critic_features)
            
            # 检查输出的有效性
            if torch.isnan(mean).any() or torch.isnan(std).any() or torch.isnan(value).any():
                raise ValueError("NaN values detected in network outputs")
            
            return dist, value, aux_value
            
        except Exception as e:
            print(f"Forward pass failed: {str(e)}")
            torch.cuda.empty_cache()
            gc.collect()
            raise

class PPOAgent:
    """PPO agent for cancer region detection"""
    
    def __init__(
        self,
        state_dim: Dict,
        action_dim: Any,
        device: str = 'cuda',
        lr: float = 1e-4,
        gamma: float = 0.99,
        epsilon: float = 0.2,
        c1: float = 1.0,
        c2: float = 0.01,
        memory_batch_size: int = 128,
        gradient_clip: float = 0.5,
        use_amp: bool = True,
        gae_lambda: float = 0.95  # 添加GAE lambda参数
    ):
        """
        Initialize PPO agent
        
        Args:
            state_dim: State space dimensions
            action_dim: Action space dimensions
            device: Device to run on
            lr: Learning rate
            gamma: Discount factor
            epsilon: PPO clipping parameter
            c1: Value loss coefficient
            c2: Entropy coefficient
            memory_batch_size: Batch size for memory
            gradient_clip: Maximum gradient norm
            use_amp: Whether to use automatic mixed precision
            gae_lambda: GAE lambda parameter
        """
        self.device = device
        self.gamma = gamma
        self.epsilon = epsilon
        self.c1 = c1
        self.c2 = c2
        self.gradient_clip = gradient_clip
        self.use_amp = use_amp
        self.gae_lambda = gae_lambda  # 添加GAE lambda属性
        
        # 添加动作范围限制
        self.action_low = torch.tensor([0, 0, 0]).to(device)
        self.action_high = torch.tensor([127, 127, 127]).to(device)  # 根据体素大小调整
        
        # Initialize actor-critic network
        self.ac = ActorCritic(
            state_dim=state_dim,
            action_dim=action_dim
        ).to(device)
        
        # 使用Adam优化器并添加weight decay
        self.optimizer = torch.optim.Adam(
            self.ac.parameters(),
            lr=lr,
            weight_decay=1e-5  # 添加L2正则化
        )
        
        # Initialize memory with larger batch size
        self.memory = Memory(max_batch_size=memory_batch_size)
        
        # Initialize AMP scaler and context
        if use_amp:
            self.scaler = torch.amp.GradScaler('cuda')
            self.amp_context = torch.amp.autocast('cuda')
        else:
            self.scaler = None
            self.amp_context = nullcontext()
            
        # 设置CUDA性能优化
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
    def select_action(self, state: Any) -> Tuple[np.ndarray, float, float]:
        """选择动作并返回相关信息"""
        with torch.no_grad():
            try:
                # 处理状态输入
                if isinstance(state, tuple):
                    state = state[0]
                
                if not isinstance(state, dict):
                    raise TypeError(f"Expected state to be a dict, got {type(state)}")
                
                # 处理volume
                volume = state['volume']
                if not isinstance(volume, (np.ndarray, torch.Tensor)):
                    raise TypeError(f"Volume must be ndarray or Tensor, got {type(volume)}")
                
                if isinstance(volume, np.ndarray):
                    volume = torch.FloatTensor(volume)
                
                # 确保volume是5D张量 (B, C, H, W, D)
                if volume.dim() == 4:
                    volume = volume.unsqueeze(0)
                elif volume.dim() != 5:
                    raise ValueError(f"Invalid volume dimensions: {volume.shape}")
                
                # 处理掩码并合并为一个tensor
                masks = []
                for mask_name in ['current_mask', 'entropy_map', 'history_mask']:
                    mask = state.get(mask_name, None)
                    if mask is None or not isinstance(mask, (np.ndarray, torch.Tensor)):
                        mask = np.zeros_like(volume[0,0].cpu().numpy())
                    
                    if isinstance(mask, np.ndarray):
                        mask = torch.FloatTensor(mask)
                    
                    # 确保mask是4D (B, H, W, D)
                    if mask.dim() == 3:
                        mask = mask.unsqueeze(0)
                    elif mask.dim() == 2:
                        mask = mask.unsqueeze(0).unsqueeze(0)
                    
                    masks.append(mask)
                
                # 将masks堆叠在一起
                masks_tensor = torch.stack(masks, dim=1)  # (B, 3, H, W, D)
                
                # 创建状态字典
                state_tensors = {
                    'volume': volume.to(self.device),
                    'masks': masks_tensor.to(self.device)
                }
                
                # Forward pass through actor-critic
                dist, value, _ = self.ac(state_tensors)
                
                # 计算熵来指导探索
                entropy = dist.entropy().mean()
                
                # 使用熵和历史信息来调整动作选择
                history_mask = masks_tensor[:, 2]  # history_mask
                entropy_weight = torch.exp(-3 * history_mask)  # 减小历史区域的惩罚
                
                # 采样动作并应用引导
                if np.random.random() < 0.1:  # 增加随机探索概率
                    action = dist.sample() * entropy_weight.mean()
                else:
                    action = dist.rsample() * entropy_weight.mean()
                
                action_valid = False
                
                # 尝试找到有效的动作
                for _ in range(3):  # 最多尝试3次
                    # 确保action是正确的形状
                    if action.dim() == 1:
                        action = action.unsqueeze(0)
                    
                    # 将动作转换到[0,1]范围并缩放到实际范围
                    action_normalized = torch.sigmoid(action)
                    action_scaled = action_normalized * (self.action_high - self.action_low) + self.action_low
                    
                    # 获取整数坐标并确保在有效范围内
                    action_point = torch.round(action_scaled).long()
                    action_point = torch.clamp(action_point, min=0, max=127)
                    
                    if action_point.dim() > 1:
                        action_point = action_point[0]
                    
                    try:
                        # 检查是否在已访问区域
                        if history_mask[0, action_point[0], action_point[1], action_point[2]] > 0.5:
                            # 在已访问区域，重新采样
                            if np.random.random() < 0.1:  # 增加随机探索概率
                                action = dist.sample() * entropy_weight.mean()
                            else:
                                action = dist.rsample() * entropy_weight.mean()
                        else:
                            # 找到有效动作
                            action = action_scaled
                            action_valid = True
                            break
                    except IndexError:
                        # 索引错误，重新采样
                        if np.random.random() < 0.1:  # 增加随机探索概率
                            action = dist.sample() * entropy_weight.mean()
                        else:
                            action = dist.rsample() * entropy_weight.mean()
                
                # 如果没有找到有效动作，使用最后一次采样的动作
                if not action_valid:
                    action = action_scaled
                
                # 确保最终动作在有效范围内
                action = torch.clamp(action, self.action_low, self.action_high)
                
                # 计算log概率
                log_prob = dist.log_prob(action).sum(-1)
                
                # 确保返回正确的形状
                if action.dim() > 1:
                    action = action[0]
                
                return (
                    action.cpu().numpy(),
                    log_prob.cpu().numpy()[0],
                    value.cpu().numpy()[0, 0]
                )
                
            except Exception as e:
                print(f"Warning: Action selection failed with error: {str(e)}")
                # 返回随机动作作为后备方案
                random_action = np.random.uniform(
                    low=self.action_low.cpu().numpy(),
                    high=self.action_high.cpu().numpy()
                )
                return random_action, 0.0, 0.0
        
    def update(
        self,
        batch_size: int = 4,
        epochs: int = 2,
        accumulation_steps: int = 8
    ) -> Dict[str, float]:
        """Update policy with improved stability"""
        try:
            if len(self.memory) == 0:
                return {
                    'policy_loss': 0.0,
                    'value_loss': 0.0,
                    'entropy': 0.0
                }
            
            # 清理显存
            torch.cuda.empty_cache()
            gc.collect()
            
            # 使用较小的子批次
            sub_batch_size = min(4, len(self.memory) // 4)
            
            # 初始化指标
            total_policy_loss = 0
            total_value_loss = 0
            total_aux_value_loss = 0
            total_entropy = 0
            num_updates = 0
            
            # 计算优势和回报
            with torch.no_grad():
                try:
                    all_states, all_actions, all_rewards, all_values, all_log_probs, all_dones = self.memory.get_batch(self.device)
                    
                    # 计算GAE
                    advantages = torch.zeros_like(all_rewards)
                    returns = torch.zeros_like(all_rewards)
                    
                    gae = 0
                    for t in reversed(range(len(all_rewards))):
                        if t == len(all_rewards) - 1:
                            next_value = 0
                        else:
                            next_value = all_values[t + 1]
                            
                        delta = all_rewards[t] + self.gamma * next_value * (1 - all_dones[t]) - all_values[t]
                        gae = delta + self.gamma * self.gae_lambda * (1 - all_dones[t]) * gae
                        advantages[t] = gae
                        returns[t] = advantages[t] + all_values[t]
                    
                    # 标准化优势
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                    
                    # 将数据移到CPU
                    advantages = advantages.cpu()
                    returns = returns.cpu()
                    all_log_probs = all_log_probs.cpu()
                    
                    del all_rewards, all_values, all_dones
                    torch.cuda.empty_cache()
                    
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        torch.cuda.empty_cache()
                        gc.collect()
                        sub_batch_size = max(1, sub_batch_size // 2)
                        print(f"Warning: OOM in advantage computation, reducing sub_batch_size to {sub_batch_size}")
                        return {
                            'policy_loss': 0.0,
                            'value_loss': 0.0,
                            'entropy': 0.0
                        }
                    else:
                        raise e
            
            # 训练循环
            for epoch in range(epochs):
                indices = torch.randperm(len(returns))
                
                for start_idx in range(0, len(returns), sub_batch_size):
                    try:
                        # 检查显存使用
                        if torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() > 0.7:
                            torch.cuda.empty_cache()
                            gc.collect()
                        
                        end_idx = min(start_idx + sub_batch_size, len(returns))
                        batch_indices = indices[start_idx:end_idx]
                        
                        # 获取批次数据
                        batch_data = self.memory.get_batch_indices(batch_indices.numpy(), self.device)
                        batch_states, batch_actions = batch_data[0], batch_data[1]
                        batch_advantages = advantages[batch_indices].to(self.device)
                        batch_returns = returns[batch_indices].to(self.device)
                        batch_old_log_probs = all_log_probs[batch_indices].to(self.device)
                        
                        # Forward pass
                        dist, value, aux_value = self.ac(batch_states)
                        
                        # 检查NaN
                        if torch.isnan(dist.loc).any() or torch.isnan(dist.scale).any():
                            print("Warning: NaN values detected in distribution parameters")
                            continue
                        
                        entropy = dist.entropy().mean()
                        
                        # 计算log probabilities和ratio
                        log_probs = dist.log_prob(batch_actions).sum(-1)
                        ratio = torch.exp(torch.clamp(log_probs - batch_old_log_probs, -20, 20))
                        
                        # 计算losses
                        surr1 = ratio * batch_advantages
                        surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * batch_advantages
                        
                        policy_loss = -torch.min(surr1, surr2).mean()
                        value_loss = F.mse_loss(value.squeeze(), batch_returns)
                        aux_value_loss = F.mse_loss(aux_value.squeeze(), batch_returns)
                        
                        # 总loss
                        loss = (
                            policy_loss + 
                            self.c1 * value_loss + 
                            0.5 * aux_value_loss -  # 辅助任务loss权重较小
                            self.c2 * entropy
                        ) / accumulation_steps
                        
                        # 检查loss是否为NaN
                        if torch.isnan(loss).any():
                            print("Warning: NaN values detected in loss")
                            continue
                        
                        # Backward pass
                        loss.backward()
                        
                        # 梯度裁剪和更新
                        if (num_updates + 1) % accumulation_steps == 0:
                            if self._check_gradients():  # 添加梯度检查
                                torch.nn.utils.clip_grad_norm_(self.ac.parameters(), self.gradient_clip)
                                self.optimizer.step()
                                self.optimizer.zero_grad()
                        
                        # 累积metrics
                        total_policy_loss += policy_loss.item() * accumulation_steps
                        total_value_loss += value_loss.item() * accumulation_steps
                        total_aux_value_loss += aux_value_loss.item() * accumulation_steps
                        total_entropy += entropy.item() * accumulation_steps
                        num_updates += 1
                        
                        # 清理内存
                        del batch_states, batch_actions, batch_advantages, batch_returns
                        del dist, value, aux_value, entropy, log_probs, ratio
                        del policy_loss, value_loss, aux_value_loss, loss
                        torch.cuda.empty_cache()
                        
                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            torch.cuda.empty_cache()
                            gc.collect()
                            sub_batch_size = max(1, sub_batch_size // 2)
                            print(f"Warning: OOM occurred, reducing sub_batch_size to {sub_batch_size}")
                            continue
                        else:
                            raise e
            
            # 清理内存
            self.memory.clear()
            
            # 计算平均metrics
            num_updates = max(1, num_updates)
            return {
                'policy_loss': total_policy_loss / num_updates,
                'value_loss': total_value_loss / num_updates,
                'aux_value_loss': total_aux_value_loss / num_updates,
                'entropy': total_entropy / num_updates
            }
            
        except Exception as e:
            print(f"Warning: Update failed with error: {str(e)}")
            torch.cuda.empty_cache()
            gc.collect()
            return {
                'policy_loss': 0.0,
                'value_loss': 0.0,
                'entropy': 0.0
            }
        
    def _check_gradients(self) -> bool:
        """检查梯度的有效性"""
        for param in self.ac.parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    print("Warning: NaN gradients detected")
                    return False
                if torch.abs(param.grad).max() > 1000:
                    print("Warning: Gradient explosion detected")
                    return False
        return True
        
    def save(self, path: str):
        """Save model state"""
        state = {
            'actor_critic': self.ac.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        torch.save(state, path)
        
    def load(self, path: str):
        """Load model state"""
        state = torch.load(path, map_location=self.device)
        self.ac.load_state_dict(state['actor_critic'])
        self.optimizer.load_state_dict(state['optimizer'])
        
    def save_checkpoint(self, path: str, epoch: int, metrics: Dict = None):
        """Save a training checkpoint with additional information"""
        state = {
            'epoch': epoch,
            'actor_critic': self.ac.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'metrics': metrics or {}
        }
        torch.save(state, path)
        
    def load_checkpoint(self, path: str) -> Dict:
        """Load a training checkpoint and return the saved metrics"""
        state = torch.load(path, map_location=self.device)
        self.ac.load_state_dict(state['actor_critic'])
        self.optimizer.load_state_dict(state['optimizer'])
        return {
            'epoch': state.get('epoch', 0),
            'metrics': state.get('metrics', {})
        } 