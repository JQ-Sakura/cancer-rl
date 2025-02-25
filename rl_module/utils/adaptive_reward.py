import numpy as np
from typing import Dict, List, Optional
from collections import deque

class AdaptiveRewardShaper:
    """Adaptive reward shaping with curriculum learning"""
    
    def __init__(
        self,
        window_size: int = 100,
        initial_scale: float = 1.0,
        min_scale: float = 0.1,
        max_scale: float = 10.0,
        adaptation_rate: float = 0.01
    ):
        """
        Initialize adaptive reward shaper
        
        Args:
            window_size: Size of moving average window
            initial_scale: Initial reward scaling factor
            min_scale: Minimum reward scaling factor
            max_scale: Maximum reward scaling factor
            adaptation_rate: Rate of reward scale adaptation
        """
        self.window_size = window_size
        self.scale = initial_scale
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.adaptation_rate = adaptation_rate
        
        # 历史记录
        self.reward_history = []
        self.iou_history = []
        self.entropy_history = []
        self.episode_lengths = []
        
        # 移动平均
        self.avg_reward = 0
        self.avg_iou = 0
        self.avg_entropy = 0
        self.avg_episode_length = 0
        
        # 新增：记录连续性指标
        self.last_iou = 0
        self.improvement_streak = 0
        
        # 性能指标
        self.best_iou = 0.0
        self.best_reward = float('-inf')
        self.stagnation_counter = 0
        
    def update_history(
        self,
        reward: float,
        iou: float,
        episode_length: int
    ):
        """更新历史记录"""
        self.reward_history.append(reward)
        self.iou_history.append(iou)
        self.episode_lengths.append(episode_length)
        
        # 维护窗口大小
        if len(self.reward_history) > self.window_size:
            self.reward_history.pop(0)
            self.iou_history.pop(0)
            self.episode_lengths.pop(0)
        
        # 更新平均值
        self.avg_reward = sum(self.reward_history) / len(self.reward_history)
        self.avg_iou = sum(self.iou_history) / len(self.iou_history)
        self.avg_episode_length = sum(self.episode_lengths) / len(self.episode_lengths)
        
        # 更新最佳性能
        if iou > self.best_iou:
            self.best_iou = iou
            self.stagnation_counter = 0
        else:
            self.stagnation_counter += 1
            
    def adapt_scale(self) -> float:
        """自适应调整奖励比例"""
        if len(self.reward_history) < self.window_size:
            return self.scale
            
        # 计算性能指标
        avg_reward = np.mean(self.reward_history)
        avg_iou = np.mean(self.iou_history)
        avg_length = np.mean(self.episode_lengths)
        
        # 根据性能调整比例
        if avg_iou > 0.8 * self.best_iou:  # 性能良好
            self.scale *= (1 - self.adaptation_rate)  # 降低奖励以增加挑战
        elif self.stagnation_counter > 10:  # 性能停滞
            self.scale *= (1 + self.adaptation_rate)  # 增加奖励以促进探索
            
        # 限制比例范围
        self.scale = np.clip(self.scale, self.min_scale, self.max_scale)
        
        return self.scale
        
    def shape_reward(
        self,
        base_reward: float,
        iou: float,
        entropy: float,
        exploration_factor: float,
        steps_taken: int,
        is_boundary: bool
    ) -> float:
        """改进的奖励整形函数"""
        
        # 计算IoU改进
        iou_improvement = max(0, iou - self.last_iou)
        self.last_iou = iou
        
        # 更新连续改进计数
        if iou_improvement > 0:
            self.improvement_streak += 1
        else:
            self.improvement_streak = 0
        
        # 基础奖励组成部分
        iou_reward = iou * 2.0  # IoU的基础奖励
        improvement_reward = iou_improvement * 3.0  # IoU改进的奖励
        streak_bonus = min(self.improvement_streak * 0.1, 1.0)  # 连续改进奖励
        
        # 探索奖励
        exploration_reward = entropy * (1.0 - exploration_factor) * 0.5
        
        # 边界奖励
        boundary_reward = 0.5 if is_boundary else 0.0
        
        # 步数惩罚
        step_penalty = -0.01 * steps_taken
        
        # 组合所有奖励
        shaped_reward = (
            base_reward +
            iou_reward +
            improvement_reward +
            streak_bonus +
            exploration_reward +
            boundary_reward +
            step_penalty
        ) * self.scale
        
        # 更新历史记录
        self.reward_history.append(shaped_reward)
        self.iou_history.append(iou)
        self.entropy_history.append(entropy)
        
        # 维护窗口大小
        if len(self.reward_history) > self.window_size:
            self.reward_history.pop(0)
            self.iou_history.pop(0)
            self.entropy_history.pop(0)
        
        # 更新移动平均
        self.avg_reward = sum(self.reward_history) / len(self.reward_history)
        self.avg_iou = sum(self.iou_history) / len(self.iou_history)
        self.avg_entropy = sum(self.entropy_history) / len(self.entropy_history)
        
        # 自适应调整scale
        target_reward = 1.0
        if self.avg_reward != 0:
            scale_adjustment = (target_reward / abs(self.avg_reward)) ** self.adaptation_rate
            self.scale = min(max(self.scale * scale_adjustment, self.min_scale), self.max_scale)
        
        return shaped_reward
        
    def get_curriculum_stage(self) -> Dict:
        """获取课程学习阶段的参数"""
        if len(self.iou_history) < self.window_size:
            return {
                'max_steps': 50,  # 较短的episode
                'num_points': 1,  # 较少的prompt点
                'distance_threshold': 0.3  # 较大的距离阈值
            }
            
        avg_iou = np.mean(self.iou_history)
        if avg_iou < 0.3:  # 初级阶段
            return {
                'max_steps': 50,
                'num_points': 1,
                'distance_threshold': 0.3
            }
        elif avg_iou < 0.6:  # 中级阶段
            return {
                'max_steps': 100,
                'num_points': 2,
                'distance_threshold': 0.2
            }
        else:  # 高级阶段
            return {
                'max_steps': 200,
                'num_points': 3,
                'distance_threshold': 0.1
            }
            
    def get_stats(self) -> dict:
        """获取当前统计信息"""
        return {
            'current_scale': self.scale,
            'avg_reward': self.avg_reward,
            'avg_iou': self.avg_iou,
            'avg_entropy': self.avg_entropy,
            'improvement_streak': self.improvement_streak,
            'best_iou': self.best_iou,
            'stagnation_counter': self.stagnation_counter,
            'avg_episode_length': self.avg_episode_length
        } 