import numpy as np
from typing import Dict, Optional

class RewardShaper:
    """Reward shaping utility for RL training"""
    
    def __init__(
        self,
        config: Dict,
        scale_factor: float = 0.1,
        running_mean: Optional[float] = None,
        running_std: Optional[float] = None,
        epsilon: float = 1e-8
    ):
        """
        Initialize reward shaper
        
        Args:
            config: Reward shaping configuration
            scale_factor: Reward scaling factor
            running_mean: Initial running mean for normalization
            running_std: Initial running standard deviation
            epsilon: Small value to prevent division by zero
        """
        self.config = config
        self.scale_factor = scale_factor
        self.running_mean = running_mean or 0.0
        self.running_std = running_std or 1.0
        self.epsilon = epsilon
        self.alpha = 0.001  # Update rate for running statistics
        
    def shape_reward(
        self,
        base_reward: float,
        entropy: float,
        exploration_factor: float,
        progress: float
    ) -> float:
        """
        Shape the reward using various bonus terms
        
        Args:
            base_reward: Original environment reward
            entropy: Current policy entropy
            exploration_factor: Measure of exploration (e.g., state visitation count)
            progress: Task progress measure (0 to 1)
            
        Returns:
            Shaped reward
        """
        # Add exploration bonus
        exploration_bonus = self.config['exploration_bonus'] * exploration_factor
        
        # Add entropy bonus
        entropy_bonus = self.config['entropy_bonus'] * entropy
        
        # Add progress bonus
        progress_bonus = self.config['progress_bonus'] * progress
        
        # Combine all rewards
        total_reward = base_reward + exploration_bonus + entropy_bonus + progress_bonus
        
        # Update running statistics
        self.update_stats(total_reward)
        
        # Normalize and scale reward
        if self.config.get('use_reward_scaling', True):
            total_reward = self.normalize_reward(total_reward)
            total_reward = total_reward * self.scale_factor
            
        return total_reward
    
    def update_stats(self, reward: float):
        """Update running statistics for reward normalization"""
        self.running_mean = (1 - self.alpha) * self.running_mean + self.alpha * reward
        self.running_std = (1 - self.alpha) * self.running_std + self.alpha * (reward - self.running_mean) ** 2
        
    def normalize_reward(self, reward: float) -> float:
        """Normalize reward using running statistics"""
        std = np.sqrt(self.running_std + self.epsilon)
        return (reward - self.running_mean) / std
    
    def get_stats(self) -> Dict:
        """Get current reward statistics"""
        return {
            'running_mean': self.running_mean,
            'running_std': np.sqrt(self.running_std + self.epsilon)
        } 