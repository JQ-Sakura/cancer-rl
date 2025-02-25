import os
import torch
import numpy as np
from datetime import datetime
import json
import logging
from tqdm import tqdm
import wandb
import yaml
from typing import Dict, Optional

from rl_module.env.cancer_env import CancerEnv
from rl_module.agents.ppo_agent import PPOAgent

DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'configs/train_config.yaml')

def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 动态设置设备
    config['training']['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Validate required paths
    if not os.path.exists(config['paths']['data_dir']):
        raise FileNotFoundError(f"Data directory not found at: {config['paths']['data_dir']}")
    
    if config['paths']['model_path'] and not os.path.exists(config['paths']['model_path']):
        raise FileNotFoundError(f"Model file not found at: {config['paths']['model_path']}")
    
    return config

def setup_logging(log_dir: str) -> None:
    """Setup logging configuration"""
    os.makedirs(log_dir, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'training.log')),
            logging.StreamHandler()
        ]
    )

def train(config: Dict) -> None:
    """
    Train PPO agent for cancer region detection
    
    Args:
        config: Configuration dictionary
    """
    # Setup logging
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join(config['paths']['log_dir'], f'run_{timestamp}')
    setup_logging(log_dir)
    
    # Initialize wandb
    if config['training']['use_wandb']:
        wandb.init(
            project=config['wandb']['project'],
            entity=config['wandb']['entity'],
            config=config
        )
    
    # Create environment
    env = CancerEnv(
        data_dir=config['paths']['data_dir'],
        config=config,
        model_path=config['paths']['model_path'],
        max_steps=config['training']['max_steps_per_episode'],
        device=config['training']['device']
    )
    
    # Initialize agent
    agent = PPOAgent(
        state_dim=env.observation_space['volume'].shape,
        action_dim=3,  # (z, y, x) coordinates
        device=config['training']['device'],
        lr=config['ppo']['learning_rate'],
        gamma=config['ppo']['gamma'],
        epsilon=config['ppo']['epsilon'],
        c1=config['ppo']['c1'],
        c2=config['ppo']['c2']
    )
    
    # Training loop
    episode_rewards = []
    best_reward = float('-inf')
    
    for episode in range(config['training']['num_episodes']):
        # Reset environment
        state, _ = env.reset()
        episode_reward = 0
        
        # Episode loop
        for step in range(config['training']['max_steps_per_episode']):
            # Select action
            action, value, log_prob = agent.select_action(state)
            
            # Take action in environment
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Store transition
            agent.memory.push(
                state=state,
                action=action,
                reward=reward,
                value=value,
                log_prob=log_prob,
                done=done
            )
            
            state = next_state
            episode_reward += reward
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        
        # Log episode results
        logging.info(f"Episode {episode + 1}: Reward = {episode_reward:.4f}, IoU = {info['current_iou']:.4f}")
        
        if config['training']['use_wandb']:
            wandb.log({
                "episode_reward": episode_reward,
                "current_iou": info['current_iou'],
                "steps_taken": info['steps_taken']
            })
        
        # Update policy if enough episodes collected
        if (episode + 1) % config['training']['update_interval'] == 0:
            update_info = agent.update(
                batch_size=config['training']['batch_size'],
                epochs=config['training']['ppo_epochs']
            )
            
            logging.info(f"Policy updated: {update_info}")
            if config['training']['use_wandb']:
                wandb.log(update_info)
        
        # Save model if it's the best so far
        if episode_reward > best_reward:
            best_reward = episode_reward
            model_path = os.path.join(log_dir, 'best_model.pth')
            torch.save({
                'episode': episode,
                'model_state_dict': agent.ac.state_dict(),
                'optimizer_state_dict': agent.optimizer.state_dict(),
                'reward': best_reward
            }, model_path)
        
        # Regular model saving
        if (episode + 1) % config['training']['save_interval'] == 0:
            model_path = os.path.join(log_dir, f'model_episode_{episode+1}.pth')
            torch.save({
                'episode': episode,
                'model_state_dict': agent.ac.state_dict(),
                'optimizer_state_dict': agent.optimizer.state_dict(),
                'reward': episode_reward
            }, model_path)
            
            # Save training statistics
            stats_path = os.path.join(log_dir, 'training_stats.json')
            with open(stats_path, 'w') as f:
                json.dump({
                    'episode_rewards': episode_rewards,
                    'best_reward': best_reward
                }, f)
    
    # Final model save
    final_model_path = os.path.join(log_dir, 'final_model.pth')
    torch.save({
        'episode': config['training']['num_episodes'],
        'model_state_dict': agent.ac.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
        'reward': episode_rewards[-1]
    }, final_model_path)
    
    if config['training']['use_wandb']:
        wandb.finish()

if __name__ == '__main__':
    try:
        config = load_config(DEFAULT_CONFIG_PATH)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print(f"Please ensure the config file exists at: {DEFAULT_CONFIG_PATH}")
        exit(1)
    except Exception as e:
        print(f"Error loading config: {e}")
        exit(1)
    
    train(config) 