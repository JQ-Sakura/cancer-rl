import os
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import wandb
import yaml
import argparse
from rl_module.agents.ppo_agent import PPOAgent
from rl_module.env.cancer_env import CancerEnv
from rl_module.utils.adaptive_reward import AdaptiveRewardShaper
import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion
from datetime import datetime, timedelta
from torch.utils.tensorboard import SummaryWriter
import socket
import random
import tempfile
import shutil
import glob
import gc
from contextlib import nullcontext
import torch.nn as nn

def find_free_port():
    """Find a free port to use for distributed training"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
        # 确保端口真的被释放
        s.close()
        time.sleep(1)
    return port

def setup_distributed(rank, world_size, config):
    """Setup distributed training environment"""
    max_retries = 5
    retry_delay = 2
    
    for retry in range(max_retries):
        try:
            # 清理之前的环境
            cleanup_distributed()
            
            # 确保CUDA设备正确设置
            if torch.cuda.is_available():
                torch.cuda.set_device(rank)
            
            # 动态查找可用端口
            if rank == 0:
                port = find_free_port()
                # 将端口号写入临时文件
                with open('.port_file', 'w') as f:
                    f.write(str(port))
            
            # 等待主进程写入端口号
            if rank != 0:
                while not os.path.exists('.port_file'):
                    time.sleep(0.1)
                with open('.port_file', 'r') as f:
                    port = int(f.read().strip())
            
            # 设置环境变量
            os.environ['MASTER_ADDR'] = '127.0.0.1'
            os.environ['MASTER_PORT'] = str(port)
            os.environ['WORLD_SIZE'] = str(world_size)
            os.environ['RANK'] = str(rank)
            
            if rank == 0:
                print(f"Attempting to initialize distributed training on port {port} (attempt {retry + 1}/{max_retries})")
            
            # 初始化进程组
            dist.init_process_group(
                backend='nccl',
                init_method=f'tcp://127.0.0.1:{port}',
                world_size=world_size,
                rank=rank,
                timeout=timedelta(minutes=5)
            )
            
            # 确保所有进程都完成初始化
            dist.barrier()
            
            if rank == 0:
                print(f"Successfully initialized distributed training on port {port}")
                # 清理临时文件
                if os.path.exists('.port_file'):
                    os.remove('.port_file')
            return
            
        except Exception as e:
            if rank == 0:
                print(f"Attempt {retry + 1}/{max_retries} failed: {str(e)}")
                # 清理临时文件
                if os.path.exists('.port_file'):
                    os.remove('.port_file')
            
            # 清理环境
            cleanup_distributed()
            
            if retry == max_retries - 1:
                raise RuntimeError(f"Failed to initialize distributed training after {max_retries} attempts")
            
            # 在重试之前等待
            time.sleep(retry_delay * (retry + 1))

def is_port_available(port):
    """检查端口是否可用"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('127.0.0.1', port))
            return True
    except:
        return False

def cleanup_distributed():
    """Cleanup distributed training environment"""
    try:
        if dist.is_initialized():
            dist.barrier()  # 确保所有进程都到达这里
            dist.destroy_process_group()
    except Exception as e:
        print(f"Warning in cleanup_distributed: {str(e)}")
    finally:
        # 清理环境变量
        for env_var in ['MASTER_ADDR', 'MASTER_PORT', 'WORLD_SIZE', 'RANK', 'LOCAL_RANK']:
            if env_var in os.environ:
                del os.environ[env_var]
        
        # 清理临时文件
        if os.path.exists('.port_file'):
            try:
                os.remove('.port_file')
            except:
                pass
        
        # 清理CUDA缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # 重置CUDA设备
        if torch.cuda.is_available():
            torch.cuda.set_device('cuda:0')

def train(rank, world_size, config):
    """训练函数"""
    try:
        setup_distributed(rank, world_size, config)
        torch.manual_seed(42 + rank)
        np.random.seed(42 + rank)
        
        # 更新CUDA性能优化配置
        if torch.cuda.is_available():
            try:
                # 更新CUDA内存分配器配置，启用expandable_segments
                os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:64,garbage_collection_threshold:0.6,expandable_segments:True,backend:native'
                
                # 设置更激进的内存管理策略
                torch.cuda.set_per_process_memory_fraction(0.8)  # 增加到80%
                
                # 预留一些GPU内存防止碎片化
                reserved_memory = torch.zeros(int(1e9), device=f'cuda:{rank}')  # 预留1GB
                del reserved_memory
                
                # 设置更小的初始batch size和更新参数
                config['training']['batch_size'] = 4  # 直接从小batch size开始
                config['training']['ppo_epochs'] = 2
                config['training']['update_interval'] = 4
                config['training']['max_steps_per_episode'] = 32  # 进一步减少每个episode的步数
                
                # 强制进行垃圾回收
                torch.cuda.empty_cache()
                gc.collect()
                
            except Exception as e:
                print(f"Warning: CUDA optimization setup failed: {e}")
        
        # 启用CUDA图优化
        torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7 and torch.cuda.set_sync_debug_mode(0)
        
        if rank == 0:
            print("\n=== Training Configuration ===")
            print(f"Total Episodes: {config['training']['num_episodes']}")
            print(f"Steps per Episode: {config['training']['max_steps_per_episode']}")
            print(f"Update Interval: {config['training']['update_interval']}")
            print(f"Batch Size: {config['training']['batch_size']}")
            print(f"GPU Memory Limit: 85%")
            print(f"PyTorch Version: {torch.__version__}")
            print(f"CUDA Version: {torch.version.cuda}")
            print("-" * 35)
            print("Episode    Steps    Reward    IoU")
            print("-" * 35)

            run_name = f"run_{time.strftime('%Y%m%d_%H%M%S')}"
            wandb.init(
                project=config['wandb']['project'],
                config=config,
                name=run_name,
                sync_tensorboard=True,
                settings=wandb.Settings(start_method="thread")
            )
            
            # wandb配置
            wandb.define_metric("step")
            for metric in ["value_loss", "policy_loss", "learning_rate", "episode", 
                         "entropy", "avg_reward", "avg_iou", "dice", "avg_dice", "best_dice"]:  # 添加Dice相关指标
                wandb.define_metric(metric, step_metric="step")
            
            wandb.run.tags = ["training"]
            wandb.run.notes = "Training with enhanced metrics tracking including Dice coefficient"
            
            wandb.log({
                "step": 0,
                "value_loss": 0.0,
                "policy_loss": 0.0,
                "learning_rate": config['ppo']['learning_rate'],
                "episode": 0,
                "entropy": 0.0,
                "avg_reward": 0.0,
                "avg_iou": 0.0,
                "dice": 0.0,
                "avg_dice": 0.0,
                "best_dice": 0.0
            })

        # 创建环境
        env = CancerEnv(
            data_dir=config['paths']['data_dir'],
            config=config,
            model_path=config['paths']['model_path'],
            max_steps=config['training']['max_steps_per_episode'],
            device=f'cuda:{rank}',
            use_prompt=config['env']['use_prompt']
        )
        
        # 创建agent
        agent = PPOAgent(
            state_dim=env.observation_space,
            action_dim=env.action_space,
            device=f'cuda:{rank}',
            lr=config['ppo']['learning_rate'],
            gamma=config['ppo']['gamma'],
            epsilon=config['ppo']['epsilon'],
            c1=config['ppo']['c1'],
            c2=config['ppo']['c2'],
            memory_batch_size=8,  # 降低内存batch size
            gradient_clip=config['ppo']['clip_grad_norm'],
            use_amp=False,  # 禁用AMP
            gae_lambda=config['ppo']['gae_lambda']  # 添加GAE lambda参数
        )
        
        # 禁用所有高级优化
        torch._dynamo.config.suppress_errors = True
        torch._dynamo.config.verbose = False
        
        # 确保模型在GPU上
        agent.ac = agent.ac.to(f'cuda:{rank}')
        agent.ac.train()
        
        # 同步所有进程
        dist.barrier()
        
        # 简化的DDP配置
        agent.ac = DDP(
            agent.ac,
            device_ids=[rank],
            find_unused_parameters=False,  # 禁用未使用参数检查
            static_graph=True  # 启用静态图
        )
        
        # 使用基础的Adam优化器
        agent.optimizer = torch.optim.Adam(
            agent.ac.parameters(),
            lr=config['ppo']['learning_rate'],
            eps=1e-5
        )
        
        # 禁用所有额外的上下文管理器
        amp_context = nullcontext()
        sdpa_context = nullcontext()
        
        if rank == 0:
            print("Using simplified DDP configuration")
            
        # 初始化静态图
        with torch.no_grad():
            # 创建正确维度的dummy state
            dummy_state = {
                'volume': torch.zeros(1, 3, 64, 64, 64, device=f'cuda:{rank}'),  # (B, C, H, W, D)
                'current_mask': torch.zeros(1, 1, 64, 64, 64, device=f'cuda:{rank}'),  # (B, C, H, W, D)
                'entropy_map': torch.zeros(1, 1, 64, 64, 64, device=f'cuda:{rank}'),  # (B, C, H, W, D)
                'history_mask': torch.zeros(1, 1, 64, 64, 64, device=f'cuda:{rank}')   # (B, C, H, W, D)
            }
            
            # 将masks合并为一个tensor
            dummy_state['masks'] = torch.cat([
                dummy_state.pop('current_mask'),
                dummy_state.pop('entropy_map'),
                dummy_state.pop('history_mask')
            ], dim=1)  # 在通道维度上连接
            
            # 前向传播
            agent.ac(dummy_state)
            
        # 同步进程
        dist.barrier()
        
        reward_shaper = AdaptiveRewardShaper(
            window_size=config['env']['reward_shaping']['window_size'],
            initial_scale=config['env']['reward_shaping']['initial_scale'],
            min_scale=config['env']['reward_shaping']['min_scale'],
            max_scale=config['env']['reward_shaping']['max_scale'],
            adaptation_rate=config['env']['reward_shaping']['adaptation_rate']
        )
        
        # 配置学习率调度器
        if config['ppo']['adaptive_lr']:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                agent.optimizer,
                T_max=config['training']['num_episodes'],
                eta_min=config['training']['min_lr']
            )
        
        episode = 0
        best_iou = 0.0
        patience_counter = 0
        
        if rank == 0:
            os.makedirs(config['paths']['log_dir'], exist_ok=True)
            writer = SummaryWriter(log_dir=os.path.join(config['paths']['log_dir'], 'tensorboard'))
        
        # 记录开始时间
        start_time = time.time()
        
        # 初始化metrics累积器
        metrics_accumulator = {
            'value_loss': [],
            'policy_loss': [],
            'entropy': [],
            'approx_kl': [],
            'clip_fraction': [],
            'explained_variance': []
        }
        
        while episode < config['training']['num_episodes']:
            try:
                # 预分配内存缓冲区并清理
                if episode % 5 == 0:  # 每5个episode清理一次
                    torch.cuda.empty_cache()
                    gc.collect()
                
                state, _ = env.reset()
                episode_reward = 0
                episode_steps = 0
                episode_iou = 0
                episode_dice = 0  # 添加Dice追踪
                best_dice = 0  # 添加最佳Dice追踪
                
                # 使用新的上下文管理器
                with amp_context, sdpa_context:
                    while True:
                        action, log_prob, value = agent.select_action(state)
                        next_state, reward, terminated, truncated, info = env.step(action)
                        
                        # 更新Dice值
                        current_dice = info.get('dice', 0.0)
                        episode_dice = max(episode_dice, current_dice)
                        best_dice = max(best_dice, current_dice)
                        
                        is_boundary = info.get('is_boundary', False)
                        if not is_boundary and 'current_mask' in next_state:
                            current_mask = next_state['current_mask']
                            boundary = binary_dilation(current_mask) & ~binary_erosion(current_mask)
                            action_point = tuple(map(int, action))
                            if 0 <= action_point[0] < boundary.shape[0] and \
                               0 <= action_point[1] < boundary.shape[1] and \
                               0 <= action_point[2] < boundary.shape[2]:
                                is_boundary = boundary[action_point]
                        
                        # 确保value是标量值
                        value_scalar = float(value.item() if torch.is_tensor(value) else value)
                        
                        shaped_reward = reward_shaper.shape_reward(
                            base_reward=reward,
                            iou=info['current_iou'],
                            entropy=value_scalar,
                            exploration_factor=episode_steps / config['training']['max_steps_per_episode'],
                            steps_taken=episode_steps,
                            is_boundary=is_boundary
                        )
                        
                        agent.memory.push(
                            state,
                            action,
                            shaped_reward,
                            value_scalar,  # 使用转换后的标量值
                            log_prob,
                            terminated or truncated
                        )
                        
                        state = next_state
                        episode_reward += shaped_reward
                        episode_steps += 1
                        episode_iou = max(episode_iou, info['current_iou'])
                        
                        if terminated or truncated:
                            break
                
                # 更新batch size动态调整
                if (episode + 1) % config['training']['update_interval'] == 0:
                    try:
                        # 同步所有进程
                        dist.barrier()
                        
                        # 确保优化器梯度为零
                        agent.optimizer.zero_grad()
                        
                        # 计算当前batch size
                        current_batch_size = min(config['training']['batch_size'], len(agent.memory))
                        
                        # 进行策略更新
                        metrics = agent.update(
                            batch_size=current_batch_size,
                            epochs=config['training']['ppo_epochs'],
                            accumulation_steps=8  # 增加梯度累积步数
                        )
                        
                        # 同步梯度
                        dist.barrier()
                        
                        # 更新metrics累积器
                        if metrics:
                            for key in metrics_accumulator:
                                if key in metrics:
                                    metrics_accumulator[key].append(metrics[key])
                        
                        # 清理内存
                        torch.cuda.empty_cache()
                        gc.collect()
                        
                    except Exception as e:
                        print(f"Warning: Update failed with error: {str(e)}")
                        # 确保所有进程都继续执行
                        dist.barrier()
                        continue
                
                # 每个episode结束时记录所有指标
                if rank == 0:
                    # 计算平均奖励和IoU
                    avg_reward = episode_reward / episode_steps if episode_steps > 0 else 0
                    avg_iou = episode_iou / episode_steps if episode_steps > 0 else 0
                    
                    # 获取奖励整形器的统计信息
                    shaper_stats = reward_shaper.get_stats()
                    
                    # 计算当前metrics的平均值
                    current_metrics = {}
                    for key in metrics_accumulator:
                        values = metrics_accumulator[key]
                        current_metrics[key] = np.mean(values) if values else 0.0
                    
                    # 合并所有指标到一个字典中
                    log_dict = {
                        'step': episode,
                        'episode': episode,
                        'global_step': episode * config['training']['max_steps_per_episode'],
                        
                        # PPO指标
                        'value_loss': current_metrics['value_loss'],
                        'policy_loss': current_metrics['policy_loss'],
                        'entropy': current_metrics['entropy'],
                        'learning_rate': agent.optimizer.param_groups[0]['lr'],
                        
                        # 性能指标
                        'avg_reward': avg_reward,
                        'avg_iou': avg_iou,
                        
                        # 详细训练指标
                        'training/episode_steps': episode_steps,
                        'training/episode_reward': episode_reward,
                        'training/episode_iou': episode_iou,
                        'training/best_iou': best_iou,
                        
                        # 奖励相关指标
                        'reward/raw': reward,
                        'reward/shaped': shaped_reward,
                        'reward/scale': shaper_stats['current_scale'],
                        'reward/avg': shaper_stats['avg_reward'],
                        
                        # 环境相关指标
                        'env/current_iou': info['current_iou'],
                        'env/region_score': info.get('region_score', 0.0),
                        'env/avg_iou': shaper_stats['avg_iou'],
                        
                        # PPO详细指标
                        'ppo/approx_kl': current_metrics['approx_kl'],
                        'ppo/clip_fraction': current_metrics['clip_fraction'],
                        'ppo/explained_variance': current_metrics['explained_variance'],
                        
                        # Dice相关指标
                        'dice': episode_dice,
                        'avg_dice': episode_dice / episode_steps if episode_steps > 0 else 0.0,
                        'best_dice': best_dice
                    }
                    
                    # 记录系统指标
                    try:
                        log_dict.update({
                            'system/gpu_memory_allocated': torch.cuda.memory_allocated(rank) / 1024**3,
                            'system/gpu_memory_cached': torch.cuda.memory_reserved(rank) / 1024**3,
                            'system/gpu_utilization': float(torch.cuda.utilization(rank)) if hasattr(torch.cuda, 'utilization') else 0.0
                        })
                    except Exception as e:
                        if rank == 0:
                            print(f"Warning: Could not log system metrics: {e}")
                    
                    # 每100个episode记录分布信息
                    if episode % 100 == 0:
                        log_dict.update({
                            'distributions/rewards': wandb.Histogram(np.array([episode_reward])),
                            'distributions/iou': wandb.Histogram(np.array([episode_iou])),
                            'distributions/actions': wandb.Histogram(action.numpy() if isinstance(action, torch.Tensor) else action),
                            'distributions/values': wandb.Histogram(np.array([float(value_scalar)]))
                        })
                    
                    # 一次性记录所有指标
                    wandb.log(log_dict, step=episode)
                    
                    # 更新tensorboard
                    if writer is not None:
                        for key, value in log_dict.items():
                            if isinstance(value, (int, float)):
                                writer.add_scalar(key, value, episode)
                
                if episode_iou > best_iou:
                    best_iou = episode_iou
                    patience_counter = 0
                    if rank == 0:
                        avg_iou = episode_iou / episode_steps if episode_steps > 0 else 0
                        avg_dice = episode_dice / episode_steps if episode_steps > 0 else 0
                        print(f"\n[New Record] Best IoU: {best_iou:.4f}, Avg IoU: {avg_iou:.4f}, Best Dice: {best_dice:.4f}, Avg Dice: {avg_dice:.4f}")
                else:
                    patience_counter += 1
                    
                # Display training status every 10 episodes
                if rank == 0 and episode % 10 == 0:
                    avg_iou = episode_iou / episode_steps if episode_steps > 0 else 0
                    avg_dice = episode_dice / episode_steps if episode_steps > 0 else 0
                    print(f"Episode {episode}: Current IoU: {episode_iou:.4f}, Avg IoU: {avg_iou:.4f}, Current Dice: {episode_dice:.4f}, Avg Dice: {avg_dice:.4f}")
                
                if patience_counter >= config['training']['patience']:
                    current_lr = agent.optimizer.param_groups[0]['lr']
                    if current_lr <= config['training']['min_lr']:
                        if rank == 0:
                            print("\n=== Training Complete ===")
                            print(f"Best IoU: {best_iou:.4f}")
                        break
                    
                    new_lr = current_lr * config['training']['lr_decay_factor']
                    for param_group in agent.optimizer.param_groups:
                        param_group['lr'] = new_lr
                    if rank == 0:
                        print(f"\n[Learning Rate] {current_lr:.2e} -> {new_lr:.2e}\n")
                    patience_counter = 0
                
                if rank == 0 and (episode + 1) % config['training']['save_interval'] == 0:
                    agent.save_checkpoint(
                        os.path.join(config['paths']['log_dir'], f'checkpoint_{episode+1}.pt'),
                        episode,
                        {'best_iou': best_iou}
                    )
                    
                # 定期清理CUDA缓存
                empty_cache_freq = config.get('model_config', {}).get('memory', {}).get('empty_cache_freq', 50)
                if episode % empty_cache_freq == 0:
                    torch.cuda.empty_cache()
                
                # 监控GPU内存使用
                if rank == 0 and episode % 10 == 0:
                    allocated = torch.cuda.memory_allocated(rank) / 1024**3
                    reserved = torch.cuda.memory_reserved(rank) / 1024**3
                    print(f"GPU {rank} Memory: Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
                
                episode += 1
                
            except Exception as e:
                if rank == 0:
                    print(f"\n[Error] {str(e)}")
                    try:
                        wandb.log({
                            'step': episode,
                            'error': str(e)
                        })
                    except:
                        pass
                continue
                
    except Exception as e:
        if rank == 0:
            print(f"\n[Critical Error] {str(e)}")
            try:
                wandb.log({
                    'critical_error': str(e)
                })
            except:
                pass
        raise
        
    finally:
        if rank == 0:
            try:
                # Log final results
                wandb.log({
                    'final/best_iou': best_iou,
                    'final/episodes_completed': episode,
                    'final/training_time': time.time() - start_time
                })
                wandb.finish()
            except:
                pass
            
            if 'writer' in locals():
                writer.close()
        cleanup_distributed()

def main():
    try:
        # 确保CUDA可用
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. Please check your GPU setup.")
            
        with open('rl_module/configs/train_config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        world_size = torch.cuda.device_count()
        if world_size < 1:
            raise RuntimeError("No CUDA devices available")
            
        # 清理之前的进程和临时文件
        cleanup_distributed()
        if os.path.exists('.port_file'):
            os.remove('.port_file')
        
        # 设置多进程启动方式
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass  # 已经设置过了
        
        # 设置CUDA设备可见性
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(i) for i in range(world_size))
        
        # 使用spawn启动进程
        mp.spawn(
            train,
            args=(world_size, config),
            nprocs=world_size,
            join=True,
            daemon=False,
            start_method='spawn'
        )
            
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        cleanup_distributed()
    except Exception as e:
        print(f"\n[Error] {str(e)}")
        cleanup_distributed()
        raise
    finally:
        cleanup_distributed()
        # 确保清理临时文件
        if os.path.exists('.port_file'):
            os.remove('.port_file')

if __name__ == '__main__':
    main() 