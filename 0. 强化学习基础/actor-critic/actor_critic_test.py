import gym
import torch
import torch.nn.functional as F
import numpy as np
from actor_critic_visualize import PolicyNet, ValueNet, ActorCritic
import cv2
import os
from datetime import datetime

def evaluate_model(env, agent, num_episodes=10, render=True, save_video=True):
    returns = []
    
    if save_video:
        # 创建视频保存目录
        os.makedirs('videos', exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        video_path = f'videos/cartpole_test_{timestamp}.mp4'
        
        # 设置视频写入器
        frame_width = 600
        frame_height = 400
        fps = 30
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(video_path, fourcc, fps, (frame_width, frame_height))
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_return = 0
        done = False
        
        while not done:
            if render:
                # 渲染当前帧
                frame = env.render(mode='rgb_array')
                
                if save_video:
                    # 调整帧大小以匹配视频尺寸
                    frame = cv2.resize(frame, (frame_width, frame_height))
                    # OpenCV使用BGR格式，需要从RGB转换
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    video_writer.write(frame)
            
            # 选择动作
            action = agent.take_action(state)
            next_state, reward, done, _ = env.step(action)
            state = next_state
            episode_return += reward
        
        returns.append(episode_return)
        print(f'Episode {episode + 1}: Return = {episode_return}')
    
    env.close()
    if save_video:
        video_writer.release()
        print(f'\n视频已保存至: {video_path}')
    
    avg_return = np.mean(returns)
    print(f'\n平均回报: {avg_return:.2f}')
    return avg_return

if __name__ == "__main__":
    # 环境和模型参数设置
    env_name = 'CartPole-v0'
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    hidden_dim = 128
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 创建智能体
    agent = ActorCritic(
        state_dim=state_dim,
        hidden_dim=hidden_dim,
        action_dim=action_dim,
        actor_lr=1e-3,
        critic_lr=1e-2,
        gamma=0.98,
        device=device
    )

    # 加载模型参数
    # 注意：将这里的路径改为您实际保存的模型文件路径
    model_path = '/home/jaxxs/models/actor_critic_cartpole_20250214_154311.pth'  # 示例路径
    
    try:
        checkpoint = torch.load(model_path, map_location=device)
        agent.actor.load_state_dict(checkpoint['actor_state_dict'])
        agent.critic.load_state_dict(checkpoint['critic_state_dict'])
        print(f'成功加载模型: {model_path}')
    except Exception as e:
        print(f'加载模型失败: {str(e)}')
        exit(1)

    # 设置为评估模式
    agent.actor.eval()
    agent.critic.eval()

    # 测试模型并保存视频
    print('\n开始测试模型...')
    evaluate_model(env, agent, num_episodes=5, render=True, save_video=True) 