import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import sys
import os
import math
import gym_cutting_stock
import gymnasium as gym
from model import EnvironmentSimulate, ProductEncoder, StockEncoder, PPO
env = gym.make(
    "gym_cutting_stock/CuttingStock-v0",
    render_mode="human",
    min_w=50,          
    min_h=50,          
    max_w=100,         
    max_h=100,         
    num_stocks=100,     
    max_product_type=25,  
    max_product_per_type=20,  
    seed=42            
)

def load_model(model, checkpoint_path="ppo_checkpoint.pth"):
    """Load the state of actor_net, value_net, and optimizer."""
    # Load trọng số cho actor_net
    actor_checkpoint = torch.load("actor_" + checkpoint_path)
    model.policy_net.load_state_dict(actor_checkpoint['actor_net_state_dict'])
    print("Actor network weights loaded.")
    model.policy_net.eval()
    # Load trọng số cho value_net và optimizer
    critic_checkpoint = torch.load("critic_" + checkpoint_path)
    model.value_net.load_state_dict(critic_checkpoint['value_net_state_dict'])
    model.value_net.eval()
    model.optimizer.load_state_dict(critic_checkpoint['optimizer_state_dict'])
    print("Critic network weights and optimizer state loaded.")

    # Trả lại số episode đã lưu
    episode = actor_checkpoint['episode'] if 'episode' in actor_checkpoint else 0
    print(f"Checkpoint loaded from episode {episode}")
    
    return episode


def test(model: PPO, environment, num_episodes, max_steps):
    """Kiểm tra hiệu suất của mô hình PPO."""
    total_rewards = []  # Để lưu tổng phần thưởng cho mỗi episode
    
    for episode in range(num_episodes):
        observation, _ = environment.reset()
        model.observation.reset(observation)
        model.masking.reset_mask()
        episode_rewards = 0

        for step in range(max_steps):
            # Lấy hành động từ mô hình
            place, action, log_prob, entropy = model.get_action(observation)
            while place["size"] == [1000,1000]:
                place, action, log_prob, entropy = model.get_action(observation)
            # Thực hiện hành động trong môi trường
            observation, reward, terminated, truncated, info = environment.step(place)
            reward = model.calculate_reward(observation, place, info)
            episode_rewards += reward
            
            if terminated or truncated:
                break
        print(info)
        print(f"Episode {episode + 1} ended with Total Reward: {episode_rewards}")
        total_rewards.append(episode_rewards)

    # In kết quả trung bình
    avg_reward = np.mean(total_rewards)
    print(f"Average Reward over {num_episodes} episodes: {avg_reward}")
    return total_rewards

MAX_STEPS = 200
if __name__ == "__main__":
    device ='cuda' if torch.cuda.is_available() else 'cpu'
    # Load mô hình đã lưu
    saved_model_path = "ppo_model_final.pth"

    agent = PPO(100, 100, 100, 25, 128, "Test", device)
    load_model(agent)
    print("Loaded model from:", saved_model_path)
    
    # Kiểm tra mô hình
    print("\nTesting the model...")
    test_episodes = 10
    test_rewards = test(agent, env, test_episodes, MAX_STEPS)
    print("Test Rewards:", test_rewards)
    env.close()
