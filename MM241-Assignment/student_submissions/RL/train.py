import gym_cutting_stock
import gymnasium as gym
import torch
import numpy as np
from model import PPO

# Create the environment
env = gym.make(
    "gym_cutting_stock/CuttingStock-v0",
    render_mode="human",
    min_w=50,          # Kích thước nhỏ nhất của sản phẩm (chiều rộng)
    min_h=50,          # Kích thước nhỏ nhất của sản phẩm (chiều cao)
    max_w=100,         # Kích thước lớn nhất của sản phẩm (chiều rộng)
    max_h=100,         # Kích thước lớn nhất của sản phẩm (chiều cao)
    num_stocks=100,     # Số lượng stock
    max_product_type=25,  # Số loại sản phẩm tối đa
    max_product_per_type=20,  # Số lượng tối đa mỗi loại sản phẩm|
    seed=42            # Seed để tái tạo kết quả
)

#device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'
print('Training using:', device)
agent = PPO(100, 100, 100, 25, 128, "Train", device, gamma=0.99)
NUM_EPISODES = 100
MAX_STEPS = 100

def save_checkpoint(model, episode, checkpoint_path="ppo_checkpoint.pth"):
    """Save the state of actor_net, value_net, and optimizer."""
    torch.save({
        'episode': episode,
        'actor_net_state_dict': model.policy_net.state_dict(),
    }, "actor_"+checkpoint_path)
    torch.save({'value_net_state_dict': model.value_net.state_dict(),
        'optimizer_state_dict': model.optimizer.state_dict(),
    }, "critic_"+checkpoint_path)
    print(f"Checkpoint saved at episode {episode}")


def load_checkpoint(model, checkpoint_path="ppo_checkpoint.pth"):
    """Load the state of actor_net, value_net, and optimizer."""
    # Load trọng số cho actor_net
    actor_checkpoint = torch.load("actor_" + checkpoint_path)
    model.policy_net.load_state_dict(actor_checkpoint['actor_net_state_dict'])
    print("Actor network weights loaded.")

    # Load trọng số cho value_net và optimizer
    critic_checkpoint = torch.load("critic_" + checkpoint_path)
    model.value_net.load_state_dict(critic_checkpoint['value_net_state_dict'])
    model.optimizer.load_state_dict(critic_checkpoint['optimizer_state_dict'])
    print("Critic network weights and optimizer state loaded.")

    # Trả lại số episode đã lưu
    episode = actor_checkpoint['episode'] if 'episode' in actor_checkpoint else 0
    print(f"Checkpoint loaded from episode {episode}")
    
    return episode


def train(model: PPO, environment, num_episodes, max_steps, checkpoint_interval=5):
    ep = 0 #load_checkpoint(model)
    for episode in range(num_episodes):
        observation, info = environment.reset()
        model.observation.reset(observation)
        model.masking.reset_mask()
        episode_rewards = 0

        # Prepare containers for data to be accumulated
        states = []
        actions = []
        log_probs = []
        rewards = []
        dones = []

        for step in range(max_steps):
            current_obs = model.encode()  # Efficient encoding of observation
            place, action, log_prob, entropy = model.inner_get_action(observation)

            # Environment step
            observation, _, terminated, truncated, info = environment.step(place)
            reward = model.calculate_reward(observation, place, info)
            # Collect data
            states.append(current_obs)
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            dones.append(terminated or truncated)  # Combined termination check

            # Accumulate reward for this episode
            episode_rewards += reward

            # Break if episode is done
            if terminated or truncated:
                break

        # Use the last state for value estimation (if applicable)
        last_value = model.value_net(model.encode()).item()
        states = torch.stack(states).detach().requires_grad_(True)
        values = model.value_net(states).squeeze(-1)  # Value for each state
        returns, advantages = model.compute_returns_and_advantages(rewards, values, dones, last_value)

        # Update the policy and value networks
        model.update(states, actions, log_probs, returns, advantages)

        # Print the progress
        print(f"Episode {episode + ep + 1}/{num_episodes}, Total Reward: {episode_rewards}")
        print(info)
        with open("result.txt", "a") as f:  # Open the file in append mode
            f.write(f"Episode {episode + ep + 1}/{num_episodes}, Total Reward: {episode_rewards}\n")
            f.write(f"{info}\n")
        # Save checkpoint periodically
        if (episode + 1) % checkpoint_interval == 0:
            save_checkpoint(model, episode + 1 + ep)


if __name__ == "__main__":
    # Reset the environment
    # Test GreedyPolicy
    train(agent, env, NUM_EPISODES, MAX_STEPS)
    env.close()
