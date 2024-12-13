import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import math
"""
TO_DO LIST:
- Create a self-observation. We just need to encode first when the environment reset. 
Modify the self-observation easily with access by stock_index and product_index.
- Enhance the masking mechanism. We introduce a big negative reward when choose an invalid combination.
- The reward will by locally in 1 stocks.
If we reuse stocks, it will be a positive reward and otherwise.
When placing stock, we calculate the compactness, and retrieve a postive bonus base on compactness.

"""
from policy import Policy

class ProductEncoder(nn.Module):
    """Encodes the input sequence using 1D convolution."""
    def __init__(self, input_size, hidden_size):
        super(ProductEncoder, self).__init__()
        self.conv = nn.Conv1d(input_size, hidden_size, kernel_size=1)
    """
    Parameter: x - A torch tensor with shape (2*max_num_product, 3)
    Return: output - A torch tensor with shape (2*max_num_product, hidden_size)
    Usage:
        Extract feature from tuple 3: width - height - quantity of each product and its rotation
    """
    def forward(self, x):
        print("Product: ", x.shape)
        x = x.unsqueeze(0).permute(0,2,1)
        print("Product: ", x.shape)
        output = self.conv(x)  
        print("Product: ", output.shape)
        output = output.squeeze(0).permute(1, 0) 
        print("Product: ", output.shape)
        return output
    

class StockEncoder(nn.Module):
    """Encodes the states using 2D Convolution."""
    def __init__(self, hidden_size, map_size):
        super(StockEncoder, self).__init__()
        self.conv1 = nn.Conv2d(1, hidden_size // 4, stride=2, kernel_size=1)
        self.conv2 = nn.Conv2d(hidden_size // 4, hidden_size // 2, stride=2, kernel_size=1)
        self.conv3 = nn.Conv2d(hidden_size // 2, hidden_size, kernel_size=(math.ceil(map_size[0] / 4), math.ceil(map_size[1] / 4)))

    """
    Parameter: input - A tensor with shape (num_stock, max_w, max_h)
    Return: output - A tensor with shape (num_stock, hidden_size)
    Usage:
        Extract features from stock.
    """
    def forward(self, input):
        # Ensure the input tensor is compatible with Conv2D
        output = torch.where(input != -1, torch.tensor(0.0, dtype=torch.float32, device=input.device), torch.tensor(1.0, dtype=torch.float32, device=input.device))
        output = output.unsqueeze(1)  # Add channel dimension (N, C=1, H, W)
        print("Stock: ", output.shape)
        output = F.leaky_relu(self.conv1(output))
        print("Stock: ", output.shape)
        output = F.leaky_relu(self.conv2(output))
        print("Stock: ", output.shape)
        output = self.conv3(output).squeeze(-1).squeeze(-1)  # Remove height and width dimensions
        print("Stock: ", output.shape)
        return output

    
class DQN(nn.Module):
    """
    Parameter: x - A 1D tensor of concatenated product's feature and stock's feature
    Return: output - A tensor with shape (2*num_stock*max_product_type) to compute the compatible score of stock - product 

    """

    """
    Parameter: x - A 1D tensor of 1 stock - 1 product
    Return: output - A tensor with shape (1,) to estimate the compatible score of stock - product.
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size, True)
        self.fc2 = nn.Linear(hidden_size, hidden_size, False)
        self.fc3 = nn.Linear(hidden_size, output_size, False)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        print(x.shape)
        return x


class Masking():
    """
    Provides a masking mechanism to ensure only valid actions are taken.
    """
    def __init__(self, num_actions):
        self.mask = torch.zeros(num_actions)
    
    def update_mask(self, invalid_actions):
        """
        Update the mask to block invalid actions.
        
        :param invalid_actions: List of indices corresponding to invalid actions.
        """
        self.mask[invalid_actions] = 1

    def reset_mask(self):
        """Reset the mask to allow all actions."""
        self.mask = torch.zeros_like(self.mask) 
    
    def apply_mask(self, logits):
        """
        Apply the mask to the logits to ensure invalid actions are not selected.
        
        :param logits: Output from the model before applying softmax.
        :return: Masked logits.
        """
        return logits + (self.mask * -1e9)  


class PPO(nn.Module, Policy):
    def __init__(self, id, num_stock, max_w, max_h, num_product, hidden_size, lr=3e-4, gamma=0.99, clip_epsilon=0.2, value_loss_coef=0.5, entropy_coef=0.01):
        nn.Module.__init__(self)
        Policy.__init__(self, id)
        self.num_product = num_product
        self.num_stock = num_stock
        self.max_w = max_w
        self.max_h = max_h
        self.product_encoder = ProductEncoder(3, hidden_size)
        self.stock_encoder = StockEncoder(num_stock, hidden_size, (max_w, max_h))
        self.policy_net = DQN(num_stock*hidden_size + 2*num_product*hidden_size, hidden_size, 2*num_stock*num_product)  
        # Value Network (Critic)
        self.value_net = DQN(2*hidden_size, hidden_size, 1) # A simple fully connected layer for value estimation the compatible score of stock-product

        self.masking = Masking(2*num_stock*num_product)
        # Optimizer
        self.optimizer = optim.Adam(list(self.policy_net.parameters()) + list(self.value_net.parameters()), lr=lr)

        # Hyperparameters for PPO
        self.gamma = gamma  # Discount factor
        self.clip_epsilon = clip_epsilon  # PPO clip epsilon for objective function
        self.value_loss_coef = value_loss_coef  # Coefficient for value loss
        self.entropy_coef = entropy_coef  # Coefficient for entropy loss
    
    def encode(self, observation):
        product_features = []
        for product in observation["products"]:
            width, height = product["size"]
            quantity = product["quantity"]
            product_features.append([width, height, quantity])
            product_features.append([height, width, quantity]) 
        while len(product_features) < 2 * self.num_product:
            product_features.append([1000, 1000, 0])

        product_tensor = torch.tensor(product_features, dtype=torch.float)
        stock_tensor = torch.tensor(observation["stocks"], dtype=torch.float)
        encoded_product = self.product_encoder(product_tensor)
        encoded_stock = self.stock_encoder(stock_tensor)
        act_input = torch.cat((torch.flatten(encoded_stock), torch.flatten(encoded_product)))
        act_chosen = self.policy_net(act_input)
        return act_input, act_chosen
    
    def _can_place_(self, stock, position, size):
        x, y = position
        w, h = size
        return np.all(stock[x:x + w, y:y + h] == -1)
    
    def place_product(self, observation, sheet_index, product_index, rotate):
        stock = observation["stocks"][sheet_index]
        if product_index > len(observation["products"]):
            attribute = "Product_Error"
            return None, attribute
        prod_info = observation["products"][product_index]
        if observation["products"][product_index]["quantity"] == 0:
            attribute = "Product_Error"
            return None, attribute
        prod_w, prod_h = prod_info["size"]
        if rotate == 1:
            prod_w, prod_h = prod_h, prod_w 

        max_w, max_h = stock.shape
        best_pos_x, best_pos_y = None, None
        best_distance = float('inf')  

        for pos_x in range(max_w - prod_w + 1):
            for pos_y in range(max_h - prod_h + 1):
                if stock[pos_x][pos_y] != -1: 
                    continue
                if self._can_place_(stock, (pos_x, pos_y), (prod_w, prod_h)):
                    distance = (pos_x + prod_w/2)**2 + (pos_y + prod_h/2)**2
                    if distance < best_distance:  
                        best_pos_x = pos_x
                        best_pos_y = pos_y
                        best_distance = distance  

        if best_pos_x is None: 
            attribute = "Stock_Error" 
            return None, attribute
        attribute = "Complete"
        return {
            "stock_idx": sheet_index, 
            "size": [prod_w, prod_h],
            "position": [best_pos_x, best_pos_y],
        }, attribute
    
    def get_action(self, observation, info):
        """
        Given the current state, get the action from the policy network.
        """
        placed = None
        _, logits = self.encode(observation)  # Get the logits (raw action probabilities)
        while placed is None:
            logits_masked = self.masking.apply_mask(logits)
            dist = torch.distributions.Categorical(logits=logits_masked)
            action = dist.sample()  # Sample an action from the distribution
            sheet_index = action // (2 * self.num_product)
            product_index = (action % (2 * self.num_product)) // 2
            rotate = action % 2  # 0 for no rotation, 1 for rotation
            placed, attribute = self.place_product(observation, sheet_index, product_index, rotate)
            if attribute != "Complete":
                self.masking.update_mask(observation, sheet_index, product_index, rotate, attribute)

        return placed, action, dist.log_prob(action), dist.entropy()  # Return action, log probability, and entropy
    
    def calculate_reward(observation, action, num_step):
        if np.all(observation["stocks"][action["stock_idx"]] < 0):
            return 10/num_step
        else:
            return -10/num_step
    
    def compute_advantage(self, rewards, values, next_value, done):
        """
        Compute the advantage for each timestep using Generalized Advantage Estimation (GAE).
        
        :param rewards: List of rewards for each timestep
        :param values: List of value estimates for each timestep
        :param next_value: Value estimate for the next state
        :param done: Boolean indicating if the episode is done
        
        :return: A list of advantages
        """
        advantages = []
        gae = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * next_value * (1 - done) - values[t]
            gae = delta + self.gamma * 0.95 * (1 - done) * gae  # GAE lambda = 0.95
            advantages.insert(0, gae)
            next_value = values[t]
        return advantages

    def update(self, states, actions, log_probs_old, returns, advantages):
        """
        Update the policy using the PPO clipped objective.
        
        :param states: A batch of states
        :param actions: A batch of actions
        :param log_probs_old: Old log probabilities for the actions
        :param returns: The returns for each timestep
        :param advantages: The advantages for each timestep
        """
        # Convert inputs to tensors
        states = torch.stack(states)
        actions = torch.stack(actions)
        log_probs_old = torch.stack(log_probs_old)
        returns = torch.stack(returns)
        advantages = torch.stack(advantages)

        # Get new action probabilities and values
        logits = self.policy_net(states)
        dist = torch.distributions.Categorical(logits=logits)
        log_probs_new = dist.log_prob(actions)
        entropy = dist.entropy().mean()
        values = self.value_net(states).squeeze()

        # Compute the ratio of new and old probabilities
        ratio = torch.exp(log_probs_new - log_probs_old)

        # Clipped objective for PPO
        clip_loss = torch.min(ratio * advantages, torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages)
        policy_loss = -clip_loss.mean()

        # Value function loss (MSE)
        value_loss = F.mse_loss(values, returns)

        # Total loss: Policy loss + Value loss + Entropy loss (to encourage exploration)
        loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy

        # Perform the optimization step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def evaluate(self, states, actions):
        """
        Evaluate the policy network given a batch of states and actions.
        
        :param states: A batch of states
        :param actions: A batch of actions
        :return: Log probabilities, entropy, and state value estimates
        """
        logits = self.policy_net(states)
        dist = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()
        values = self.value_net(states).squeeze()
        return log_probs, entropy, values

    def train_step(self, states, actions, rewards, next_states, dones):
        """
        Perform a single training step.
        
        :param states: Current states
        :param actions: Actions taken
        :param rewards: Rewards obtained
        :param next_states: Next states
        :param dones: Whether the episode is done
        :return: The loss value for the training step
        """
        # Estimate the values of the states and compute the returns
        values = self.value_net(states).squeeze()
        next_values = self.value_net(next_states).squeeze()
        returns = rewards + self.gamma * next_values * (1 - dones)
        advantages = self.compute_advantage(rewards, values, next_values, dones)

        # Update the policy and value networks
        self.update(states, actions, rewards, returns, advantages)

        return advantages

    