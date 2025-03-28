import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import torch.nn.functional as F
import time
import os

class PPONetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super(PPONetwork, self).__init__()
        
        # Shared feature extractor
        self.shared_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        
        # Policy head
        self.policy_mean = nn.Linear(hidden_dim, output_dim)
        self.policy_std = nn.Parameter(torch.zeros(output_dim))
        
        # Value head
        self.value_head = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        features = self.shared_net(x)
        
        # Policy
        mean = self.policy_mean(features)
        std = torch.exp(self.policy_std).expand_as(mean)
        dist = Normal(mean, std)
        
        # Value
        value = self.value_head(features)
        
        return dist, value


class PPOAgent:
    def __init__(self, env, load_path=None, gamma=0.99, lr=3e-4, 
                 clip_epsilon=0.2, epochs=10, batch_size=64, 
                 hidden_dim=128, entropy_coef=0.01):
        self.env = env
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.epochs = epochs
        self.batch_size = batch_size
        self.entropy_coef = entropy_coef
        
        # Initialize networks
        obs_dim = env.obs().shape[0]
        act_dim = 1  # Assuming 1D continuous action
        
        self.policy = PPONetwork(obs_dim, act_dim, hidden_dim)
        self.old_policy = PPONetwork(obs_dim, act_dim, hidden_dim)
        
        if load_path:
            self.load_model(load_path)
        else:
            self.old_policy.load_state_dict(self.policy.state_dict())
        
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        # Initialize normalization parameters
        self.state_mean = torch.zeros(obs_dim)
        self.state_std = torch.ones(obs_dim)
        self.return_mean, self.return_std = 0.0, 1.0
        
        # Memory buffers
        self.buffer = {
            'states': [],
            'actions': [],
            'rewards': [],
            'next_states': [],
            'dones': [],
            'log_probs': []
        }
    
    def normalize_state(self, state):
        return (state - self.state_mean) / (self.state_std + 1e-8)
    
    def act(self, state, deterministic=False):
        state = self.normalize_state(torch.FloatTensor(state)).unsqueeze(0)
        with torch.no_grad():
            dist, _ = self.old_policy(state)
            if deterministic:
                action = dist.mean
            else:
                action = dist.sample()
            log_prob = dist.log_prob(action)
            
        return action.item(), log_prob.item()
    
    def store_transition(self, transition):
        for key, value in transition.items():
            self.buffer[key].append(value)
    
    def compute_returns_and_advantages(self):
        states = torch.FloatTensor(np.array(self.buffer['states']))
        rewards = torch.FloatTensor(self.buffer['rewards'])
        dones = torch.FloatTensor(self.buffer['dones'])
        
        # Compute normalized returns
        returns = []
        R = 0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                R = 0
            R = reward + self.gamma * R
            returns.insert(0, R)
        
        returns = torch.FloatTensor(returns)
        self.return_mean = returns.mean()
        self.return_std = returns.std()
        returns = (returns - self.return_mean) / (self.return_std + 1e-8)
        
        # Compute advantages
        with torch.no_grad():
            _, values = self.policy(states)
            values = values.squeeze()
            advantages = returns - values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
        return returns, advantages
    
    def update(self):
        # Update normalization parameters
        self.state_mean = torch.FloatTensor(np.array(self.buffer['states'])).mean(0)
        self.state_std = torch.FloatTensor(np.array(self.buffer['states'])).std(0)
        
        # Convert to tensors
        returns, advantages = self.compute_returns_and_advantages()
        states = self.normalize_state(torch.FloatTensor(np.array(self.buffer['states'])))
        actions = torch.FloatTensor(np.array(self.buffer['actions'])).unsqueeze(1)
        old_log_probs = torch.FloatTensor(np.array(self.buffer['log_probs'])).unsqueeze(1)
        
        # Optimize policy for K epochs
        for _ in range(self.epochs):
            indices = np.arange(len(states))
            np.random.shuffle(indices)
            
            for start in range(0, len(states), self.batch_size):
                end = start + self.batch_size
                batch_idx = indices[start:end]
                
                batch_states = states[batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_advantages = advantages[batch_idx]
                batch_returns = returns[batch_idx]
                
                # Get new policy
                dist, values = self.policy(batch_states)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()
                
                # Policy loss
                ratio = (new_log_probs - batch_old_log_probs).exp()
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1-self.clip_epsilon, 1+self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy
                
                # Value loss
                value_loss = F.mse_loss(values.squeeze(), batch_returns)
                
                # Total loss
                loss = policy_loss + 0.5 * value_loss
                
                # Backpropagate
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()
        
        # Clear buffer
        for key in self.buffer:
            self.buffer[key].clear()
        
        # Update old policy
        self.old_policy.load_state_dict(self.policy.state_dict())
    
    def save_model(self, path):
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'state_mean': self.state_mean,
            'state_std': self.state_std,
            'return_mean': self.return_mean,
            'return_std': self.return_std
        }, path)
    
    def load_model(self, path):
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.old_policy.load_state_dict(checkpoint['policy_state_dict'])
        self.state_mean = checkpoint['state_mean']
        self.state_std = checkpoint['state_std']
        self.return_mean = checkpoint['return_mean']
        self.return_std = checkpoint['return_std']

    def train_ppo(self, env, agent, save_dir='models', episodes=5000, max_steps=500, update_freq=2048):
        os.makedirs(save_dir, exist_ok=True)
        episode_rewards = []

        for episode in range(episodes):
            state = env.reset_model()
            episode_reward = 0
            episode_actions = []

            for step in range(max_steps):
                # Get action
                action, log_prob = agent.act(state)

                # Take step
                next_state, _, done = env.step([action])

                # Custom reward calculation
                angle = next_state[1] % (2*np.pi)
                if angle > np.pi:
                    angle = angle - 2*np.pi

                # Основная награда за положение
                position_reward = 1.0 - abs(angle)/np.pi

                # Штраф за большие действия и резкие движения
                action_penalty = 0.01 * (action**2)
                velocity_penalty = 0.005 * abs(next_state[3])  # qvel[1] - угловая скорость

                total_reward = position_reward - action_penalty - velocity_penalty

                # Store transition
                transition = {
                    'states': state,
                    'actions': action,
                    'rewards': total_reward,
                    'next_states': next_state,
                    'dones': done,
                    'log_probs': log_prob
                }
                agent.store_transition(transition)

                state = next_state
                episode_reward += total_reward
                episode_actions.append(action)

                # Render occasionally
                if episode % 100 == 0:
                    env.draw_ball([0, 0, 0.5], color=[0, 1, 0, 1], radius=0.05)
                    time.sleep(0.01)

                # Update if we have enough samples
                if len(agent.buffer['states']) >= update_freq:
                    agent.update()

                if done:
                    break

            episode_rewards.append(episode_reward)

            # Save model
            if episode % 500 == 0 or episode == episodes - 1:
                save_path = os.path.join(save_dir, f'ppo_model_{episode}.pth')
                agent.save_model(save_path)
                print(f"Model saved at {save_path}")

            # Print progress
            if episode % 50 == 0:
                avg_reward = np.mean(episode_rewards[-50:])
                avg_action = np.mean(np.abs(episode_actions))
                print(f"Episode {episode}")
                print(f"Avg Reward (last 50): {avg_reward:.2f}")
                print(f"Avg |Action|: {avg_action:.4f}")
                print(f"Current Angle: {angle:.2f} rad\n")

        return episode_rewards

# Initialize environment and agent
if __name__ =='__main__':
    env = InvertedPendulumEnv()
    agent = PPOAgent(env, lr=3e-4, hidden_dim=128)

    # Train the agent
    rewards = agent.train_ppo(env, agent, episodes=500)