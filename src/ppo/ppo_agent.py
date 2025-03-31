import numpy as np
import torch.optim as optim
import torch.nn.functional as F
import time
from .model import PPONetwork, torch
#from model import PPONetwork, nn, torch

class PPOAgent:
    def __init__(self, env, gamma=0.99, lr=3e-4, clip_epsilon=0.2, 
                 epochs=10, batch_size=64, hidden_dim=64, load_path=None):
        self.env = env
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.epochs = epochs
        self.batch_size = batch_size
        
        # Initialize networks
        obs_dim = env.obs().shape[0]
        act_dim = 1  # Assuming 1D continuous action
        
        self.policy = PPONetwork(obs_dim, act_dim, hidden_dim)
        self.old_policy = PPONetwork(obs_dim, act_dim, hidden_dim)
        if load_path is not None:
            self.load_model(load_path)
        else:
            self.old_policy.load_state_dict(self.policy.state_dict())
        
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        # Memory buffers
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.log_probs = []

    def load_model(self, path):
        """Загружает сохранённые веса модели"""
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint)
        self.old_policy.load_state_dict(checkpoint)
        print(f"Loaded model weights from {path}")

    def act(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            dist, _ = self.old_policy(state)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
        return action.item(), log_prob.item()
    
    def store_transition(self, state, action, reward, next_state, done, log_prob):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
        self.log_probs.append(log_prob)
    
    def compute_returns(self):
        returns = []
        R = 0
        for reward, done in zip(reversed(self.rewards), reversed(self.dones)):
            if done:
                R = 0
            R = reward + self.gamma * R
            returns.insert(0, R)
        returns = torch.FloatTensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        return returns
    
    def update(self):
        # Convert to tensors
        states = torch.FloatTensor(np.array(self.states))
        actions = torch.FloatTensor(np.array(self.actions)).unsqueeze(1)
        old_log_probs = torch.FloatTensor(np.array(self.log_probs)).unsqueeze(1)
        returns = self.compute_returns()
        
        # Calculate advantages
        with torch.no_grad():
            _, values = self.policy(states)
            advantages = returns - values.squeeze()
        
        # Optimize policy for K epochs
        for _ in range(self.epochs):
            # Shuffle the data
            indices = np.arange(len(self.states))
            np.random.shuffle(indices)
            
            for start in range(0, len(self.states), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]
                
                # Get batch
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]
                
                # Get new policy
                dist, values = self.policy(batch_states)
                new_log_probs = dist.log_prob(batch_actions)
                
                # Calculate ratio (pi_theta / pi_theta_old)
                ratio = (new_log_probs - batch_old_log_probs).exp()
                
                # Policy loss
                surr1 = ratio * batch_advantages.unsqueeze(1)
                surr2 = torch.clamp(ratio, 1-self.clip_epsilon, 1+self.clip_epsilon) * batch_advantages.unsqueeze(1)
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = F.mse_loss(values.squeeze(), batch_returns)
                
                # Total loss
                loss = policy_loss + 0.5 * value_loss
                
                # Backpropagate
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        
        # Clear memory
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.log_probs = []
        
        # Update old policy
        self.old_policy.load_state_dict(self.policy.state_dict())

    def train_ppo(self, env, agent, episodes=1000, max_steps=500, update_freq=2048, dynamic = False, freq = None, act_k=0, pos_k=0):
        episode_rewards = []
        if not dynamic:
            for episode in range(episodes):
                state = env.reset_model()
                episode_reward = 0
                
                for step in range(max_steps):
                    # Get action
                    action, log_prob = agent.act(state)
                    
                    # Take step
                    next_state, reward, done = env.step([action])
                    
                    # Custom reward for inverted pendulum
                    # Reward for being upright (angle close to 0)
                    angle = next_state[1]
                    #if angle > np.pi:
                    #    angle = angle - 2*np.pi
                    reward = np.cos(angle)  # Max reward when angle=0
                    
                    # Store transition
                    agent.store_transition(state, action, reward, next_state, done, log_prob)
                    
                    state = next_state
                    episode_reward += reward
                    
                    # Render occasionally
                    if episode % 10 == 0:
                        env.draw_ball([0, 0, 0.5], color=[0, 1, 0, 1], radius=0.05)
                        time.sleep(0.01)
                    
                    # Update if we have enough samples
                    if len(agent.states) >= update_freq:
                        agent.update()
                    
                    if done:
                        break
                        
                episode_rewards.append(episode_reward)
                
                # Print progress
                if episode % 10 == 0:
                    avg_reward = np.mean(episode_rewards[-10:])
                    print(f"Episode {episode}, Avg Reward: {avg_reward:.2f}")
                    if episode %500 ==0:
                        torch.save(self.policy.state_dict(), f"weights/ppo_{episode}.pth")
        else:
            act_k=1
            pos_k=1
            for episode in range(episodes):
                state = env.reset_model()
                episode_reward = 0
                target_update_freq = 100 if freq is None else freq  # Каждые 100 эпизодов меняем цель

                # Генерируем новую цель (шарик) каждые target_update_freq эпизодов
                if episode % target_update_freq == 0:
                    target_pos = [np.random.rand() - 0.5, 0, 0.6]

                # Визуализируем цель
                env.draw_ball(target_pos, color=[1, 0, 0, 1], radius=0.05)

                for step in range(max_steps):
                    # Получаем действие от агента
                    action, log_prob = agent.act(state)

                    # Выполняем шаг в среде
                    next_state, _, done = env.step([action])

                    # Вычисляем угол колонны
                    angle = next_state[1]

                    # Вычисляем расстояние до цели
                    cart_position = next_state[0]

                    distance_to_target = abs(cart_position - env.ball_position[0])

                    # Компоненты награды:
                    angle_reward = np.cos(angle)  # Награда за вертикальное положение (1 когда angle=0)
                    distance_reward = 10 * np.exp(-distance_to_target)  # Награда за приближение к цели
                    action_penalty = 0.5 * (next_state[2]**2)  # Штраф за большие скорости

                    #reward = 10*angle_reward + pos_k*distance_reward - 5*distance_to_target if angle_reward >0 else 10*angle_reward - act_k*action_penalty
                    reward = 10*angle_reward + pos_k*distance_reward - 5*distance_to_target - act_k*action_penalty

                    # Сохраняем переход
                    agent.store_transition(state, action, reward, next_state, done, log_prob)

                    state = next_state
                    episode_reward += reward

                    # Визуализация (реже, чтобы не замедлять обучение)
                    if episode % 50 == 0:
                        env.draw_ball(target_pos, color=[1, 0, 0, 1], radius=0.05)
                        time.sleep(0.01)

                    # Обновляем политику, если накопили достаточно данных
                    if len(agent.states) >= update_freq:
                        agent.update()

                    if done:
                        break

                episode_rewards.append(episode_reward)

                # Логирование и сохранение
                if episode % 10 == 0:
                    avg_reward = np.mean(episode_rewards[-10:])
                    print(f"Episode {episode}, Avg Reward: {avg_reward:.2f},angle_r: {10*angle_reward:.2f}, dist_r: {pos_k*distance_reward:.2f}, {env.ball_position[0]:.2f}")

                if episode % 500 == 0 or avg_reward > 5000:
                    torch.save(agent.policy.state_dict(), f"dynamic_weights/ppo_{episode}_{avg_reward}.pth")
        return episode_rewards

# Initialize environment and agent
if __name__ =='__main__':
    from examples.dynamic_test import InvertedPendulumEnv
    env = InvertedPendulumEnv()
    agent = PPOAgent(env, lr=3e-4, hidden_dim=128)

    # Train the agent
    rewards = train_ppo(env, agent, episodes=500)