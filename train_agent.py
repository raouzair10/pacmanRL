import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import os
import ale_py

# Register ALE environments
gym.register_envs(ale_py)


# Callback to print episode progress
class EpisodeProgressCallback(BaseCallback):
    
    def __init__(self, verbose=0):
        super(EpisodeProgressCallback, self).__init__(verbose)
        self.episode_count = 0
        self.last_printed_episode = 0
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_reward = 0
        self.current_episode_length = 0
    
    def _on_step(self) -> bool:
        # Track current episode reward and length
        reward = self.locals.get('rewards', [0])[0]
        self.current_episode_reward += reward
        self.current_episode_length += 1
        
        # Check if episode ended
        if self.locals.get('dones', [False])[0]:  # Episode ended
            self.episode_count += 1
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            
            # Reset for next episode
            self.current_episode_reward = 0
            self.current_episode_length = 0
            
            # Print progress every 10 episodes
            if self.episode_count >= self.last_printed_episode + 10:
                # Calculate statistics for last 10 episodes
                recent_rewards = self.episode_rewards[-10:]
                recent_lengths = self.episode_lengths[-10:]
                
                avg_reward = np.mean(recent_rewards)
                avg_length = np.mean(recent_lengths)
                best_reward = max(recent_rewards)
                worst_reward = min(recent_rewards)
                
                # Get current loss if available
                current_loss = "N/A"
                if hasattr(self.model, 'logger') and self.model.logger.name_to_value:
                    current_loss = f"{self.model.logger.name_to_value.get('train/loss', 'N/A'):.4f}"
                
                print(f"Episode {self.episode_count}: Avg Reward={avg_reward:.2f} | "
                      f"Avg Length={avg_length:.1f} | Best={best_reward:.2f} | "
                      f"Worst={worst_reward:.2f} | Loss={current_loss}")
                
                self.last_printed_episode = self.episode_count
        
        return True


def create_pacman_env():
    # Create base environment
    env = gym.make("ALE/Pacman-v5", frameskip=1)
    
    # Apply Atari wrapper with proper settings
    env = AtariWrapper(
        env,
        frame_skip=4,
        terminal_on_life_loss=False,  # Don't end episode on life loss
        clip_reward=True
    )
    
    return env

def train(model_path="ppo_pacman.zip"):
    print("Starting PPO training on ALE Pacman...")
    
    # Check GPU availability
    import torch
    if torch.cuda.is_available():
        print(f"GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("No GPU detected, using CPU")
    
    # Create environment
    env = create_pacman_env()
    
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    # PPO model
    model = PPO(
        "CnnPolicy",
        env,
        n_steps=128,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.1,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        learning_rate=2.5e-4,
        verbose=0,
    )
    
    # Create callback for episode progress tracking
    episode_callback = EpisodeProgressCallback()
    
    # Train the model
    print("Starting training for 10M timesteps...")
    model.learn(total_timesteps=10_000_000, callback=episode_callback)
    
    model.save(model_path)
    print(f"Model saved as '{model_path}'")
    
    env.close()
    return model

def evaluate_model(model_path="ppo_pacman.zip", episodes=100):
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found!")
        return None
    
    env = create_pacman_env()
    
    model = PPO.load(model_path)
    print(f"Model loaded from '{model_path}'")
    
    episode_rewards = []
    episode_lengths = []
    
    print(f"Evaluating over {episodes} episodes...")
    
    for episode in range(episodes):
        obs, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            
            episode_reward += reward
            episode_length += 1
            
            if terminated or truncated:
                break
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        # Print progress every 10 episodes
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards)
            print(f"Episode {episode + 1}/{episodes}: Avg Reward = {avg_reward:.2f}")
    
    env.close()
    
    # Print final statistics
    print(f"\nEvaluation Results:")
    print(f"Mean Reward: {np.mean(episode_rewards):.2f}")
    print(f"Mean Episode Length: {np.mean(episode_lengths):.1f}")
    print(f"Best Episode: {max(episode_rewards):.2f}")
    print(f"Worst Episode: {min(episode_rewards):.2f}")
    
    return {
        'rewards': episode_rewards,
        'lengths': episode_lengths,
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards)
    }

if __name__ == "__main__":
    # Train the model
    model = train()
    
    # Evaluate the model
    print("\n" + "="*50)
    print("EVALUATING TRAINED MODEL")
    print("="*50)
    evaluate_model()
