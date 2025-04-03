import os
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt

# Import our custom environment
import sys
sys.path.append("../environment")
from custom_env import AccessLearnEnv

class SaveOnBestTrainingRewardCallback(BaseCallback):
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf
        self.successes = 0
        self.total_episodes = 0

    def _init_callback(self) -> None:
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        # Track success rate
        if self.locals["dones"][0]:  # Episode ended
            self.total_episodes += 1
            info = self.locals["infos"][0]
            if info.get("understanding", 0) >= 4:
                self.successes += 1

        if self.n_calls % self.check_freq == 0:
            x, y = ts2xy(load_results(self.log_dir), 'timesteps')
            if len(x) > 0:
                mean_reward = np.mean(y[-100:])
                success_rate = self.successes / self.total_episodes * 100 if self.total_episodes > 0 else 0
                if self.verbose > 0:
                    print(f"Num timesteps: {self.num_timesteps}")
                    print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward: {mean_reward:.2f}")
                    print(f"Success rate: {success_rate:.2f}% (Successes: {self.successes}/{self.total_episodes})")

                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    if self.verbose > 0:
                        print(f"Saving new best model to {self.save_path}")
                    self.model.save(f"{self.save_path}/ppo_model")
        return True

def train_ppo():
    log_dir = "../models/pg/"
    os.makedirs(log_dir, exist_ok=True)
    
    env = AccessLearnEnv()
    env = Monitor(env, log_dir)
    
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.1,  # Reduced for more conservative updates
        ent_coef=0.05,  # Added to balance exploration
        verbose=1
    )
    
    callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)
    
    model.learn(total_timesteps=200000, callback=callback)  # Increased timesteps
    
    model.save(f"{log_dir}/final_model")
    
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    
    plot_results(log_dir, "PPO Training")
    
    return model

def plot_results(log_dir, title):
    x, y = ts2xy(load_results(log_dir), 'timesteps')
    plt.figure(figsize=(10, 5))
    plt.plot(x, y)
    plt.xlabel('Timesteps')
    plt.ylabel('Rewards')
    plt.title(f'{title} Learning Curve')
    plt.grid(True)
    plt.savefig(f"{log_dir}/learning_curve.png")
    plt.close()

if __name__ == "__main__":
    model = train_ppo()