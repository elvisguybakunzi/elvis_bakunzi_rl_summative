import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.evaluation import evaluate_policy
from environment.rendering import OpenGLRenderer
from environment.custom_env import AccessLearnEnv
import pandas as pd
import time
import torch

class MetricsTracker:
    def __init__(self, model_type, model_path, num_episodes=100, eval_seeds=None):
        self.model_type = model_type.lower()
        self.model_path = model_path
        self.num_episodes = num_episodes
        self.eval_seeds = eval_seeds if eval_seeds else list(range(num_episodes))
        self.env = AccessLearnEnv()
        self.renderer = OpenGLRenderer()
        self.renderer.init_display()
        
        # Load model
        if self.model_type == "dqn":
            self.model = DQN.load(model_path)
        elif self.model_type == "ppo":
            self.model = PPO.load(model_path)
        else:
            raise ValueError("Model type must be 'dqn' or 'ppo'")

        # Metrics storage
        self.episode_rewards = []
        self.episode_lengths = []
        self.q_variances = []  # For DQN stability (Q-value variance)
        self.entropies = []  # For PPO stability (policy entropy)
        self.successes = 0  # For generalization (understanding >= 4)

    def evaluate_episode(self, seed, visualize=False):
        obs, _ = self.env.reset(seed=seed)
        total_reward = 0
        steps = 0
        done = False
        q_values_episode = []  # For DQN
        entropy_episode = []  # For PPO

        while not done:
            with torch.no_grad():
                action, _states = self.model.predict(obs, deterministic=True)
                if self.model_type == "dqn":
                    q_values = self.model.q_net(torch.tensor(obs[None], dtype=torch.float32))
                    q_values_episode.append(q_values.cpu().numpy().flatten())
                elif self.model_type == "ppo":
                    dist = self.model.policy.get_distribution(torch.tensor(obs[None], dtype=torch.float32))
                    entropy = dist.entropy().mean().item()
                    entropy_episode.append(entropy)

            obs, reward, terminated, truncated, _ = self.env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated

            if visualize:
                self.renderer.render_static_scene(self.env.state, action, reward, steps, self.env.max_steps, done)
                time.sleep(0.1)

        understanding = self.env.state[3]
        success = 1 if understanding >= 4 else 0
        q_variance = np.var(q_values_episode) if q_values_episode else 0
        entropy_mean = np.mean(entropy_episode) if entropy_episode else 0
        return total_reward, steps, q_variance, entropy_mean, success

    def run_evaluation(self, visualize_first_episode=False):
        print(f"Evaluating {self.model_type.upper()} model over {self.num_episodes} episodes")
        for i, seed in enumerate(self.eval_seeds):
            visualize = visualize_first_episode and i == 0
            total_reward, steps, q_variance, entropy, success = self.evaluate_episode(seed, visualize)
            self.episode_rewards.append(total_reward)
            self.episode_lengths.append(steps)
            if self.model_type == "dqn":
                self.q_variances.append(q_variance)
            elif self.model_type == "ppo":
                self.entropies.append(entropy)
            self.successes += success
            print(f"Episode {i+1}: Reward = {total_reward:.2f}, Steps = {steps}, Success = {success}")

        self.renderer.close()
        self.env.close()

    def smooth_data(self, data, window=5):
        """Apply moving average smoothing to data."""
        return pd.Series(data).rolling(window=window, min_periods=1).mean().to_numpy()

    def save_metrics(self, output_dir="metrics_analysis", dqn_data=None, ppo_data=None):
        os.makedirs(output_dir, exist_ok=True)
        prefix = f"{self.model_type}_"

        # Save individual model data
        if self.model_type == "dqn":
            stability_key = "Q Variance"
            stability_data = self.q_variances
        else:
            stability_key = "Entropy"
            stability_data = self.entropies

        df = pd.DataFrame({
            "Episode": range(1, self.num_episodes + 1),
            "Cumulative Reward": self.episode_rewards,
            "Episode Length": self.episode_lengths,
            stability_key: stability_data
        })
        df.to_csv(os.path.join(output_dir, f"{prefix}metrics.csv"), index=False)

        # Combined plots if both DQN and PPO data are provided
        if dqn_data is not None and ppo_data is not None:
            # Smooth the data for better visualization
            dqn_rewards_smooth = self.smooth_data(dqn_data["Cumulative Reward"])
            ppo_rewards_smooth = self.smooth_data(ppo_data["Cumulative Reward"])
            dqn_stability_smooth = self.smooth_data(dqn_data["Q Variance"])
            ppo_stability_smooth = self.smooth_data(ppo_data["Entropy"])

            # Cumulative Reward Plot
            plt.figure(figsize=(12, 6))
            plt.plot(dqn_data["Episode"], dqn_rewards_smooth, label="DQN Reward", color="blue")
            plt.plot(ppo_data["Episode"], ppo_rewards_smooth, label="PPO Reward", color="orange")
            plt.xlabel("Episode")
            plt.ylabel("Cumulative Reward")
            plt.title("Cumulative Reward Comparison: DQN vs PPO")
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, "cumulative_reward_comparison.png"))
            plt.close()

            # Stability Plot (Dual Axis)
            fig, ax1 = plt.subplots(figsize=(12, 6))
            ax1.plot(dqn_data["Episode"], dqn_stability_smooth, label="DQN Q-Variance", color="blue")
            ax1.set_xlabel("Episode")
            ax1.set_ylabel("Q-Variance (DQN)", color="blue")
            ax1.tick_params(axis="y", labelcolor="blue")

            ax2 = ax1.twinx()
            ax2.plot(ppo_data["Episode"], ppo_stability_smooth, label="PPO Policy Entropy", color="orange")
            ax2.set_ylabel("Entropy (PPO)", color="orange")
            ax2.tick_params(axis="y", labelcolor="orange")

            fig.suptitle("Stability Comparison: DQN Q-Variance vs PPO Entropy")
            fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, "stability_comparison.png"))
            plt.close()

    def print_summary(self):
        avg_reward = np.mean(self.episode_rewards)
        # Fix convergence calculation
        if np.std(self.episode_rewards) < 1.0:  # Further relaxed threshold
            convergence_episode = 1  # Converged immediately
        else:
            # Look for 20 consecutive episodes within 90% of the average reward
            convergence_episode = next(
                (i for i in range(len(self.episode_rewards) - 20) if all(
                    abs(r - avg_reward) <= 0.1 * avg_reward for r in self.episode_rewards[i:i+20]
                )),
                self.num_episodes
            )
        success_rate = self.successes / self.num_episodes * 100

        print(f"\n{self.model_type.upper()} Metrics Summary:")
        print(f"Average Cumulative Reward: {avg_reward:.2f}")
        print(f"Episodes to Convergence: {convergence_episode} (90% of avg reward sustained)")
        print(f"Generalization Success Rate: {success_rate:.2f}%")
        print(f"Average Episode Length: {np.mean(self.episode_lengths):.1f}")
        if self.model_type == "dqn":
            print(f"Average Q-Variance: {np.mean(self.q_variances):.4f}")
        else:
            print(f"Average Entropy: {np.mean(self.entropies):.4f}")

def main():
    # Paths to your trained models
    dqn_path = "models/dqn/best_model/dqn_model"
    ppo_path = "models/pg/best_model/ppo_model"
    # dqn_path = "models/dqn/final_model"
    # ppo_path = "models/pg/final_model"
    num_episodes = 1000  # Match your evaluation
    
    # Evaluate DQN
    dqn_tracker = MetricsTracker("dqn", dqn_path, num_episodes=num_episodes, eval_seeds=range(num_episodes))
    dqn_tracker.run_evaluation(visualize_first_episode=True)
    dqn_df = pd.DataFrame({
        "Episode": range(1, num_episodes + 1),
        "Cumulative Reward": dqn_tracker.episode_rewards,
        "Episode Length": dqn_tracker.episode_lengths,
        "Q Variance": dqn_tracker.q_variances
    })
    dqn_tracker.save_metrics(dqn_data=dqn_df)

    # Evaluate PPO
    ppo_tracker = MetricsTracker("ppo", ppo_path, num_episodes=num_episodes, eval_seeds=range(num_episodes))
    ppo_tracker.run_evaluation(visualize_first_episode=True)
    ppo_df = pd.DataFrame({
        "Episode": range(1, num_episodes + 1),
        "Cumulative Reward": ppo_tracker.episode_rewards,
        "Episode Length": ppo_tracker.episode_lengths,
        "Entropy": ppo_tracker.entropies
    })
    ppo_tracker.save_metrics(ppo_data=ppo_df, dqn_data=dqn_df)

    # Print summaries
    dqn_tracker.print_summary()
    ppo_tracker.print_summary()

if __name__ == "__main__":
    main()