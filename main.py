import os
import sys
import argparse
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import matplotlib.pyplot as plt
import pygame
import time

# Add paths for custom modules
sys.path.append("environment")
sys.path.append("training")

# Import custom modules
from custom_env import AccessLearnEnv
from rendering import OpenGLRenderer, create_static_visualization
import dqn_training
import pg_training

def visualize_environment():
    """Create a static visualization of the environment"""
    env = AccessLearnEnv()
    create_static_visualization(env)

def train_models():
    """Train both DQN and PPO models"""
    print("Training DQN model...")
    dqn_model = dqn_training.train_dqn()
    
    print("\nTraining PPO model...")
    ppo_model = pg_training.train_ppo()
    
    return dqn_model, ppo_model

def record_videos(dqn_path=None, ppo_path=None):
    """Record videos of trained agents"""
    video_dir = "videos"
    os.makedirs(video_dir, exist_ok=True)
    
    if dqn_path:
        from stable_baselines3 import DQN
        dqn_model = DQN.load(dqn_path)
    else:
        dqn_model = None
    
    if ppo_path:
        from stable_baselines3 import PPO
        ppo_model = PPO.load(ppo_path)
    else:
        ppo_model = None
    
    if dqn_model:
        env = AccessLearnEnv(render_mode="rgb_array")
        env = RecordVideo(
            env, 
            video_folder=f"{video_dir}/dqn",
            episode_trigger=lambda episode_id: True
        )
        record_agent(env, dqn_model, "DQN", num_episodes=3)
    
    if ppo_model:
        env = AccessLearnEnv(render_mode="rgb_array")
        env = RecordVideo(
            env, 
            video_folder=f"{video_dir}/ppo",
            episode_trigger=lambda episode_id: True
        )
        record_agent(env, ppo_model, "PPO", num_episodes=3)

def record_agent(env, model, model_name, num_episodes=3):
    """Record an agent's performance for a specified number of episodes"""
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated
        
        print(f"{model_name} Episode {episode+1}: Reward = {total_reward}, Steps = {steps}")
    
    env.close()

def play_agent(model_type="dqn", model_path=None, num_episodes=5, step_delay=0.1):
    if model_type.lower() == "dqn":
        from stable_baselines3 import DQN
        default_path = "models/dqn/best_model/dqn_model" if os.path.exists("models/dqn/best_model/dqn_model.zip") else "models/dqn/final_model"
        model = DQN.load(model_path or default_path)
    elif model_type.lower() == "ppo":
        from stable_baselines3 import PPO
        default_path = "models/pg/best_model/ppo_model" if os.path.exists("models/pg/best_model/ppo_model.zip") else "models/pg/final_model"
        model = PPO.load(model_path or default_path)
    else:
        raise ValueError("Model type must be 'dqn' or 'ppo'")

    env = AccessLearnEnv()
    renderer = OpenGLRenderer()
    renderer.init_display()

    print(f"Playing {model_type.upper()} agent in 3D environment for {num_episodes} episodes")
    print("Press ESC to exit, SPACE to reset episode early")

    running = True
    episode = 0

    while running and episode < num_episodes:
        obs, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0
        episode += 1
        print(f"\nStarting Episode {episode}/{num_episodes}")

        while running and not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated

            renderer.render_static_scene(env.state, action, reward, steps, env.max_steps, done)
            time.sleep(step_delay)

            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    done = True
                    print("Manual reset triggered")

            if steps >= 1000:
                done = True
                print("Episode capped at 1000 steps")

        print(f"{model_type.upper()} Episode {episode}: Total Reward = {total_reward:.2f}, Steps = {steps}")

    renderer.close()
    env.close()

def compare_models():
    """Compare the performance of the two models"""
    from stable_baselines3 import DQN, PPO
    
    try:
        dqn_model = DQN.load("models/dqn/best_model/dqn_model")
    except:
        dqn_model = DQN.load("models/dqn/final_model")
    
    try:
        ppo_model = PPO.load("models/pg/best_model/ppo_model")
    except:
        ppo_model = PPO.load("models/pg/final_model")
    
    env = AccessLearnEnv()
    
    num_episodes = 100
    dqn_rewards = []
    ppo_rewards = []
    dqn_steps = []
    ppo_steps = []
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        while not done:
            action, _ = dqn_model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated
        
        dqn_rewards.append(total_reward)
        dqn_steps.append(steps)
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        while not done:
            action, _ = ppo_model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated
        
        ppo_rewards.append(total_reward)
        ppo_steps.append(steps)
    
    dqn_mean_reward = np.mean(dqn_rewards)
    dqn_std_reward = np.std(dqn_rewards)
    dqn_mean_steps = np.mean(dqn_steps)
    
    ppo_mean_reward = np.mean(ppo_rewards)
    ppo_std_reward = np.std(ppo_rewards)
    ppo_mean_steps = np.mean(ppo_steps)
    
    print("\nModel Comparison:")
    print(f"DQN: Mean Reward = {dqn_mean_reward:.2f} ± {dqn_std_reward:.2f}, Mean Steps = {dqn_mean_steps:.2f}")
    print(f"PPO: Mean Reward = {ppo_mean_reward:.2f} ± {ppo_std_reward:.2f}, Mean Steps = {ppo_mean_steps:.2f}")
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.bar(['DQN', 'PPO'], [dqn_mean_reward, ppo_mean_reward], yerr=[dqn_std_reward, ppo_std_reward])
    plt.title('Average Reward')
    plt.ylabel('Reward')
    
    plt.subplot(1, 2, 2)
    plt.bar(['DQN', 'PPO'], [dqn_mean_steps, ppo_mean_steps])
    plt.title('Average Steps per Episode')
    plt.ylabel('Steps')
    
    plt.tight_layout()
    plt.savefig("comparison_results.png")
    plt.close()
    
    return {
        'dqn_mean_reward': dqn_mean_reward,
        'dqn_std_reward': dqn_std_reward,
        'dqn_mean_steps': dqn_mean_steps,
        'ppo_mean_reward': ppo_mean_reward,
        'ppo_std_reward': ppo_std_reward,
        'ppo_mean_steps': ppo_mean_steps
    }

def main():
    parser = argparse.ArgumentParser(description='AccessLearn Navigator - RL Project')
    parser.add_argument('--visualize', action='store_true', help='Run static visualization')
    parser.add_argument('--train', action='store_true', help='Train models')
    parser.add_argument('--record', action='store_true', help='Record videos')
    parser.add_argument('--compare', action='store_true', help='Compare models')
    parser.add_argument('--play', choices=['dqn', 'ppo'], help='Play agent in 3D environment (dqn or ppo)')
    parser.add_argument('--model_path', type=str, help='Path to specific model file')
    parser.add_argument('--all', action='store_true', help='Run all steps except play')
    
    args = parser.parse_args()
    
    if args.all:
        args.visualize = args.train = args.record = args.compare = True
    
    if args.visualize:
        print("Creating static visualization...")
        visualize_environment()
    
    if args.train:
        print("Training models...")
        train_models()
    
    if args.record:
        print("Recording videos...")
        dqn_path = "models/dqn/best_model/dqn_model"
        if not os.path.exists(dqn_path + ".zip"):
            dqn_path = "models/dqn/final_model"
        
        ppo_path = "models/pg/best_model/ppo_model"
        if not os.path.exists(ppo_path + ".zip"):
            ppo_path = "models/pg/final_model"
        
        record_videos(dqn_path, ppo_path)
    
    if args.compare:
        print("Comparing models...")
        compare_models()
    
    if args.play:
        print(f"Playing {args.play.upper()} agent...")
        play_agent(args.play, args.model_path)
    
    if not any([args.visualize, args.train, args.record, args.compare, args.play]):
        parser.print_help()

if __name__ == "__main__":
    main()