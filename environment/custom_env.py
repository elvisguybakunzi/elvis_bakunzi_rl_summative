import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces

class AccessLearnEnv(gym.Env):
    def __init__(self, render_mode=None):
        super(AccessLearnEnv, self).__init__()
        self.render_mode = render_mode

        # Action space: 9 discrete actions
        self.action_space = spaces.Discrete(9)

        # Observation space: 7 dimensions
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
            high=np.array([8, 3, 127, 5, 10, 255, 10], dtype=np.float32),
            dtype=np.float32
        )

        # Content types dictionary
        self.content_types = {
            0: "Text",
            1: "Simple Diagram",
            2: "Complex Diagram",
            3: "Chart/Graph",
            4: "Video",
            5: "Interactive Simulation",
            6: "Math Equation",
            7: "Map",
            8: "3D Model"
        }

        # State variables
        self.state = None
        self.max_steps = 20
        self.current_step = 0
        self.np_random = None
        self.last_action = None  # Track the last action for repetition penalty

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed(seed)

        # Initialize state
        content_type = self.np_random.integers(0, 9)
        impairment_level = self.np_random.integers(0, 4)
        devices = self.np_random.integers(0, 128)
        understanding = self.np_random.uniform(0, 2)
        time_remaining = 10.0
        previous_actions = 0
        energy = 10.0

        self.state = np.array([
            content_type,
            impairment_level,
            devices,
            understanding,
            time_remaining,
            previous_actions,
            energy
        ], dtype=np.float32)

        self.current_step = 0
        self.last_action = None  # Reset last action
        return self.state, {}

    def step(self, action):
        assert self.action_space.contains(action), f"Invalid action: {action}"

        # Unpack state
        content_type, impairment, devices, understanding, time_remaining, previous_actions, energy = self.state

        # Update state
        self.current_step += 1
        time_remaining = max(0, time_remaining - 0.5)
        energy = max(0, energy - 0.3)

        # Update previous actions
        prev_actions_int = int(previous_actions)
        previous_actions = float(((prev_actions_int << 3) & 0xFF) | (action & 0x7))

        # Calculate reward
        reward = 0
        if self._is_appropriate_action(action, content_type, impairment, devices):
            reward += 2  # Appropriate action
            if action in [0, 1, 3, 4, 5]:
                understanding = min(5, understanding + self.np_random.uniform(0.5, 1.5))
            if understanding >= 4:
                reward += 10  # Increased success bonus
        else:
            reward -= 2  # Inappropriate action penalty

        # Penalty for low understanding
        if understanding < 4:
            reward -= 0.2

        # Penalty for repeating the same action
        if self.last_action is not None and action == self.last_action:
            reward -= 0.2
        self.last_action = action

        if action == 7:  # Human assistance
            reward -= 1

        if energy > 7:
            reward += 0.5  # Reduced energy bonus

        if time_remaining <= 0 or energy <= 0:
            reward -= 2

        # Update state
        self.state = np.array([
            content_type,
            impairment,
            devices,
            understanding,
            time_remaining,
            previous_actions,
            energy
        ], dtype=np.float32)

        # Termination conditions
        terminated = (understanding >= 4) or (time_remaining <= 0) or (energy <= 0)
        truncated = self.current_step >= self.max_steps

        # Render if needed
        if self.render_mode == "human":
            self.render()

        return self.state, reward, terminated, truncated, {}

    def _is_appropriate_action(self, action, content_type, impairment, devices):
        devices_int = int(devices)
        has_screen_reader = devices_int & 0x1
        has_braille = devices_int & 0x2
        has_audio = devices_int & 0x4
        has_tactile = devices_int & 0x8

        if content_type == 0:  # Text
            if action in [0, 4] and has_audio:
                return True
            elif action == 1:
                return False
        elif content_type in [1, 2]:  # Diagrams
            if action in [1, 2] and has_audio:
                return True
            elif action == 0 and not has_audio:
                return False
        elif content_type == 3:  # Chart/Graph
            if action in [1, 2] and has_audio:
                return True
            elif action == 3 and not has_tactile:
                return False
        elif content_type == 4:  # Video
            if action in [1, 4] and has_audio:
                return True
            elif action == 2:
                return False
        elif content_type == 5:  # Interactive Simulation
            if action in [3, 5] and has_tactile:
                return True
            elif action == 0:
                return False
        elif content_type == 6:  # Math Equation
            if action in [0, 4] and has_audio:
                return True
            elif action == 1:
                return False
        elif content_type == 7:  # Map
            if action in [1, 3] and (has_audio or has_tactile):
                return True
            elif action == 2:
                return False
        elif content_type == 8:  # 3D Model
            if action in [3, 5] and has_tactile:
                return True
            elif action == 0:
                return False

        if action == 6:  # Adjust pace
            return True
        if action == 7:  # Human assistance
            return True
        if action == 8:  # Comprehension check
            return True
        return False

    def render(self):
        if self.render_mode == "human":
            pass
        elif self.render_mode == "rgb_array":
            return np.zeros((400, 600, 3), dtype=np.uint8)

    def close(self):
        pygame.quit()

    def get_state_description(self):
        content_type, impairment, devices, understanding, time_remaining, previous_actions, energy = self.state
        devices_list = [name for i, name in enumerate(["Screen Reader", "Braille", "Audio", "Tactile", 
                                                       "High Contrast", "Zoom", "Voice"])
                        if int(devices) & (1 << i)]
        return {
            "Content Type": self.content_types[int(content_type)],
            "Impairment Level": ["Mild", "Moderate", "Severe", "Complete"][int(impairment)],
            "Devices": devices_list if devices_list else "None",
            "Understanding": understanding,
            "Time Remaining": time_remaining,
            "Previous Actions": bin(int(previous_actions))[2:].zfill(8),
            "Energy": energy
        }