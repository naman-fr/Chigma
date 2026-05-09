"""
RL Navigation Agent — PPO/SAC for Autonomous Drone Flight
============================================================
Reinforcement learning agent for obstacle avoidance and
target-seeking navigation using Stable-Baselines3.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

try:
    import gymnasium as gym
    from gymnasium import spaces
    HAS_GYM = True
except ImportError:
    HAS_GYM = False


class DroneNavEnv(gym.Env):
    """Custom Gymnasium environment for drone navigation.

    Observation space: [position(3), velocity(3), orientation(3),
                        obstacle_distances(8), target_direction(3)]
    Action space: [thrust, roll, pitch, yaw] (continuous)

    Reward design:
    - +100 for reaching target
    - -50 for collision
    - -0.1 per step (efficiency)
    - +1.0 proximity bonus (closer to target)
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__()

        self.config = config or {}
        reward_cfg = self.config.get("reward", {})

        # State dimensions
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(20,), dtype=np.float32
        )

        # Action: [thrust, roll, pitch, yaw] normalized [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32
        )

        # Reward parameters
        self.reward_target = reward_cfg.get("target_reached", 100.0)
        self.reward_collision = reward_cfg.get("collision", -50.0)
        self.reward_step = reward_cfg.get("step_penalty", -0.1)
        self.reward_proximity = reward_cfg.get("proximity_bonus", 1.0)

        # State
        self.position = np.zeros(3, dtype=np.float32)
        self.velocity = np.zeros(3, dtype=np.float32)
        self.target = np.array([50.0, 50.0, 30.0], dtype=np.float32)
        self.obstacles: list[np.ndarray] = []
        self.step_count = 0
        self.max_steps = 1000

    def reset(self, seed: int | None = None, **kwargs) -> tuple[np.ndarray, dict]:
        """Reset environment to initial state."""
        super().reset(seed=seed)

        self.position = np.zeros(3, dtype=np.float32)
        self.velocity = np.zeros(3, dtype=np.float32)
        self.target = self.np_random.uniform(-50, 50, size=3).astype(np.float32)
        self.target[2] = abs(self.target[2]) + 5  # Ensure positive altitude

        # Random obstacles
        n_obstacles = self.np_random.integers(5, 15)
        self.obstacles = [
            self.np_random.uniform(-40, 40, size=3).astype(np.float32)
            for _ in range(n_obstacles)
        ]

        self.step_count = 0
        return self._get_obs(), {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one environment step."""
        self.step_count += 1

        # Apply action as velocity change
        thrust = (action[0] + 1) / 2  # [0, 1]
        direction = action[1:4]
        self.velocity = self.velocity * 0.9 + direction * thrust * 2.0
        self.position += self.velocity * 0.1

        # Compute reward
        dist_to_target = np.linalg.norm(self.position - self.target)
        reward = self.reward_step

        # Proximity bonus
        reward += self.reward_proximity * max(0, 1.0 - dist_to_target / 100.0)

        # Check target reached
        terminated = False
        if dist_to_target < 2.0:
            reward += self.reward_target
            terminated = True

        # Check collisions
        for obs in self.obstacles:
            if np.linalg.norm(self.position - obs) < 3.0:
                reward += self.reward_collision
                terminated = True
                break

        # Check ground collision
        if self.position[2] < 0:
            reward += self.reward_collision
            terminated = True

        truncated = self.step_count >= self.max_steps

        return self._get_obs(), float(reward), terminated, truncated, {
            "distance_to_target": float(dist_to_target),
        }

    def _get_obs(self) -> np.ndarray:
        """Build observation vector."""
        # Obstacle distances (8 directions)
        obstacle_dists = np.full(8, 100.0, dtype=np.float32)
        for obs in self.obstacles:
            diff = obs - self.position
            dist = np.linalg.norm(diff)
            angle = np.arctan2(diff[1], diff[0])
            sector = int((angle + np.pi) / (2 * np.pi / 8)) % 8
            obstacle_dists[sector] = min(obstacle_dists[sector], dist)

        # Target direction (normalized)
        target_dir = self.target - self.position
        target_dist = np.linalg.norm(target_dir)
        target_dir_norm = target_dir / max(target_dist, 1e-6)

        orientation = np.zeros(3, dtype=np.float32)  # Simplified

        obs = np.concatenate([
            self.position, self.velocity, orientation,
            obstacle_dists, target_dir_norm,
        ]).astype(np.float32)

        return obs[:20]  # Ensure shape match


class RLNavigator:
    """RL-based drone navigation agent.

    Trains PPO/SAC policies for autonomous obstacle avoidance
    and target-seeking behavior.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}
        self.agent = None
        self.env = None

    def train(self, total_timesteps: int = 500000, save_path: str = "models/rl_agent") -> None:
        """Train the RL navigation agent."""
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv

        self.env = DummyVecEnv([lambda: DroneNavEnv(self.config)])

        algorithm = self.config.get("algorithm", "PPO")
        lr = self.config.get("learning_rate", 3e-4)

        self.agent = PPO(
            "MlpPolicy",
            self.env,
            learning_rate=lr,
            n_steps=self.config.get("n_steps", 2048),
            batch_size=self.config.get("batch_size", 64),
            n_epochs=self.config.get("n_epochs", 10),
            gamma=self.config.get("gamma", 0.99),
            verbose=1,
        )

        logger.info(f"Training {algorithm} for {total_timesteps} timesteps")
        self.agent.learn(total_timesteps=total_timesteps)

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        self.agent.save(save_path)
        logger.info(f"Agent saved: {save_path}")

    def load(self, path: str) -> None:
        """Load a trained agent."""
        from stable_baselines3 import PPO
        self.agent = PPO.load(path)
        logger.info(f"Agent loaded: {path}")

    def predict(self, observation: np.ndarray) -> np.ndarray:
        """Get action from the trained agent."""
        if self.agent is None:
            raise RuntimeError("Agent not loaded. Call train() or load() first.")
        action, _ = self.agent.predict(observation, deterministic=True)
        return action
