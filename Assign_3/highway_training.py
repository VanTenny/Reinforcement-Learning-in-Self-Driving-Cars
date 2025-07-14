# I used google colab for this since my PC is very old and almost died so here below are the commands to install the modules in colab
!pip install gymnasium
!pip install highway-env -q
!pip install stable-baselines3[extra] -q
!pip install moviepy -q # For saving video renderings

import gymnasium as gym
import highway_env
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
import os

config = {
    "action": {
        "type": "ContinuousAction"
    },
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 10, # Number of nearby vehicles to observe
        "features": ["presence", "x", "y", "vx", "vy"],
        "absolute": False,
    },
    "lanes_count": 3,
    "vehicles_count": 20, # Total vehicles in the simulation
    "reward_speed_range": [20, 30], # Reward for maintaining a speed between 20 and 30 m/s
    "simulation_frequency": 15, # Run the simulation at 15 Hz
    "policy_frequency": 5, # The agent makes a decision 5 times per second
    "collision_reward": -2, # A significant penalty for collisions[1]
}

env = make_vec_env("highway-v0", n_envs=1, env_kwargs={"config": config})

# --- PPO Model Definition ---
model = PPO(
    policy='MlpPolicy',
    env=env,
    learning_rate=5e-4,
    n_steps=1024,
    batch_size=64,
    n_epochs=10,
    gamma=0.9,
    gae_lambda=0.95,
    clip_range=0.2,
    verbose=1,
    tensorboard_log="./highway_ppo_tensorboard/"
)

callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=160, verbose=1)
eval_callback = EvalCallback(env,
                             best_model_save_path='./logs/',
                             log_path='./logs/',
                             eval_freq=5000,
                             callback_on_new_best=callback_on_best,
                             deterministic=True,
                             render=False)


model.learn(total_timesteps=100000, callback=eval_callback)
model.save("ppo_highway_final")

from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv

model = PPO.load("./logs/best_model")


video_folder = "Bideos/"
video_length = 400 # Recording for 400 steps

eval_env = DummyVecEnv([lambda: gym.make("highway-v0", render_mode="rgb_array", config=config)])
eval_env = VecVideoRecorder(eval_env, video_folder,
                            record_video_trigger=lambda x: x == 0,
                            video_length=video_length,
                            name_prefix="ppo-highway-agent")

obs = eval_env.reset()
for _ in range(video_length + 1):
    action, _ = model.predict(obs, deterministic=True)
    obs, _, _, _ = eval_env.step(action)

eval_env.close()

print("Evaluation finished. Video saved to the 'videos' folder.")

