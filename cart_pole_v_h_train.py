# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 20:08:12 2022

@author: krishna
"""
import gym
from stable_baselines3 import PPO,SAC
from stable_baselines3.common.env_checker import check_env
from cart_pole_v_h import IPHVC
env = IPHVC()

# check_env(env)

# env = IPHVC()
model = SAC("MlpPolicy", env, verbose=1, seed=0)
# cpcallback = CheckpointCallback(save_freq=25000, save_path=r"C:\Users\krish\Google Drive\IVHCP on HVC\logs", name_prefix='saciponhvc')
# model.learn(total_timesteps=int(5e6),callback=cpcallback)

# model.save(r"C:\Users\krish\Google Drive\IVHCP on HVC\sac_iponhvc")

# del model

# loaded_model = SAC.load(r"C:\Users\krish\Google Drive\IVHCP on HVC\sac_iponhvc", verbose=1)
# 
import os


models_dir = "VHICP/SAC"
logdir = 'logs'
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
    
if not os.path.exists(logdir):
    os.makedirs(logdir)

# env = gym.make('LunarLander-v2') 
env.reset()

model = SAC('MlpPolicy', env, verbose=1,tensorboard_log=logdir)

TIMESTEPS = 10000

for i in range(30):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="SAC")
    model.save(f"{models_dir}/{TIMESTEPS*i}")
# for i in range(40):
#     observation = env.reset()
#     # print("Episode finished after {} timesteps".format(i+1))
#     total_reward = 0
#     for t in range(100):
#         env.render()
#         #print(observation)
#         action = env.action_space.sample()
#         observation, reward , done, info = env.step(action)
#         total_reward += reward
#         if done:
#             # print("Episode finished after {} timesteps".format(t+1))
#             # print(f"Angle of pole at start of each step: {np.degrees(observation[1])}")
#             print(f"Total Reward :{total_reward}")
#             break
# env.close()   