# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 17:57:18 2022

@author: krishna
"""

"""
We have these models saved, so we might as well let things continue to train as
long as the model is improving quickly. While we wait, we can start programming
the code required to load and actually run the model, so we can see it visually.
In a separate script, the main bit of code for loading a model is:
"""
import gym
from stable_baselines3 import SAC
from cart_pole_v_h import IPHVC
env = IPHVC()
models_dir = "VHICP/SAC"
logdir = 'logs'



model_path = f"{models_dir}/20000"
model = SAC.load(model_path, env=env)

episodes = 10

for ep in range(episodes):
    obs = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        env.render()
        # print(rewards)
        
        
env.close()