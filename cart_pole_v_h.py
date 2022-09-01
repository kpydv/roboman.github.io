# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 20:04:21 2022

@author: krishna
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 14:27:25 2021

@author: krish
"""



import numpy as np
from numpy.lib import npyio
from scipy.integrate import solve_ivp
import gym
from gym import spaces
import matplotlib.pyplot as plt
import time

from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback


class IPHVC(gym.Env):
    metadata = {
        'render.modes': ['human','rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self):
        super(IPHVC,self).__init__()
        self.viewer = None
        self.counter = 0
        self.maxsteps = 1000
        self.mp = 0.25
        self.Jp = 0.0208
        self.lp = 1.0
        self.mc = 1
        self.g = 9.81
        self.fxscale = 10
        self.fyscale = 10
        self.xmax = 0.4
        self.ymax = 0.4
        self.state = np.array([0,0,0,0,0,0])
        self.action_space = spaces.Box(low=-1*np.ones((2,)),high=1*np.ones((2,)),dtype=np.float32)
        self.observation_space = spaces.Box(low=-100*np.ones((6,)),high=100*np.ones((6,)),dtype=np.float32)
    
    def reset(self):
        self.counter = 0
        self.state = np.array([np.pi+np.random.normal(0,1.2),0,0,np.random.normal(0,1.2),0,0])
        self.state[0] = self.state[0]%(2*np.pi)
        obs = self.state
        return obs

    def __f(self,t,y):
        mp, Jp, lp, mc, g = self.mp, self.Jp, self.lp, self.mc, self.g
        q, xc, yc, qd, xcd, ycd = y[0:6]
        M = np.array([[Jp + lp ** 2 * mp / 4,-mp * lp * np.cos(q) / 2,-mp * lp * np.sin(q) / 2],[-mp * lp * np.cos(q) / 2,mc + mp,0],[-mp * lp * np.sin(q) / 2,0,mc + mp]])
        C = np.array([[0],[mp * lp * qd ** 2 * np.sin(q) / 2],[-mp * lp * qd ** 2 * np.cos(q) / 2]])
        G = np.array([[-mp * g * lp * np.sin(q) / 2],[0],[mc * g + mp * g]])

        tau = self.tau.reshape((3,1))+np.array([0,0,(mp+mc)*g]).reshape((3,1))
        Minv = np.linalg.inv(M)
        acc = Minv@(tau-C-G)
        dy = np.array([y[3],y[4],y[5],acc[0],acc[1],acc[2]],dtype=np.float32)
        return dy

    def step(self,action):
        next_state, reward, done, info = self.state, 0, False, {'Terminal':''}
        self.counter += 1
        if self.counter>self.maxsteps:
            done, info['Terminal'] = True, 'Timeout'
            return next_state, reward, done, info
        self.tau = np.array([0,self.fxscale*action[0],self.fyscale*action[1]])
        self.state[0] = self.state[0]%(2*np.pi)
        sol = solve_ivp(self.__f,[0,0.01],self.state,rtol=1e-8,atol=1e-8)
        next_state = sol.y[:,-1]
        next_state[0] = next_state[0]%(2*np.pi)
        self.state = next_state
        reward = 0.1*abs(next_state[0]-np.pi)**2-0.005*abs(next_state[0]-np.pi)*next_state[3]**2

        if abs(next_state[1])>self.xmax or abs(next_state[2])>self.ymax:
            reward, done, info['Terminal'] = -0.1, True, 'Limit'
        if self.counter==self.maxsteps:
            done, info['Terminal'] = True, 'Timeout'
        
        print(reward)
        return next_state, reward, done, info

    def render(self, mode='human'):
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(400,400)
            self.viewer.set_bounds(-1.0,1.0,-1.0,1.0)#left,right,bottom,top

            gnd = rendering.make_polygon([(-0.1-self.xmax,-0.1-self.ymax),(0.1+self.xmax,-0.1-self.ymax),(0.1+self.xmax,0.1+self.ymax),(-0.1-self.xmax,0.1+self.ymax)],filled=False)
            gnd.set_color(0.5,0.5,0.5)

            cart = rendering.make_polygon([(-0.1,-0.1),(0.1,-0.1),(0.1,0.1),(-0.1,0.1)],filled=True)
            cart.set_color(1,0.5,0.5)
            self.cartt = rendering.Transform(translation=(0,0))
            cart.add_attr(self.cartt)

            pend = rendering.make_capsule(0.6,0.02)
            pend.set_color(0.5,0.5,1.0)
            self.pendt = rendering.Transform(rotation=self.state[0]+np.pi/2)
            pend.add_attr(self.pendt)

            self.viewer.add_geom(gnd)
            self.viewer.add_geom(cart)
            self.viewer.add_geom(pend)
        self.cartt.set_translation(self.state[1],self.state[2])
        self.pendt.set_translation(self.state[1],self.state[2])
        self.pendt.set_rotation(self.state[0]+np.pi/2)
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

# env = IPHVC()
# observation = env.reset()
# for _ in range(1000):
#   env.render()
#   action = env.action_space.sample() # your agent here (this takes random actions)
#   observation, reward, done, info = env.step(action)

#   if done:
#     observation = env.reset()
# env.close()
