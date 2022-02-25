import imp
import numpy as np
from envs import gym_compete
from envs.env_wrappers import SubprocVecEnv, DummyVecEnv
from envs.JSBSim.core.catalog import Catalog as c
import logging
import gym
from time import sleep

class GymEnv:
    def __init__(self, env):
        self.env = env
        self.num_agents = self.env.num_agents
        self.action_space = self.env.action_space[0] if isinstance(self.env.action_space, gym.spaces.Tuple) \
                                                        else self.env.action_space 
        self.observation_space = self.env.observation_space[0] if isinstance(self.env.observation_space, gym.spaces.Tuple) \
                                                        else self.env.observation_space 
        self.obs_shape = (self.num_agents, *self.observation_space.shape)
        self.act_shape = (self.num_agents, *self.action_space.shape)
        self.rew_shape = (self.num_agents, 1)
        self.done_shape = (self.num_agents, 1)

    def reset(self):
        observation = self.env.reset()
        return np.array(observation).reshape(self.obs_shape)

    def step(self, action):
        action = np.array(action).reshape(self.act_shape)
        observation, reward, done, info = self.env.step(action)
        observation = np.array(observation).reshape(self.obs_shape)
        done = np.array(done).reshape(self.done_shape)
        reward = np.array(reward).reshape(self.rew_shape)
        return observation, reward, done, info

    def render(self, mode="human"):
        self.env.render(mode)

    def close(self):
        self.env.close()
    
    def seed(self, v):
        self.env.seed(v)

parallel_num = 1
env = gym.make('run-to-goal-humans-v0')
env = GymEnv(env)
env.seed(0)
# env = DummyVecEnv([lambda : GymEnv(env) for _ in range(parallel_num)])

obs = env.reset()
env.render()
actions = [np.zeros((env.action_space.shape[0],)) for _ in range(env.num_agents)]
while True:
    obs, reward, done, info = env.step(actions)
    env.render()
    if np.all(done):
        break
