#!/usr/bin/env python
import imp
import sys
import os
import torch
import random
import logging
import numpy as np
import gym
from pathlib import Path
import setproctitle
# Deal with import error
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
from config import get_config
from envs import gym_compete
from runner.gym_runner import GymRunner
from envs.env_wrappers import SubprocVecEnv, DummyVecEnv, ShareSubprocVecEnv, ShareDummyVecEnv

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

def make_render_env(all_args):
    env = gym.make(all_args.scenario_name)
    # env.seed(all_args.seed + rank * 1000)
    return GymEnv(env)

def parse_args(args, parser):
    group = parser.add_argument_group("JSBSim Env parameters")
    group.add_argument('--episode-length', type=int, default=1000,
                       help="the max length of an episode")
    group.add_argument('--scenario-name', type=str, default='singlecombat_vsbaseline',
                       help="number of fighters controlled by RL policy")
    group.add_argument('--num-agents', type=int, default=1,
                       help="number of fighters controlled by RL policy")
    all_args = parser.parse_known_args(args)[0]
    return all_args


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)
    assert all_args.model_dir is not None
    # seed
    np.random.seed(all_args.seed)
    random.seed(all_args.seed)
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)

    # cuda
    if all_args.cuda and torch.cuda.is_available():
        logging.info("choose to use gpu...")
        device = torch.device("cuda:0")  # use cude mask to control using which GPU
        torch.set_num_threads(all_args.n_training_threads)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
    else:
        logging.info("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    # run dir
    run_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/results") \
        / all_args.env_name / all_args.scenario_name / all_args.algorithm_name / all_args.experiment_name
    if not run_dir.exists():
        os.makedirs(str(run_dir))
    curr_run = 'render'
    run_dir = run_dir / curr_run
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    setproctitle.setproctitle(str(all_args.algorithm_name) + "-" + str(all_args.env_name)
                              + "-" + str(all_args.experiment_name) + "@" + str(all_args.user_name))

    # env init
    envs = make_render_env(all_args)
    num_agents = all_args.num_agents

    config = {
        "all_args": all_args,
        "eval_envs": None,
        "envs": envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir
    }

    # run experiments
    runner = GymRunner(config)
    runner.render()

    # post process
    envs.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="%(message)s")
    main(sys.argv[1:])
