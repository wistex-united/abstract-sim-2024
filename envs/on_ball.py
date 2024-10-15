'''
Description: Full suite of method
'''

import copy
import functools
import time
from pettingzoo import ParallelEnv
import gymnasium as gym
import pygame
import math
import numpy as np
import random
import sys
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
import rich
import torch
from envs.base import BaseEnv

from utils.env import ( 
    DotDict,
    History, 
    dist, 
    get_rel_obs, 
    can_kick,
    is_facing,
    is_goal,
    is_out_of_bounds,
    normalize_angle
)

class Env(BaseEnv):
    def __init__(self):
        super().__init__()

        self.rew_map = DotDict({
            "ball_to_goal": 0.3,
            "goal": 10000,
            "out_of_bounds": 0, # -1000,
            "step": -1
        })

        self.cfg = DotDict({
            'episode_length': 1000,
            'action_radius': 500,
            'ball': {
                'radius': 10,
                'acl_coef': -0.8,
                'vel_coef': 4,
                'angle_noise': 0,
                'dist_noise': 0
            },
            'agent': {
                'size_x': 300,
                'size_y': 400,
                'disp_coef': 12,
                'angle_disp_coef': 0.02,
                'kick_coef': 60,
            },
            'target': {
                # controls how fast target roates around the ball
                'angle_disp': .05
            },
            'opponent_radius': 200,
            'teammate_radius': 200
        })

        self.possible_agents = [0]
        self.agents = [0]

        self.reset()

        action_space = gym.spaces.Box(low=-1, high=1, shape=(2,))
        self.action_spaces = { agent: action_space for agent in self.agents }

        obs_size = len(self.observation(0))
        observation_space = gym.spaces.Box(low=-1, high=1, shape=(obs_size,))
        self.observation_spaces = { agent: observation_space for agent in self.agents }

    def reset(self, return_info=False, options=None, **kwargs):
        self.state = DotDict({
            'step': 0,
            'target': None,
            # [x, y, angle, vel]
            'ball': [
                np.random.uniform(-4500, 4500),
                np.random.uniform(-3000, 3000),
                0,
                0
            ],
            # [x, y, angle]
            'agents': [
                [
                    np.random.uniform(-4500, 4500),
                    np.random.uniform(-3000, 3000),
                    0
                ]
            ]
        })

        x = self.state.ball[0] + self.cfg.action_radius * np.cos(0)
        y = self.state.ball[1] + self.cfg.action_radius * np.sin(0)
        angle = np.arctan2(self.state.ball[1] - y, self.state.ball[0] - x)
        self.state.target = [x, y, angle, 0]

        obs, info = {}, {}
        obs[0] = self.observation(0)

        return obs, info

    def observation(self, agent_idx):
        target = self.state.target
        agent = self.state.agents[agent_idx]

        obs = [
            1 if can_kick(agent, self.state.ball, tol=600) else 0,
            *get_rel_obs(target, [4500, 0]),
            *get_rel_obs(target, agent),
            *get_rel_obs(target, [4500, 250]),
            *get_rel_obs(target, [4500, -250]),
            *get_rel_obs(target, [-4500, 250]),
            *get_rel_obs(target, [-4500, -250]),
            *get_rel_obs(target, [0, 3000]),
            *get_rel_obs(target, [0, -3000])
        ]

        return obs


    def reward(self, agent_idx, action):
        if is_goal(self.state.ball):
            return self.rew_map.goal

        if is_out_of_bounds(self.state.ball, self.cfg.ball.radius * 2):
            return self.rew_map.out_of_bounds
        
        reward = 0

        reward += self.rew_map.ball_to_goal * max(0, (
            dist(self.state.prev_ball, [4500, 0]) -
            dist(self.state.ball, [4500, 0])
        ))

        reward += self.rew_map.step

        return reward

    def terminate(self):
        return is_goal(self.state.ball) or is_out_of_bounds(self.state.ball, tol=self.cfg.ball.radius * 2)

    def truncate(self):
        return self.state.step > self.cfg.episode_length

    def step(self, joint_action):
        obs, rew, term, trun, info = {}, {}, {}, {}, {}
        self.state.step += 1

        self.state.prev_ball = copy.deepcopy(self.state.ball)

        self.transition(joint_action)

        obs[0] = self.observation(0)
        rew[0] = self.reward(0, joint_action[0])
        term[0] = self.terminate()
        trun[0] = self.truncate()
        info[0] = {}

        return obs, rew, term, trun, info

    def transition(self, joint_action):
        self.update_target(joint_action[0])
        self.update_ball()
        self.update_agent(self.state.agents[0], joint_action[0])

    def update_target(self, action):
        new_angle = self.state.target[3] + action[0] * self.cfg.target.angle_disp
        new_angle = normalize_angle(new_angle)

        x = self.state.ball[0] + self.cfg.action_radius * np.cos(new_angle)
        y = self.state.ball[1] + self.cfg.action_radius * np.sin(new_angle)
        angle = np.arctan2(self.state.ball[1] - y, self.state.ball[0] - x)

        self.state.target = [x, y, angle, new_angle]

    def update_ball(self):
        ball = self.state.ball
        ball[3] += self.cfg.ball.acl_coef
        ball[3] = np.clip(ball[3], 0, 100)
        ball[0] += ball[3] * math.cos(ball[2])
        ball[1] += ball[3] * math.sin(ball[2])

    def update_agent(self, agent, action):
        ball = self.state.ball
        target = self.state.target

        agent_to_target = dist(agent, target)

        # if close enough, set to target exactly and kick the ball
        if agent_to_target < self.cfg.agent.disp_coef * 2:
            agent[0] = target[0]
            agent[1] = target[1]
            agent[2] = target[2]

            ball[2] = agent[2]
            ball[3] = self.cfg.agent.kick_coef * 1 + 20
            
        else:
            dir_x = target[0] - agent[0]
            dir_y = target[1] - agent[1]
            
            magnitude = math.sqrt(dir_x**2 + dir_y**2)
            dir_x /= magnitude
            dir_y /= magnitude

            agent[0] += dir_x * self.cfg.agent.disp_coef
            agent[1] += dir_y * self.cfg.agent.disp_coef

            if agent_to_target > 1000:
                agent[2] = np.arctan2(dir_y, dir_x)
            else:
                # if agent is close to target, gradually rotate agent to target
                angle_delta = target[2] - agent[2]
                angle_delta = normalize_angle(angle_delta)
                rotation_step = np.clip(angle_delta, -self.cfg.agent.angle_disp_coef, self.cfg.agent.angle_disp_coef)
                agent[2] += rotation_step
                agent[2] = normalize_angle(agent[2])

        # make sure robot is on field
        agent[0] = np.clip(agent[0], -5200, 5200)
        agent[1] = np.clip(agent[1], -3700, 3700)