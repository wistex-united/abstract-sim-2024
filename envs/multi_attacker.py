'''
Version of 2v0 attacker training from Adam's research.

Command to replicate results: 
Note you can change num_cpus and vec_envs if you don't have a lot of cores. It simply speeds up training.

python run.py --train --env=multi_attacker --train_steps=30000000 --num_cpus=16 --vec_envs=32 --batch_size=16384 --wandb
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
    normalize_angle,
    kick_ball
)

class Env(BaseEnv):
    def __init__(self):
        super().__init__()
        self.rew_map = DotDict({ 
            "goal": 3000,  # Team
            "out_of_bounds": 0,  # Team
            "ball_to_goal": 0.5,  # Team
            "agent_to_ball": 0.3,  # Team
            "kick": 6,  # Individual
            "missed_kick": -5,  # Individual
            "movement_penalty": -0.4,
            "position_penalty": -1,
            "facing_ball": 0.5,
            "position_reward": 0.3,
            "robot_out_of_bounds": 0,
            "too_close": -50,
            "time_step": 0,
        })
        
        self.cfg = DotDict({
            'episode_length': 1000,
            'goal_size': 500,
            'history_length': 3,
            'action_noise': 0,
            'observation_noise': 0,
            'num_opponents': 5,
            'ball': {
                'radius': 10,
                'acl_coef': -0.8,
                'vel_coef': 3,
                'angle_noise': np.pi / 1.5,
                'dist_noise': 2.5
            },
            'agent': {
                'size_x': 300,
                'size_y': 400,
                'disp_coef': 0.15,
                'angle_disp_coef': 0.2,
                'kick_coef': 60,
            },
            'opponent': {
                'radius': 200
            }
        })

        self.possible_agents = [0, 1]
        self.agents = [0, 1]

        self.obs_history = {agent: History(self.cfg.history_length) for agent in self.agents}
        self.reset()

        # 3D vector of (x, y, angle, kick) velocity changes
        action_space = gym.spaces.Box(low=-1, high=1, shape=(5,))
        self.action_spaces = {agent: action_space for agent in self.agents}

        obs_size = len(self.observation(0, include_history=True))
        observation_space = gym.spaces.Box(low=-1, high=1, shape=(obs_size,))
        self.observation_spaces = {agent: observation_space for agent in self.agents}

    def observation(self, i: int, include_history=False):
        agent = self.state.agents[i]
        
        obs = []
        obs.extend(get_rel_obs(agent, self.state.ball))
        obs.extend([1] if can_kick(agent, self.state.ball, tol=150) else [0])

        for j in range(len(self.state.agents)):
            if i != j:
                obs.extend(get_rel_obs(agent, self.state.agents[j]))

        obs.extend(get_rel_obs(agent, [4500, self.cfg.goal_size]))
        obs.extend(get_rel_obs(agent, [4500, -self.cfg.goal_size]))
        obs.extend(get_rel_obs(agent, [-4500, self.cfg.goal_size]))
        obs.extend(get_rel_obs(agent, [-4500, -self.cfg.goal_size]))
        obs.extend(get_rel_obs(agent, [0, 3000]))
        obs.extend(get_rel_obs(agent, [0, -3000]))

        if include_history:
            for prev_obs in self.obs_history[i].get():
                obs.extend(prev_obs)
        
        return obs

    def reset(self, seed=None, return_info=False, options=None, **kwargs):
        obs, info = {}, {}
        self.state = DotDict({
            'step': 0,
            'agents': [
                [ # far robot
                    np.random.uniform(-4500, -0),
                    np.random.uniform(-3000, 3000),
                    np.random.uniform(-np.pi, np.pi)
                ],
                [ # close robot
                    np.random.uniform(0, 4500),
                    np.random.uniform(-3000, 3000),
                    np.random.uniform(-np.pi, np.pi)
                ]
            ]
        })

        if np.random.uniform() < 0.5:
            self.state.ball = [
                np.clip(self.state.agents[1][0] + np.random.uniform(-1000, 1000), -4400, 4400),
                np.clip(self.state.agents[1][1] + np.random.uniform(-1000, 1000), -2900, 2900),
                0,
                0
            ]
        else:
            self.state.ball = [
                np.clip(self.state.agents[0][0] + np.random.uniform(-1000, 1000), -4400, 4400),
                np.clip(self.state.agents[0][1] + np.random.uniform(-1000, 1000), -2900, 2900),
                0,
                0
            ]

        for i in self.agents:
            for _ in range(self.obs_history[i].max_length):
                self.obs_history[i].add(get_rel_obs(self.state.agents[i], self.state.ball))

        for i in self.agents:
            obs[i] = self.observation(i, include_history=True)
            info[i] = {}
                
        return obs, info
    
    def truncate(self):
        return self.state.step > self.cfg.episode_length
    
    def terminate(self):
        return is_goal(self.state.ball, goal_size=self.cfg.goal_size) \
            or is_out_of_bounds(self.state.ball)

    def step(self, joint_action):
        obs, rew, term, trun, info = {}, {}, {}, {}, {}
        self.state.step += 1

        self.state.prev_ball = copy.deepcopy(self.state.ball)
        self.state.prev_agents = copy.deepcopy(self.state.agents)

        self.transition(joint_action)

        for i in self.agents:
            obs[i] = self.observation(i, include_history=True)
            rew[i] = self.reward(i, joint_action[i])
            term[i] = self.terminate()
            trun[i] = self.truncate()
            info[i] = {}

        for i in self.agents:
            self.obs_history[i].add(get_rel_obs(self.state.agents[i], self.state.ball))
        
        return obs, rew, term, trun, info


    def transition(self, joint_action):
        for i in self.agents:
            self.update_agent(i, joint_action[i])
        self.update_ball()
    
    def reward(self, i, action):
        reward = 0
        agent = self.state.agents[i]
        prev_agent = self.state.prev_agents[i]
        ball = self.state.ball
        prev_ball = self.state.prev_ball

        if is_facing(agent, ball, tol=40):
            reward += self.rew_map.facing_ball

        agent_to_goal_distances = [dist(agent, [4800, 0]) for agent in self.state.agents]      
        agent_to_goal_order = np.argsort(agent_to_goal_distances)
        
        # attempting to kick
        if action[3] > 0.6:
            if can_kick(agent, ball, tol=150):
                reward += self.rew_map.kick
            else:
                reward += self.rew_map.missed_kick

        if is_goal(ball, goal_size=self.cfg.goal_size):
            reward += self.rew_map.goal
        
        if agent_to_goal_order[0] == i:
            target = [4500, 0]
        else:
            target = [self.state.agents[agent_to_goal_order[0]][0] + 200, 0]

        reward += self.rew_map.ball_to_goal * (
            dist(prev_ball, target) - dist(ball, target)
        )

        agent_to_ball_distances = [dist(agent, ball) for agent in self.state.agents]      
        agent_to_ball_order = np.argsort(agent_to_ball_distances)

        distance_robot_closest = dist(agent, self.state.agents[agent_to_ball_order[0]])
        if distance_robot_closest < 300 and i != agent_to_ball_order[0]:
            reward += self.rew_map.too_close

        # if closest to ball
        if i == agent_to_ball_order[0]:
            reward += self.rew_map.agent_to_ball * (
                dist(prev_agent, ball) - dist(agent, ball)
            )
        
        if is_out_of_bounds(agent):
            reward += self.rew_map.robot_out_of_bounds

        if action[4] < 0 and i != agent_to_ball_order[0]:
            reward += self.rew_map.movement_penalty

        reward += self.rew_map.time_step
        return reward
    
    def scale_action(self, action):        
        clips = [
            [1, -0.3],
            [0.5, -0.5],
            [1, -1],
        ]

        # clip
        for i in range(len(clips)):
            action[i] = np.clip(action[i], clips[i][1], clips[i][0])

        # Add noise to action (not kick)
        for i in range(len(action) -  1):
            action[i] += np.random.normal(0, self.cfg.action_noise)
        
        return action
    
    def update_agent(self, i, action):
        agent = self.state.agents[i]
        if action[4] > 0:
            # Stand still
            return
        if action[3] > 0.6:
            # Kick the ball
            kick_ball(
                agent, 
                self.state.ball, 
                action[3],
                tol=self.cfg.agent.size_x * 2
            )
        else:
            scaled_action = self.scale_action(action)
            policy_goal_x = agent[0] + (
                (
                    (np.cos(agent[2]) * scaled_action[0])
                    + (
                        np.cos(agent[2] + np.pi / 2)
                        * scaled_action[1]
                    )
                ) * 100
            )
            policy_goal_y = agent[1] + (
                (
                    (np.sin(agent[2]) * scaled_action[0])
                    + (
                        np.sin(agent[2] + np.pi / 2)
                        * scaled_action[1]
                    )
                ) * 100
            )
            # the idea is we move towards the target position and angle
            agent[0] = agent[0] * (1 - self.cfg.agent.disp_coef) + policy_goal_x * self.cfg.agent.disp_coef
            agent[1] = agent[1] * (1 - self.cfg.agent.disp_coef) + policy_goal_y * self.cfg.agent.disp_coef
            agent[2] += scaled_action[2] * self.cfg.agent.angle_disp_coef

        # make sure agent is on field
        agent[0] = np.clip(agent[0], -5200, 5200)
        agent[1] = np.clip(agent[1], -3700, 3700)
        
    def check_collision_ball(self, i):
        agent = self.state.agents[i]
        ball = self.state.ball

        # Rotate ball's center point back to axis-aligned
        dx = ball[0] - agent[0]
        dy = ball[1] - agent[1]
        rotated_x = agent[0] + dx * math.cos(-agent[2]) - dy * math.sin(-agent[2])
        rotated_y = agent[1] + dx * math.sin(-agent[2]) + dy * math.cos(-agent[2])

        # Closest point in the rectangle to the center of circle rotated backwards (unrotated)
        closest_x = min(agent[0] + self.cfg.agent.size_x / 2, max(agent[0] - self.cfg.agent.size_x / 2, rotated_x))
        closest_y = min(agent[1] + self.cfg.agent.size_y / 2, max(agent[1] - self.cfg.agent.size_y / 2, rotated_y))

        # Re-rotate the closest point back to the rotated coordinates
        dx = closest_x - agent[0]
        dy = closest_y - agent[1]
        closest_x = agent[0] + dx * math.cos(agent[2]) - dy * math.sin(agent[2])
        closest_y = agent[1] + dx * math.sin(agent[2]) + dy * math.cos(agent[2])

        # If the distance is less than the ball's radius, an intersection occurs
        collision = dist(ball, [closest_x, closest_y]) <= self.cfg.ball.radius
        
        if collision:
            # Normalize the direction vector by the robot's width and height
            direction_dx = (closest_x - agent[0]) / self.cfg.agent.size_x
            direction_dy = (closest_y - agent[1]) / self.cfg.agent.size_y

            # Calculate the angle between the normalized direction vector and the robot's orientation
            direction_angle = math.atan2(direction_dy, direction_dx) - agent[2]
            direction_angle = normalize_angle(direction_angle)

            # Determine the side of the collision based on the direction angle
            if -math.pi / 4 <= direction_angle < math.pi / 4:
                direction = [1, 0]
            elif math.pi / 4 <= direction_angle < 3 * math.pi / 4:
                direction = [0, 1]
            elif -3 * math.pi / 4 <= direction_angle < -math.pi / 4:
                direction = [0, -1]
            else:
                direction = [-1, 0]

            # Rotate the direction vector back to global coordinates
            direction_x = direction[0] * math.cos(agent[2]) - direction[1] * math.sin(agent[2])
            direction_y = direction[0] * math.sin(agent[2]) + direction[1] * math.cos(agent[2])

            # Convert the direction vector to an angle in radians
            angle = math.atan2(direction_y, direction_x)
        else:
            angle = None

        return collision, angle
    
    def check_collision_robots(self, i, j):
        # Unpack robot properties
        robot1_x, robot1_y, robot1_angle = self.state.agents[i]
        robot2_x, robot2_y, robot2_angle = self.state.agents[j]

        # Rotate robot2's center point back to axis-aligned with robot1
        dx = robot2_x - robot1_x
        dy = robot2_y - robot1_y
        rotated_x = robot1_x + dx * math.cos(-robot1_angle) - dy * math.sin(-robot1_angle)
        rotated_y = robot1_y + dx * math.sin(-robot1_angle) + dy * math.cos(-robot1_angle)

        # Closest point in the rectangle to the center of robot2 rotated backwards (unrotated)
        closest_x = min(robot1_x + self.robot_x_size / 2, max(robot1_x - self.robot_x_size / 2, rotated_x))
        closest_y = min(robot1_y + self.robot_y_size / 2, max(robot1_y - self.robot_y_size / 2, rotated_y))

        # Re-rotate the closest point back to the rotated coordinates
        dx = closest_x - robot1_x
        dy = closest_y - robot1_y
        closest_x = robot1_x + dx * math.cos(robot1_angle) - dy * math.sin(robot1_angle)
        closest_y = robot1_y + dx * math.sin(robot1_angle) + dy * math.cos(robot1_angle)

        # Calculate the distance between the robot2's center and this closest point
        distance = dist([closest_x, closest_y], [robot2_x, robot2_y])

        # If the distance is less than the sum of half of the robots' sizes, an intersection occurs
        return distance <= (self.cfg.agent.size_x + self.cfg.agent.size_y) / 2

    def update_ball(self):
        ball = self.state.ball

        # update ball velocity
        ball[3] += self.cfg.ball.acl_coef
        ball[3] = np.clip(ball[3], 0, 100)

        # update ball position
        ball[0] += ball[3] * math.cos(ball[2])
        ball[1] += ball[3] * math.sin(ball[2])

        # If ball touches robot, push ball away
        for agent in self.agents:
            collision, angle = self.check_collision_ball(agent)
            if collision:
                ball[2] = angle + np.random.uniform(-self.cfg.ball.angle_noise, self.cfg.ball.angle_noise)
                ball[3] = self.cfg.ball.vel_coef * 10