import copy
import functools
import time
import gymnasium as gym
import pygame
import math
import numpy as np
import random
import sys
import rich
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
            "agent_to_point": 2,  # Team
            "missed_kick": -50,  # Individual
            "standing_reward": 10,  # Individual
            "facing_ball": 20,
            "angle_penalty": -10,
            'too_close_to_opponent': -3000
        })

        self.cfg = DotDict({
            'episode_length': 1000,
            'goal_size': 500,
            'history_length': 3,
            'action_noise': 0,
            'observation_noise': 0,
            'num_opponents': 5,
            # possible positions that the off ball robot should go to
            # randomly selected for each episode
            'target_positions': [
                [-3500, 1000], # ball is in positive y
                [-3500,- 1000], # ball is in negative y
                [-500, 0],
                [1500, -1000], # ball is in positive y
                [1500, 1000], # ball is in negative y
            ],
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

        self.possible_agents = [0]
        self.agents = [0]

        self.obs_history = {agent: History(self.cfg.history_length) for agent in self.agents}
        self.reset()

        # 3D vector of (x, y, angle, kick) velocity changes
        action_space = gym.spaces.Box(low=-1, high=1, shape=(5,))
        self.action_spaces = {agent: action_space for agent in self.agents}

        obs_size = len(self.observation(0, include_history=True))
        observation_space = gym.spaces.Box(low=-1, high=1, shape=(obs_size,))
        self.observation_spaces = {agent: observation_space for agent in self.agents}

    def observation(self, agent_idx, include_history=False):
        agent = self.state.agents[agent_idx]
        
        obs = []
        obs.extend(get_rel_obs(agent, self.state.ball))
        obs.extend(get_rel_obs(agent, self.state.target_positions[agent_idx]))

        for opponent in self.state.opponents:
            obs.extend(get_rel_obs(agent, opponent))

        obs.extend(get_rel_obs(agent, [4500, self.cfg.goal_size]))
        obs.extend(get_rel_obs(agent, [4500, -self.cfg.goal_size]))
        obs.extend(get_rel_obs(agent, [-4500, self.cfg.goal_size]))
        obs.extend(get_rel_obs(agent, [-4500, -self.cfg.goal_size]))
        obs.extend(get_rel_obs(agent, [0, 3000]))
        obs.extend(get_rel_obs(agent, [0, -3000]))

        if include_history:
            for prev_obs in self.obs_history[agent_idx].get():
                obs.extend(prev_obs)

        return obs

    def reset(self, seed=None, return_info=False, options=None, **kwargs):
        self.state = DotDict({
            'step': 0,
            'ball': [
                np.random.uniform(-4500, 4500),
                np.random.uniform(-3000, 3000),
                np.random.uniform(0, 2 * np.pi),
                np.random.uniform(20, 30)
            ],
            'agents': [[
                np.random.uniform(-4500, 4500), 
                np.random.uniform(-3000, 3000), 
                np.random.uniform(0, 2 * np.pi)
            ]],
            'opponents': [[
                np.random.uniform(-4500, 4500), 
                np.random.uniform(-3000, 3000), 
                np.random.uniform(0, 2 * np.pi)
            ] for _ in range(self.cfg.num_opponents)],
            'target_positions': [
                self.cfg.target_positions[np.random.choice(len(self.cfg.target_positions))]
            ]
        })
        
        # make sure defenders are not within 500 mm of the agent
        for opponent in self.state.opponents:
            while dist(self.state.agents[0], opponent) < 500:
                opponent[0] = np.random.uniform(-4500, 4500)
                opponent[1] = np.random.uniform(-3000, 3000)

        # put one defender exactly in the middle between the agent and the target position
        self.state.opponents[0] = [
            (self.state.agents[0][0] + self.state.target_positions[0][0]) / 2,
            (self.state.agents[0][1] + self.state.target_positions[0][1]) / 2,
            np.random.uniform(0, 2 * np.pi)
        ]

        obs, info = {}, {}

        for _ in range(self.obs_history[0].max_length):
            self.obs_history[0].add(get_rel_obs(self.state.agents[0], self.state.ball))

        obs[0] = self.observation(0, include_history=True)
        return obs, info
    
    def terminate(self):
        return is_goal(self.state.ball, goal_size=self.cfg.goal_size) \
            or is_out_of_bounds(self.state.ball) \
            or any([ # If distance between agent and a defender is less than 200 mm, terminate
                dist(self.state.agents[0], self.state.opponents[opponent_idx]) < 400
            for opponent_idx in range(self.cfg.num_opponents)])

    def truncate(self):
        return self.state.step > self.cfg.episode_length

    def step(self, joint_action):
        obs, rew, term, trun, info = {}, {}, {}, {}, {}
        self.state.step += 1

        self.state.prev_ball = copy.deepcopy(self.state.ball)
        self.state.prev_agents = copy.deepcopy(self.state.agents)

        if self.state.step % 250 == 0:
            # randomly choose a direction and velocity for the ball
            self.state.ball[2] = np.random.uniform(0, 2 * np.pi)
            self.state.ball[3] = np.random.uniform(60, 70)

        self.transition(joint_action)

        obs[0] = self.observation(0, include_history=True)
        rew[0] = self.reward(0, joint_action[0])
        term[0] = self.terminate()
        trun[0] = self.truncate()
        info[0] = {}

        self.obs_history[0].add(get_rel_obs(self.state.agents[0], self.state.ball))

        return obs, rew, term, trun, info

    def reward(self, agent_idx, action):
        agent = self.state.agents[agent_idx]
        prev_agent = self.state.prev_agents[agent_idx]
        target = self.state.target_positions[agent_idx]
        ball = self.state.ball

        reward = 0
        reward += self.rew_map.angle_penalty * np.abs(action[2])
        
        # reward for moving toward role location
        reward += self.rew_map.agent_to_point * (dist(prev_agent, target) - dist(agent, target))

        # attempting to kick
        if action[3] > 0.6:
            reward += self.rew_map.missed_kick

        # If with 200 mm of point, reward for standing still
        if dist(agent, target) < 200:
            if is_facing(agent, ball, tol=10):
                reward += self.rew_map.facing_ball

            if action[4] > 0:
                reward += self.rew_map.standing_reward
        else:
            if is_facing(agent, ball, tol=40):
                reward += self.rew_map.facing_ball

        for opponents in self.state.opponents:
            if dist(agent, opponents) < 400:
                reward = self.rew_map.too_close_to_opponent

        return reward
    
    def transition(self, joint_action):
        self.update_agent(0, joint_action[0])
        self.update_ball()

    def scale_action(self, action):
        clips = [
            [1, -0.3],
            [0.5, -0.5],
            [1, -1]
        ]

        for i in range(len(clips)):
            action[i] = np.clip(action[i], clips[i][1], clips[i][0])
        
        return action
    
    def update_agent(self, agent_idx, action):
        agent = self.state.agents[agent_idx]

        # stand still
        if action[4] > 0:
            return
        
        # kick the ball
        if action[3] > 0.6:
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
        
    def check_collision_ball(self, agent_idx):
        agent = self.state.agents[agent_idx]
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
        
        # If ball OOB, clip to in bounds and change direction
        if is_out_of_bounds(ball):
            ball[0] = np.clip(ball[0], -4500, 4500)
            ball[1] = np.clip(ball[1], -3000, 3000)
            ball[2] = np.random.uniform(0, 2 * np.pi)