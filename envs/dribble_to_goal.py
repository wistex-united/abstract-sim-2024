import copy
import gymnasium as gym
import math
import numpy as np
from envs.base import BaseEnv
import rich

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
            "goal": 3000,  # Team
            "out_of_bounds": -1000,  # Team
            "ball_to_goal": 0.5,  # Team
            "agent_to_ball": 0.3,  # Team
            "movement_penalty": -5,
            "position_penalty": -1,
            "facing_ball": 3,
            "agent_out_of_bounds": -50,
            "too_close": -50,
            "time_step": -1,
        })

        self.cfg = DotDict({
            'episode_length': 1000,
            'goal_size': 500, # half of the goal size
            'history_length': 3,
            'action_noise': 0,
            'observation_noise': 0,
            'ball': {
                'radius': 10,
                'acl_coef': -0.8,
                'vel_coef': 3,
                'angle_noise': 0,
                'dist_noise': 2.5
            },
            'agent': {
                'size_x': 300,
                'size_y': 400,
                'disp_coef': 0.15,
                'angle_disp_coef': 0.1,
                'kick_coef': 60,
            },
        })

        self.possible_agents = [0]
        self.agents = [0]

        self.obs_history = { agent: History(self.cfg.history_length) for agent in self.agents }
        self.reset()

        # [x, y, angle]
        action_space = gym.spaces.Box(low=-1, high=1, shape=(3,))
        self.action_spaces = { agent: action_space for agent in self.agents }
        
        obs_size = len(self.observation(0, include_history=True))
        observation_space = gym.spaces.Box(low=-1, high=1, shape=(obs_size,))
        self.observation_spaces = {agent: observation_space for agent in self.agents}

    def observation(self, agent_idx, include_history=False):
        agent = self.state.agents[agent_idx]
        
        obs = []
        obs.extend(get_rel_obs(agent, self.state.ball))
        obs.extend([1] if can_kick(agent, self.state.ball, tol=150) else [0])
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
            'agents': [
                [
                    np.random.uniform(-4500, 4500),
                    np.random.uniform(-3000, 3000),
                    np.random.uniform(-np.pi, np.pi)
                ]
            ],
            'ball': [
                np.random.uniform(-4500, 4500),
                np.random.uniform(-3000, 3000),
                0,
                0
            ]
        })

        obs, info = {}, {}
        
        # populate history with initial obs 
        for _ in range(self.obs_history[0].max_length):
            self.obs_history[0].add(get_rel_obs(self.state.agents[0], self.state.ball))

        obs[0] = self.observation(0, include_history=True)
        return obs, info
    
    def terminate(self):
        return is_goal(self.state.ball, goal_size=self.cfg.goal_size) or is_out_of_bounds(self.state.ball)

    def truncate(self):
        return self.state.step > self.cfg.episode_length

    def step(self, joint_action):
        obs, rew, term, trun, info = {}, {}, {}, {}, {}
        self.state.step += 1

        self.state.prev_ball = copy.deepcopy(self.state.ball)
        self.state.prev_agents = copy.deepcopy(self.state.agents)

        self.transition(joint_action)

        obs[0] = self.observation(0, include_history=True)
        rew[0] = self.reward(0, joint_action[0])
        term[0] = self.terminate()
        trun[0] = self.truncate()
        info[0] = {}

        self.obs_history[0].add(get_rel_obs(self.state.agents[0], self.state.ball))

        return obs, rew, term, trun, info

    def transition(self, joint_action):
        self.update_agent(0, joint_action[0])
        self.update_ball()

    def reward(self, agent_idx, action):
        ball = self.state.ball
        prev_ball = self.state.prev_ball
        agent = self.state.agents[agent_idx]
        prev_agent = self.state.prev_agents[agent_idx]

        reward = 0

        if is_facing(agent, self.state.ball, tol=10):
            reward += self.rew_map.facing_ball

        if is_goal(self.state.ball, goal_size=self.cfg.goal_size):
            reward += self.rew_map.goal

        reward += self.rew_map.ball_to_goal * (
            dist(prev_ball, [4500, 0]) -
            dist(ball, [4500, 0])
        )

        reward += self.rew_map.agent_to_ball * (
            dist(prev_agent, ball) - 
            dist(agent, ball)
        )
        
        if is_out_of_bounds(agent, tol=100):
            reward += self.rew_map.agent_out_of_bounds

        if is_out_of_bounds(ball):
            reward += self.rew_map.out_of_bounds

        return reward

    def scale_action(self, action):
        clips = [
            [1, -1],
            [1, -1],
            [1, -1],
            [1, -1]
        ]
        
        for i in range(len(clips)):
            action[i] = np.clip(action[i], clips[i][1], clips[i][0])

        scales = [
            [1, -0.3], 
            [0.5, -0.5], 
            [1, -1]
        ]

        # scale actions
        for i in range(len(scales[0])):
            if action[i] > 0:
                action[i] = action[i] * scales[i][0]
            else:
                action[i] = action[i] * scales[i][1]

        return action
    
    def update_agent(self, agent_idx, action):
        agent = self.state.agents[agent_idx]
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

        # Calculate the distance between the circle's center and this closest point
        distance = math.sqrt((closest_x - ball[0]) ** 2 + (closest_y - ball[1]) ** 2)
        
        # If the distance is less than the ball's radius, an intersection occurs
        collision = distance <= self.cfg.ball.radius
        
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
