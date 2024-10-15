import time

from pettingzoo import ParallelEnv
import pygame
import numpy as np
import functools

from utils.env import (
    DotDict
)

class BaseEnv(ParallelEnv):
    metadata = {
        "render_modes": ["human"],
        "render_fps": 30,
    }

    def __init__(self, render_mode="human"):
        super().__init__()
        self.render_mode = render_mode
        self.render_init = False

        # all dimensions are in mm proportional to render_length
        # based on the official robocup rule book ratio
        self.render_cfg = DotDict({
            'render_length': 1200,
            'field': {
                'color': (17, 130, 59),
                'line_color': (206, 202, 189),
                'out_of_bounds_line_color': (206, 202, 189),
                'left_goal_color': (240,212,58),
                'right_goal_color': (64,223,239),
                'length': 9000,
                'width': 6000,
                'border_width': 700,
                'line_width': 2,
                'penalty_mark_size': 100,
                'goal_area_length': 600,
                'goal_area_width': 2200,
                'goal_length': 500,
                'goal_width': 750 * 2,
                'penalty_area_length': 1650,
                'penalty_area_width': 4000,
                'penalty_mark_distance': 1300,
                'center_circle_diameter': 1500,
            },
            'ball': {
                'color': (255, 255, 255)
            },
            'agent': {
                'color': (240,212,58),
                'stroke_width': 4,
                'direction_color': (240,212,58),
                'direction_width': 2,
                'label_size': 20,
                'label_color': (240,212,58)
            },
            'target': {
                'radius_color': (255, 255, 255),
                'radius_line_width': 2,
                'color': (255, 255, 255),
                'line_width': 2,
                'direction_color': (255, 255, 255),
                'direction_length': 16,
                'direction_width': 2,
            },
            'opponent': {
                'color': (64,223,239),
                'stroke_width': 4,
                'direction_color': (64,223,239),
                'direction_width': 2,
                'label_size': 20,
                'label_color': (64,223,239),
            }
        })

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self.observation_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self.action_spaces[agent]

    def init_pygame(self):
        pygame.init()
        field_length_render = (
            self.render_cfg.field.length * self.render_cfg.render_length / 
            (self.render_cfg.field.length + 2 * self.render_cfg.field.border_width)
        )

        field_width_render = (
            self.render_cfg.field.width * self.render_cfg.render_length / 
            (self.render_cfg.field.length + 2 * self.render_cfg.field.border_width)
        )

        border_width_render = (
            self.render_cfg.field.border_width * self.render_cfg.render_length /
            (self.render_cfg.field.length + 2 * self.render_cfg.field.border_width)
        )

        surface_width = int(field_length_render + 2 * border_width_render)
        surface_height = int(field_width_render + 2 * border_width_render)

        self.field = pygame.display.set_mode((surface_width, surface_height))
        pygame.display.set_caption("AbstractSim")
        self.clock = pygame.time.Clock()

    def render(self, mode="human"):
        time.sleep(0.01)

        if not self.render_init:
            self.init_pygame()
            self.render_init = True

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        # if space on keyboard is pressed, reset the environment
        keys = pygame.key.get_pressed()
        if keys[pygame.K_SPACE]:
            self.reset()

        self.render_field()

        if 'target' in self.state:
            self.render_target(self.state.target)

        if 'ball' in self.state:
            self.render_ball(self.state.ball)

        if 'agents' in self.state:
            self.render_agents(self.state.agents)

        if 'opponents' in self.state:
            self.render_opponents(self.state.opponents)
        
        if 'teammates' in self.state:
            self.render_teammates(self.state.teammates)

        pygame.display.update()
        self.clock.tick(60)

    def render_target(self, target):
        target_x = int((target[0] / 5200 + 1) * (self.render_cfg.render_length / 2))
        target_y = int((target[1] / 3700 + 1) * (self.render_cfg.render_length / 3))
        ball_x = int((self.state.ball[0] / 5200 + 1) * (self.render_cfg.render_length / 2))
        ball_y = int((self.state.ball[1] / 3700 + 1) * (self.render_cfg.render_length / 3))

        # render target radius
        pygame.draw.circle(
            self.field,
            pygame.Color(*self.render_cfg.target.radius_color),
            (ball_x, ball_y),
            56,
            self.render_cfg.target.radius_line_width
        )

        pygame.draw.circle(
            self.field,
            pygame.Color(*self.render_cfg.target.color),
            (target_x, target_y),
            16,
            self.render_cfg.target.line_width,
        )

        pygame.draw.line(
            self.field,
            pygame.Color(*self.render_cfg.target.direction_color),
            (target_x, target_y),
            (
                target_x + self.render_cfg.target.direction_length * np.cos(target[2]),
                target_y + self.render_cfg.target.direction_length * np.sin(target[2]),
            ),
            width=self.render_cfg.target.direction_width,
        )

    def render_ball(self, ball):
        ball_x = int((ball[0] / 5200 + 1) * (self.render_cfg.render_length / 2))
        ball_y = int((ball[1] / 3700 + 1) * (self.render_cfg.render_length / 3))
        pygame.draw.circle(
            self.field,
            pygame.Color(*self.render_cfg.ball.color),
            (ball_x, ball_y),
            self.cfg.ball.radius
        )

    def render_agents(self, agents):
        size_x = self.cfg.agent.size_x
        size_y = self.cfg.agent.size_y

        for i in range(len(agents)):
            agent_x = int((agents[i][0] / 5200 + 1) * (self.render_cfg.render_length / 2))
            agent_y = int((agents[i][1] / 3700 + 1) * (self.render_cfg.render_length / 3))
            
            # draw robot direction
            pygame.draw.line(
                self.field,
                pygame.Color(*self.render_cfg.agent.direction_color),
                (agent_x, agent_y),
                (
                    agent_x + size_x / 20 * np.cos(agents[i][2]),
                    agent_y + size_y / 20 * np.sin(agents[i][2]),
                ),
                width = self.render_cfg.agent.direction_width,
            )

            # display robot number
            font = pygame.font.SysFont("Arial", self.render_cfg.agent.label_size)
            text = font.render(f"{i}", True, self.render_cfg.agent.label_color)
            text_rect = text.get_rect()
            text_rect.center = (agent_x, agent_y)
            self.field.blit(text, text_rect)
            
            # create a new surface with the size of the robot
            robot_surface = pygame.Surface(
                (size_x / 10, size_y / 10), pygame.SRCALPHA
            )

            # draw a rectangle on the new surface
            pygame.draw.rect(
                robot_surface,
                pygame.Color(*self.render_cfg.agent.color),
                pygame.Rect(0, 0, size_x / 10, size_y / 10),
                width=self.render_cfg.agent.stroke_width,
            )

            # create a new surface that's centered on the original surface
            centered_surface = pygame.Surface((size_x / 5, size_y / 5), pygame.SRCALPHA)
            centered_surface.blit(robot_surface, (size_x / 20, size_y / 20))

            # rotate the surface
            rotated_surface = pygame.transform.rotate(centered_surface, -agents[i][2] * 180 / np.pi)

            # calculate the position of the rotated surface
            agent_x -= rotated_surface.get_width() / 2
            agent_y -= rotated_surface.get_height() / 2

            # Draw the rotated surface on the field
            self.field.blit(rotated_surface, (agent_x, agent_y))

    def render_opponents(self, opponents):
        for i in range(len(opponents)):
            render_robot_x = int((opponents[i][0] / 5200 + 1) * (self.render_cfg.render_length / 2))
            render_robot_y = int((opponents[i][1] / 3700 + 1) * (self.render_cfg.render_length / 3))
            
            pygame.draw.circle(
                self.field, 
                self.render_cfg.opponent.color, 
                (render_robot_x, render_robot_y), 
                self.cfg.opponent.radius * self.render_cfg.render_length / (self.render_cfg.field.length + 2 * self.render_cfg.field.border_width),
                self.render_cfg.opponent.stroke_width
            )

            pygame.draw.line(
                self.field,
                self.render_cfg.opponent.direction_color,
                (render_robot_x, render_robot_y),
                (
                    render_robot_x + 20 * np.cos(opponents[i][2]),
                    render_robot_y + 20 * np.sin(opponents[i][2]),
                ),
                width=self.render_cfg.opponent.direction_width,
            )

            # add robot number
            font = pygame.font.SysFont("Arial", self.render_cfg.opponent.label_size)
            text = font.render(f"{i}", True, self.render_cfg.opponent.label_color)

            text_rect = text.get_rect()
            text_rect.center = (render_robot_x, render_robot_y)
            self.field.blit(text, text_rect)
    
    def render_teammates(self, teammates):
        pass

    def render_field(self):
        render_length = self.render_cfg.render_length

        field_length = self.render_cfg.field.length
        field_width = self.render_cfg.field.width
        border_width = self.render_cfg.field.border_width

        penalty_mark_size = self.render_cfg.field.penalty_mark_size
        goal_area_length = self.render_cfg.field.goal_area_length
        goal_area_width = self.render_cfg.field.goal_area_width
        goal_length = self.render_cfg.field.goal_length
        goal_width = self.render_cfg.field.goal_width
        penalty_area_length = self.render_cfg.field.penalty_area_length
        penalty_area_width = self.render_cfg.field.penalty_area_width
        penalty_mark_distance = self.render_cfg.field.penalty_mark_distance
        center_circle_diameter = self.render_cfg.field.center_circle_diameter
        line_width = self.render_cfg.field.line_width

        field_length_render = (
            field_length * render_length / 
            (field_length + 2 * border_width)
        )

        field_width_render = (
            field_width * render_length / 
            (field_length + 2 * border_width)
        )

        penalty_mark_size_render = (
            penalty_mark_size * render_length / 
            (field_length + 2 * border_width)
        )

        goal_area_length_render = (
            goal_area_length * render_length / 
            (field_length + 2 * border_width)
        )

        goal_area_width_render = (
            goal_area_width * render_length / 
            (field_length + 2 * border_width)
        )

        goal_length_render = (
            goal_length * render_length / 
            (field_length + 2 * border_width)
        )

        goal_width_render = (
            goal_width * render_length / 
            (field_length + 2 * border_width)
        )

        penalty_area_length_render = (
            penalty_area_length * render_length / 
            (field_length + 2 * border_width)
        )

        penalty_area_width_render = (
            penalty_area_width * render_length / 
            (field_length + 2 * border_width)
        )

        penalty_mark_distance_render = (
            penalty_mark_distance * render_length
            / (field_length + 2 * border_width)
        )

        center_circle_diameter_render = (
            center_circle_diameter * render_length / 
            (field_length + 2 * border_width)
        )

        border_width_render = int(
            border_width * render_length / 
            (field_length + 2 * border_width)
        )

        surface_width = int(field_length_render + 2 * border_width_render)

        # constant here is just to make it look correct, unsure why it is needed
        surface_height = int(field_width_render + 2 * border_width_render)

        field_color = self.render_cfg.field.color
        line_color = self.render_cfg.field.line_color
        out_of_bounds_line_color = self.render_cfg.field.out_of_bounds_line_color
        left_goal_color = self.render_cfg.field.left_goal_color
        right_goal_color = self.render_cfg.field.right_goal_color

        self.field.fill(field_color)

        # draw center line
        pygame.draw.line(
            self.field,
            pygame.Color(*line_color),
            (surface_width / 2, border_width_render),
            (surface_width / 2, surface_height - border_width_render),
            width=line_width,
        )

        # draw center circle
        pygame.draw.circle(
            self.field,
            pygame.Color(*line_color),
            (int(surface_width / 2), int(surface_height / 2)),
            int(center_circle_diameter_render / 2),
            width=line_width,
        )

        # draw center dot
        pygame.draw.circle(
            self.field,
            pygame.Color(*line_color),
            (int(surface_width / 2), int(surface_height / 2)),
            int(line_width / 2),
        )

        # left penalty area
        # should be 1650mm long and 4000mm wide starting at goal line
        pygame.draw.line(
            self.field,
            pygame.Color(*line_color),
            (
                border_width_render,
                surface_height / 2 - penalty_area_width_render / 2,
            ),
            (
                border_width_render + penalty_area_length_render,
                surface_height / 2 - penalty_area_width_render / 2,
            ),
            width=line_width,
        )

        pygame.draw.line(
            self.field,
            pygame.Color(*line_color),
            (
                border_width_render,
                surface_height / 2 + penalty_area_width_render / 2,
            ),
            (
                border_width_render + penalty_area_length_render,
                surface_height / 2 + penalty_area_width_render / 2,
            ),
            width=line_width,
        )

        pygame.draw.line(
            self.field,
            pygame.Color(*line_color),
            (
                border_width_render + penalty_area_length_render,
                surface_height / 2 - penalty_area_width_render / 2,
            ),
            (
                border_width_render + penalty_area_length_render,
                surface_height / 2 + penalty_area_width_render / 2,
            ),
            width=line_width,
        )

        # right penalty area
        # should be 1650mm long and 4000mm wide starting at goal line
        pygame.draw.line(
            self.field,
            pygame.Color(*line_color),
            (
                surface_width - border_width_render,
                surface_height / 2 - penalty_area_width_render / 2,
            ),
            (
                surface_width - border_width_render - penalty_area_length_render,
                surface_height / 2 - penalty_area_width_render / 2,
            ),
            width=line_width,
        )

        pygame.draw.line(
            self.field,
            pygame.Color(*line_color),
            (
                surface_width - border_width_render,
                surface_height / 2 + penalty_area_width_render / 2,
            ),
            (
                surface_width - border_width_render - penalty_area_length_render,
                surface_height / 2 + penalty_area_width_render / 2,
            ),
            width=line_width,
        )

        pygame.draw.line(
            self.field,
            pygame.Color(*line_color),
            (
                surface_width - border_width_render - penalty_area_length_render,
                surface_height / 2 - penalty_area_width_render / 2,
            ),
            (
                surface_width - border_width_render - penalty_area_length_render,
                surface_height / 2 + penalty_area_width_render / 2,
            ),
            width=line_width,
        )

        # left goal area
        # should be 600mm long and 2200mm wide starting at goal line
        pygame.draw.line(
            self.field,
            pygame.Color(*line_color),
            (
                border_width_render,
                surface_height / 2 - goal_area_width_render / 2,
            ),
            (
                border_width_render + goal_area_length_render,
                surface_height / 2 - goal_area_width_render / 2,
            ),
            width=line_width,
        )

        pygame.draw.line(
            self.field,
            pygame.Color(*line_color),
            (
                border_width_render,
                surface_height / 2 + goal_area_width_render / 2,
            ),
            (
                border_width_render + goal_area_length_render,
                surface_height / 2 + goal_area_width_render / 2,
            ),
            width=line_width,
        )

        pygame.draw.line(
            self.field,
            pygame.Color(*line_color),
            (
                border_width_render + goal_area_length_render,
                surface_height / 2 - goal_area_width_render / 2,
            ),
            (
                border_width_render + goal_area_length_render,
                surface_height / 2 + goal_area_width_render / 2,
            ),
            width=line_width,
        )

        # right goal area
        # should be 600mm long and 2200mm wide starting at goal line
        pygame.draw.line(
            self.field,
            pygame.Color(*line_color),
            (
                surface_width - border_width_render,
                surface_height / 2 - goal_area_width_render / 2,
            ),
            (
                surface_width - border_width_render - goal_area_length_render,
                surface_height / 2 - goal_area_width_render / 2,
            ),
            width=line_width,
        )

        pygame.draw.line(
            self.field,
            pygame.Color(*line_color),
            (
                surface_width - border_width_render,
                surface_height / 2 + goal_area_width_render / 2,
            ),
            (
                surface_width - border_width_render - goal_area_length_render,
                surface_height / 2 + goal_area_width_render / 2,
            ),
            width=line_width,
        )

        pygame.draw.line(
            self.field,
            pygame.Color(*line_color),
            (
                surface_width - border_width_render - goal_area_length_render,
                surface_height / 2 - goal_area_width_render / 2,
            ),
            (
                surface_width - border_width_render - goal_area_length_render,
                surface_height / 2 + goal_area_width_render / 2,
            ),
            width=line_width,
        )

        # left penalty mark
        # should be 100mm in diameter and 1300mm from goal line
        pygame.draw.circle(
            self.field,
            pygame.Color(*line_color),
            (
                border_width_render + penalty_mark_distance_render,
                surface_height / 2,
            ),
            int(penalty_mark_size_render / 2),
            width=line_width,
        )

        # right penalty mark
        # should be 100mm in diameter and 1300mm from goal line
        pygame.draw.circle(
            self.field,
            pygame.Color(*line_color),
            (
                surface_width
                - border_width_render
                - penalty_mark_distance_render,
                surface_height / 2,
            ),
            int(penalty_mark_size_render / 2),
            width=line_width,
        )

        # center point, same size as penalty mark
        pygame.draw.circle(
            self.field,
            pygame.Color(*line_color),
            (surface_width / 2, surface_height / 2),
            int(penalty_mark_size_render / 2),
            width=line_width,
        )

        # left goal area
        pygame.draw.rect(
            self.field,
            pygame.Color(*left_goal_color),
            (
                border_width_render - goal_length_render,
                surface_height / 2 - goal_width_render / 2 - line_width / 2,
                goal_length_render,
                goal_width_render,
            ),
        )

        # TODO: Make goal areas look better
        # Draw lines around goal areas
        # Left goal area
        # pygame.draw.line(
        #     self.field,
        #     pygame.Color(*line_color)
        #     (border_width_render - goal_area_length_render, surface_height / 2 - goal_area_width_render / 2),
        #     (border_width_render, surface_height / 2 - goal_area_width_render / 2),
        #     width=line_width,
        # )
        # pygame.draw.line(
        #     self.field,
        #     pygame.Color(*line_color)
        #     (border_width_render - goal_area_length_render, surface_height / 2 + goal_area_width_render / 2),
        #     (border_width_render, surface_height / 2 + goal_area_width_render / 2),
        #     width=line_width,
        # )

        # right goal area
        pygame.draw.rect(
            self.field,
            pygame.Color(*right_goal_color),
            (
                surface_width - border_width_render,
                surface_height / 2 - goal_width_render / 2 - line_width / 2,
                goal_length_render,
                goal_width_render,
            ),
        )

        # draw out of bounds lines
        pygame.draw.line(
            self.field,
            pygame.Color(*out_of_bounds_line_color),
            (border_width_render, border_width_render),
            (surface_width - border_width_render, border_width_render),
            width=line_width,
        )

        pygame.draw.line(
            self.field,
            pygame.Color(*out_of_bounds_line_color),
            (border_width_render, surface_height - border_width_render),
            (
                surface_width - border_width_render,
                surface_height - border_width_render,
            ),
            width=line_width,
        )

        pygame.draw.line(
            self.field,
            pygame.Color(*out_of_bounds_line_color),
            (border_width_render, border_width_render),
            (border_width_render, surface_height - border_width_render),
            width=line_width,
        )

        pygame.draw.line(
            self.field,
            pygame.Color(*out_of_bounds_line_color),
            (surface_width - border_width_render, border_width_render),
            (
                surface_width - border_width_render,
                surface_height - border_width_render,
            ),
            width=line_width,
        )
