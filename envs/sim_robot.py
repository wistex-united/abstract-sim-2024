"""
This environment is a wrapper around the SimRobot simulator. It works by
launching the simulator as a subprocess and communicating with it via named
pipes. The simulator will handle updating the environment at every time step
and will send observations and rewards back to this environment. This python
process will update the policy and send actions back to the cpp process.
Every episode, the simulator will be restarted with a new random scene.

Changes to the reward function or observation space are done in the cpp
code. If the length of the observation vector changes, make sure to update
the OBS_SIZE constant in this file.

1. Modify GameController.cpp and Behavior.cpp in the cpp codebase.
   Look for comments in those files that start with the string
   "[SimRobot training]" for more details. There should be two places
   total.
2. Compile the cpp code for the SimRobot target. Any changes to the cpp
   code will require recompiling before running this python code.
3. Run the following command:
    >> python run.py --train --env SimRobot --train_steps 20000

If the simulator seems stuck at the very beginning, it may be because the
observation vector being sent by the cpp code is of different length than
what is expected by this python code. Double check that the observation
vectors match in length. Also check the lengths of the action vectors
being passed from python to cpp.

Note that this environment does not support the --render option when
running run.py.
"""

import random
import os
import subprocess
import sys
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from overrides import overrides
from gymnasium.core import ActType, ObsType
from typing import List, Optional, SupportsFloat, Any, Tuple
from utils.named_pipe_utils import (
    open_pipe_for_reading,
    open_pipe_for_writing,
    read_vector_from_pipe,
    write_vector_to_pipe,
)

DEBUG_PRINTS = False
OBS_SIZE = 12

# these paths must match the ones used by the c++ code
CPP_REPO_DIR = "/Users/jkelle/BadgerRLSystem/"
PIPE_PATH_OBSERVATION = os.path.join(CPP_REPO_DIR, "pipe_observation")
PIPE_PATH_REWARD = os.path.join(CPP_REPO_DIR, "pipe_reward")
PIPE_PATH_TERMINATED = os.path.join(CPP_REPO_DIR, "pipe_terminated")
PIPE_PATH_TRUNCATED = os.path.join(CPP_REPO_DIR, "pipe_truncated")
PIPE_PATH_ACTION = os.path.join(CPP_REPO_DIR, "pipe_action")
PIPE_PATH_EXIT = os.path.join(CPP_REPO_DIR, "pipe_exit")


def print_debug(message: str) -> None:
    if DEBUG_PRINTS:
        print(message)

def get_random_float_in_range(min_val: float, max_val: float) -> float:
    range = max_val - min_val
    return random.random() * range + min_val

def get_random_int_in_range(min_val: int, max_val: int) -> int:
    range = max_val - min_val
    return random.randint(0, range) + min_val

# in meters
PENALTY_AREA_X = 2.850
PENALTY_AREA_Y = 2.000
GOAL_AREA_X = 3.900
GOAL_AREA_Y = 1.100
GOAL_LINE_X = 4.500
GOAL_POST_Y = 0.800

def create_random_scene_penalty_box() -> None:
    robot_spawn_box = [
        (PENALTY_AREA_X, GOAL_LINE_X - 0.1), # (minx, maxx)
        (-PENALTY_AREA_Y, PENALTY_AREA_Y), # (miny, maxy)
    ]
    ball_spawn_box = [
        (PENALTY_AREA_X + 0.2, GOAL_LINE_X - 0.1), # (minx, maxx)
        (-PENALTY_AREA_Y + 0.1, PENALTY_AREA_Y - 0.1), # (miny, maxy)
    ]
    create_random_scene(robot_spawn_box, ball_spawn_box)

def create_random_scene_goal_box() -> None:
    robot_spawn_box = [
        (PENALTY_AREA_X + 0.1, GOAL_AREA_X - 0.1), # (minx, maxx)
        (-GOAL_AREA_Y, GOAL_AREA_Y), # (miny, maxy)
    ]
    ball_spawn_box = [
        (GOAL_AREA_X + 0.1, GOAL_LINE_X - 0.1), # (minx, maxx)
        (-GOAL_AREA_Y, GOAL_AREA_Y), # (miny, maxy)
    ]
    create_random_scene(robot_spawn_box, ball_spawn_box)

def create_random_scene_goal_post() -> None:
    robot_spawn_box = [
        (PENALTY_AREA_X + 0.1, GOAL_LINE_X - 0.1), # (minx, maxx)
        (-GOAL_AREA_Y, GOAL_AREA_Y), # (miny, maxy)
    ]
    opp_goalie_spawn_box = [
        (GOAL_LINE_X - 0.1, GOAL_LINE_X), # (minx, maxx)
        (-GOAL_POST_Y + 0.2, GOAL_POST_Y - 0.2), # (miny, maxy)
    ]
    ball_spawn_box = [
        (GOAL_AREA_X + 0.1, GOAL_LINE_X - 0.1), # (minx, maxx)
        (-GOAL_POST_Y, GOAL_POST_Y), # (miny, maxy)
    ]
    create_random_scene(robot_spawn_box, opp_goalie_spawn_box, ball_spawn_box)

def create_random_scene(
    robot_spawn_box: List[Tuple[float, float]],
    opp_goalie_spawn_box: List[Tuple[float, float]],
    ball_spawn_box: List[Tuple[float, float]],
) -> None:
    with open(os.path.join(CPP_REPO_DIR, "Config/Scenes/randomSceneTemplate.txt")) as f:
        template = f.read()

    (robot_min_x, robot_max_x), (robot_min_y, robot_max_y) = robot_spawn_box
    robotX = get_random_float_in_range(-1 * robot_max_x, -1 * robot_min_x)
    robotY = get_random_float_in_range(robot_min_y, robot_max_y)

    (opp_goalie_min_x, opp_goalie_max_x), (opp_goalie_min_y, opp_goalie_max_y) = opp_goalie_spawn_box
    ok = False
    while not ok:
        oppGoalieX = get_random_float_in_range(-1 * opp_goalie_max_x, -1 * opp_goalie_min_x)
        oppGoalieY = get_random_float_in_range(opp_goalie_min_y, opp_goalie_max_y)
        ok = (oppGoalieX - robotX)**2 + (oppGoalieY - robotY)**2 > 0.2

    (ball_min_x, ball_max_x), (ball_min_y, ball_max_y) = ball_spawn_box
    ok = False
    while not ok:
        ballX = get_random_float_in_range(-1 * ball_max_x, -1 * ball_min_x)
        ballY = get_random_float_in_range(ball_min_y, ball_max_y)
        ok = (
            (ballX - robotX)**2 + (ballY - robotY)**2 > 0.2
            and (ballX - oppGoalieX)**2 + (ballY - oppGoalieY)**2 > 0.2
        )

    with open(os.path.join(CPP_REPO_DIR, "Config/Scenes/randomScene.ros2"), "w") as f:
        num_bytes_written = f.write(template.format(
            robotX=robotX,
            robotY=robotY,
            robotRotationDegrees=get_random_int_in_range(-180, 180),
            oppGoalieX=oppGoalieX,
            oppGoalieY=oppGoalieY,
            oppGoalieRotationDegrees=get_random_int_in_range(-0, 0),
            ballX=ballX,
            ballY=ballY,
        ))
        print_debug(f"Wrote {num_bytes_written} bytes to randomScene.ros2")

def start_cpp_simulator() -> int:
    # I haven't figured out how to use this PID to kill the process yet.
    # It seems the simulator will spawn its own child processes, so killing
    # the parent process doesn't kill the simulator.

    # Note: When I launch the simulator on Linux, it seems to be a subprocess
    # called Main and will terminate if this python process is killed.

    if sys.platform.startswith("win"):
        raise NotImplementedError("Launching SimRobot on Windows not supported")
    elif sys.platform.startswith("darwin"):
        # MACOS specific command
        command = ["open", "-g", "./Config/Scenes/randomScene.ros2"]
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=CPP_REPO_DIR,
        )
    elif sys.platform.startswith("linux"):
        compileCommand = ["./Make/Linux/compile", "Release", "SimRobot"]
        subprocess.Popen(
            compileCommand,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=CPP_REPO_DIR,
        )
        runCommand = ["./Build/Linux/SimRobot/Release/SimRobot","-g","./Config/Scenes/randomScene.ros2"]
        process = subprocess.Popen(
            runCommand,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=CPP_REPO_DIR,
        )
    else:
        raise NotImplementedError("Unsupported platform")
    print_debug(f"Opened simulator with PID: {process.pid}")
    return process.pid

class SimRobot(gym.Env):

    action_space: spaces.Space[ActType] = gym.spaces.Box(
        np.array([-1, -1, -1]),
        np.array([1, 1, 1]),
    )
    observation_space: spaces.Space[ObsType] = gym.spaces.Box(
        low=-1,
        high=1,
        shape=(OBS_SIZE,)
    )

    def __init__(self):
        super().__init__()
        self.episode_count = 0
        self.episode_step_count = 0
        self.global_step_count = 0

    def _open_pipes(self) -> None:
        """
        These must be opened in the same order in which the CPP opens them.
        """
        print_debug("Opening pipe_observation...")
        self.pipe_observation = open_pipe_for_reading(PIPE_PATH_OBSERVATION)
        print_debug("Opening pipe_reward...")
        self.pipe_reward = open_pipe_for_reading(PIPE_PATH_REWARD)
        print_debug("Opening pipe_terminated...")
        self.pipe_terminated = open_pipe_for_reading(PIPE_PATH_TERMINATED)
        print_debug("Opening pipe_truncated...")
        self.pipe_truncated = open_pipe_for_reading(PIPE_PATH_TRUNCATED)
        print_debug("Opening pipe_action...")
        self.pipe_action = open_pipe_for_writing(PIPE_PATH_ACTION)
        print_debug("Opening pipe_exit...")
        self.pipe_exit = open_pipe_for_writing(PIPE_PATH_EXIT)
        print_debug("Opened all pipes")

    def _close_pipes(self) -> None:
        self.pipe_observation.close()
        self.pipe_reward.close()
        self.pipe_terminated.close()
        self.pipe_truncated.close()
        self.pipe_action.close()
        self.pipe_exit.close()

    @overrides
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        self.episode_count += 1
        self.episode_step_count = 0
        print(f"Starting episode {self.episode_count}")

        print_debug("creating random scene...")
        create_random_scene_goal_post()
        print_debug("created random scene")

        max_trials = 3
        for trial in range(max_trials):
            print_debug("Starting cpp simulator...")
            self.sim_pid = start_cpp_simulator()
            print_debug(f"Started cpp simulator with PID: {self.sim_pid}")

            self._open_pipes()

            try:
                print_debug("Getting obs...")
                obs = read_vector_from_pipe(self.pipe_observation, OBS_SIZE)
                print_debug(f"Got obs {obs}")
                break
            except ValueError as e:
                print(f"Warning: read zero-length vector from pipe on attempt {trial+1}/{max_trials}")
                if trial == max_trials - 1:
                    raise e
                continue

        assert obs.shape == (OBS_SIZE,), f"Got {obs.shape=} but expected {OBS_SIZE=}"

        print_debug("[reset] end")
        return obs, {}

    @overrides
    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        simulator_died = False
        self.episode_step_count += 1
        self.global_step_count += 1
        print_debug(f"Episode {self.episode_count} step {self.episode_step_count} (global step {self.global_step_count})")

        print_debug(f"Writing action of shape {action.shape}...")
        write_vector_to_pipe(self.pipe_action, action)
        print_debug("Wrote action")

        try:
            print_debug("Getting obs...")
            obs = read_vector_from_pipe(self.pipe_observation, OBS_SIZE)
            print_debug(f"Got obs {obs}")

            print_debug("Getting reward...")
            reward = read_vector_from_pipe(self.pipe_reward, 1)
            print(f"Got reward {reward}")

            print_debug("Getting terminated...")
            terminated = bool(read_vector_from_pipe(self.pipe_terminated, 1))
            print_debug(f"Got terminated {terminated}")

            print_debug("Getting truncated...")
            truncated = bool(read_vector_from_pipe(self.pipe_truncated, 1))
            print_debug(f"Got truncated {truncated}")
        except ValueError as e:
            print(f"Failed to read from pipe: {e}")
            simulator_died = True
            truncated = True
            terminated = False
            # TODO: how does this affect training?
            obs = np.zeros(OBS_SIZE)
            reward = 0.0

        if terminated or truncated:
            print_debug(f"Ending episode after {self.episode_step_count} steps")
            # now that python has read all of the data from this episode,
            # we can tell the cpp process to exit by sending a message in the
            # exit pipe. (I tried to just os.kill the SimRobot's process ID but
            # couldn't get that to work.)
            if not simulator_died:
                print_debug("Writing exit message...")
                write_vector_to_pipe(self.pipe_exit, np.array([1], dtype=np.float32))
                print_debug("Wrote exit message")
            print_debug("Closing pipes...")
            self._close_pipes()
            print_debug("Closed pipes")

        info = {}

        print_debug(f"[step] end")

        return obs, reward, terminated, truncated, info
