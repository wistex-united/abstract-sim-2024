import os
import time
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
import rich
import numpy as np


import supersuit as ss
import argparse
import importlib

from wandb.integration.sb3 import WandbCallback
import wandb

import torch

"""
Important options for training
- batch size
- num policies
- opponent deterministic
- change of random opponent
"""


def parse_args():
    parser = argparse.ArgumentParser()

    # Running options
    parser.add_argument("--train", action="store_true", default=False)
    parser.add_argument("--render", action="store_true", default=False)

    parser.add_argument("--train_steps", type=int, default=3000000)
    parser.add_argument("--train_number", type=str, default="")

    # Env options
    parser.add_argument("--env", type=str, default=None)

    # Training options
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--vec_envs", type=int, default=16)
    parser.add_argument("--num_cpus", type=int, default=8)

    parser.add_argument("--curriculum_method", type=str, default="close")
    parser.add_argument("--policy_path", type=str, default=None)

    # Wandb options
    parser.add_argument("--wandb", action="store_true", default=False)

    # Render options
    parser.add_argument("--agents", type=str, default=None)

    if not parser.parse_args().env:
        raise ValueError("Must specify env")

    return parser.parse_args()


class LargeMlpPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super(LargeMlpPolicy, self).__init__(
            *args, **kwargs, net_arch=dict(pi=[1024, 1024, 1024], vf=[1024, 1024, 1024])
        )


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.multiprocessing.set_start_method('spawn')

    env = importlib.import_module("envs." + args.env).Env()
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(
        env,
        num_vec_envs=args.vec_envs,
        num_cpus=args.num_cpus,
        base_class="stable_baselines3",
    )

    env = VecMonitor(env)
    env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_obs=10)

    run_name = args.env + "_" + time.strftime("%d_%H_%M_%S")
    if args.wandb:
        run = wandb.init(
            project=args.env,
            config={
                "env": args.env,
            },
            sync_tensorboard=True,
            name=run_name,
        )

    if not args.policy_path:
        save_path = "policies/" + args.env + "_policy" + args.train_number
    else:
        save_path = "policies/" + args.env + "_curriculum_" + args.curriculum_method

    # Clear old policies
    if not args.policy_path:
        os.system("rm -rf policies/" + args.env + "_policy" + args.train_number + "_checkpoints")
    else:
        os.system("rm -rf policies/" + args.env + "_curriculum_" + args.curriculum_method + "_checkpoints")

    checkpoint_callback = CheckpointCallback(
        save_freq = max(1000000 // args.vec_envs, 1),
        save_path=save_path + "_checkpoints",
        name_prefix=args.env + "_policy",
        verbose=2,
        )
    
    if args.wandb:
        callback = CallbackList([WandbCallback(verbose=1), checkpoint_callback])
    else:
        callback = CallbackList([checkpoint_callback])

    PPO.policy_aliases["LargeMlp"] = LargeMlpPolicy

    if not args.policy_path:
        model = PPO(
            LargeMlpPolicy,
            env,
            batch_size=args.batch_size,
            ent_coef=0.01,
            verbose=1,
            tensorboard_log=f"./runs/{run_name}",
            device=device,
        )
    else:
        model = PPO.load(args.policy_path, env=env, device=device)

    model.learn(total_timesteps=args.train_steps, callback=callback)

    if not args.policy_path:
        model.save(save_path)
        print("Finished running")
        print("Saved policy to policies/" + args.env + "_policy" + args.train_number)
    else:
        model.save(save_path)
        print("Finished running")
        print("Saved policy to policies/" + args.env + "_curriculum_" + args.curriculum_method)

    env.close()    
    exit()

def render(args):

    if args.policy_path is None:
        rich.print("No policy specified; sampling random actions instead.")
        env = importlib.import_module("envs." + args.env).Env()
        _, _ = env.reset()
        while True:
            action = {agent: action_space.sample() for agent, action_space in env.action_spaces.items()}
            # action = {0: [0, 0, 1, -1]}
            obs, rew, term, _, _ = env.step(action)
            env.render()

            if term[0]: env.reset()

    else:
        if not args.policy_path:
            policy_path = "policies/" + args.env + "_policy.zip"
        else:
            policy_path = args.policy_path

        env = importlib.import_module("envs." + args.env).Env()
        env = ss.pettingzoo_env_to_vec_env_v1(env)

        PPO.policy_aliases["LargeMlp"] = LargeMlpPolicy
        model = PPO.load(policy_path)

        # Run model
        obs, _ = env.reset()
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, _, _, _ = env.step(action)
            env.render()

if __name__ == "__main__":
    args = parse_args()

    if args.train:
        train(args)
    elif args.render:
        render(args)