import os
import torch
from typing import Tuple
import stable_baselines3 as sb3
from torch.autograd import Variable
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.policies import BasePolicy

import sys

# Export code from: https://stable-baselines3.readthedocs.io/en/master/guide/export.html
class OnnxableSB3Policy(torch.nn.Module):
    def __init__(self, policy: BasePolicy):
        super().__init__()
        self.policy = policy

    def forward(self, observation: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # NOTE: Preprocessing is included, but postprocessing
        # (clipping/inscaling actions) is not,
        # If needed, you also need to transpose the images so that they are channel first
        # use deterministic=False if you want to export the stochastic policy
        # policy() returns `actions, values, log_prob` for PPO
        return self.policy(observation, deterministic=True)

if __name__ == "__main__":
    if not len(sys.argv) == 2 and not len(sys.argv) == 3:
        print("usage: ./export.py <model path>")
        exit(1)

    model_path = sys.argv[1]
    
    model = PPO.load(model_path, device="cpu")

    onnx_policy = OnnxableSB3Policy(model.policy)

    observation_size = model.observation_space.shape
    dummy_input = torch.randn(1, *observation_size)
    os.makedirs("exported", exist_ok=True)
    torch.onnx.export(
        onnx_policy,
        dummy_input,
        "exported/policy.onnx",
        opset_version=15,
        input_names=["input"],
        output_names=["output"],
    )