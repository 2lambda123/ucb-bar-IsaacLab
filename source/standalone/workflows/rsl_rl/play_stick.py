# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse

from omni.isaac.lab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--cpu", action="store_true", default=False, help="Use CPU pipeline.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch

from rsl_rl.runners import OnPolicyRunner

import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils import get_checkpoint_path, parse_env_cfg
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
)


import pynput
import threading
import xbox360controller

class KeyboardController:
    def __init__(self):
        self._listener = pynput.keyboard.Listener(on_press=self.on_key_pressed, on_release=self.on_key_released)
        self._is_stopped = threading.Event()
        self._is_stopped.clear()
        self.key_states = {}

    def get(self, key):
        return self.key_states.get(key.upper())

    def get_kc(self, key):
        if not key:
            return None
        
        if type(key) == pynput.keyboard.KeyCode:
            kc = key.char
            if kc:
                kc = kc.upper()
            return kc
        elif key == pynput.keyboard.Key.esc:
            return "ESC"
        elif key == pynput.keyboard.Key.left:
            return "LEFT"
        elif key == pynput.keyboard.Key.right:
            return "RIGHT"
        elif key == pynput.keyboard.Key.up:
            return "UP"
        elif key == pynput.keyboard.Key.down:
            return "DOWN"
        
        return None

    def on_key_pressed(self, key):
        kc = self.get_kc(key)
        if kc:
            self.key_states[kc] = True
    
    def on_key_released(self, key):
        kc = self.get_kc(key)
        if kc:
            self.key_states[kc] = False


stick = xbox360controller.Xbox360Controller(0)

def override_command(obs):

    velocity_commands_idx = 3 + 3 + 3

    obs[:, velocity_commands_idx]   = 0      # forward/backward
    obs[:, velocity_commands_idx+1] = 0      # left/right
    obs[:, velocity_commands_idx+2] = 0      # turn

    # if self.keycontroller.key_states.get("W"):
    #     self.obs_buf["policy"][:, velocity_commands_idx] = 1.2
    # if self.keycontroller.key_states.get("S"):
    #     self.obs_buf["policy"][:, velocity_commands_idx] = -0.5
    # if self.keycontroller.key_states.get("A"):
    #     self.obs_buf["policy"][:, velocity_commands_idx+1] = 0.5
    # if self.keycontroller.key_states.get("D"):
    #     self.obs_buf["policy"][:, velocity_commands_idx+1] = -0.5
    # if self.keycontroller.key_states.get("Q"):
    #     self.obs_buf["policy"][:, velocity_commands_idx+2] = 1.5
    # if self.keycontroller.key_states.get("E"):
        # self.obs_buf["policy"][:, velocity_commands_idx+2] = -1.5

    obs[:, velocity_commands_idx]   = -stick.axis_l.y
    obs[:, velocity_commands_idx+1] = -stick.axis_r.x
    obs[:, velocity_commands_idx+2] = -(stick.trigger_r.value - stick.trigger_l.value)

    print("command:", obs[0, velocity_commands_idx:velocity_commands_idx+3])

    return obs




def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, use_gpu=not args_cli.cpu, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # export policy to onnx
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(
        ppo_runner.alg.actor_critic, ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.pt"
    )
    export_policy_as_onnx(ppo_runner.alg.actor_critic, path=export_model_dir, filename="policy.onnx")

    # reset environment
    obs, _ = env.get_observations()

    obs = override_command(obs)

    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, _, _ = env.step(actions)

            obs = override_command(obs)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
