import copy
import os
import sys
import time
import numpy as np
import ruamel.yaml as yaml
import torch
from stable_baselines3.common.save_util import load_from_zip_file

from scripts.arguments import parse_params, handle_policy_args
from envs.peg_insertion import ContinuousInsertionSimGymRandomizedPointFLowEnv
from path import Path
from stable_baselines3.common.utils import set_random_seed
from utils.common import get_time, get_average_params
from loguru import logger

from scripts.arguments import parse_params
from solutions.policies import TD3PolicyForPointFlowEnv

script_path = os.path.dirname(os.path.realpath(__file__))
repo_path = os.path.join(script_path, "..")
sys.path.append(script_path)
sys.path.insert(0, repo_path)

EVAL_CFG_FILE = os.path.join(repo_path, "pretrain_weight/pretrain_peg_insertion/cfg.yaml")
PEG_NUM = 3
REPEAT_NUM = 2


def evaluate_policy(model, key, render_rgb):
    exp_start_time = get_time()
    exp_name = f"peg_insertion_{exp_start_time}"
    log_dir = Path(os.path.join(repo_path, f"eval_log/{exp_name}"))
    log_dir.makedirs_p()

    logger.remove()
    logger.add(log_dir / f"{exp_name}.log")
    logger.add(sys.stderr, format="{time:YYYY-MM-DD HH:mm:ss} {level} {message}", level="INFO")

    logger.info(f"#KEY: {key}")

    with open(EVAL_CFG_FILE, "r") as f:
        cfg = yaml.YAML(typ='safe', pure=True).load(f)

    # get simulation and environment parameters
    sim_params = cfg["env"].pop("params")
    env_name = cfg["env"].pop("env_name")

    params_lb, params_ub = parse_params(env_name, sim_params)
    average_params = get_average_params(params_lb, params_ub)
    logger.info(f"\n{average_params}")
    logger.info(cfg["env"])

    if "max_action" in cfg["env"].keys():
        cfg["env"]["max_action"] = np.array(cfg["env"]["max_action"])

    specified_env_args = copy.deepcopy(cfg["env"])

    specified_env_args.update(
        {
            "params": average_params,
            "params_upper_bound": average_params,
        }
    )
    specified_env_args["render_rgb"] = render_rgb

    # create evaluation environment
    env = ContinuousInsertionSimGymRandomizedPointFLowEnv(**specified_env_args)
    set_random_seed(0)

    offset_list = [[-4.0, -4.0, -8.0], [-4.0, -2.0, 2.0], [-4.0, 1.0, -6.0], [-4.0, 3.0, 6.0], [-3.0, -3.0, -2.0],
                   [-3.0, -1.0, 8.0], [-3.0, 2.0, 2.0], [-2.0, -4.0, -6.0], [-2.0, -2.0, 4.0], [-2.0, 1.0, -2.0],
                   [-2.0, 3.0, 8.0], [-1.0, -3.0, 0.0], [-1.0, 0.0, 6.0], [-1.0, 3.0, 4.0], [0.0, -3.0, -4.0],
                   [0.0, 0.0, 6.0], [0.0, 3.0, 4.0], [1.0, -3.0, -4.0], [1.0, 0.0, -4.0], [1.0, 3.0, 0.0],
                   [2.0, -3.0, -8.0], [2.0, -1.0, 4.0], [2.0, 2.0, -4.0], [2.0, 4.0, 6.0], [3.0, -2.0, 0.0],
                   [3.0, 1.0, -8.0], [3.0, 3.0, 2.0], [4.0, -3.0, -4.0], [4.0, -1.0, 6.0], [4.0, 2.0, -2.0]]
    test_num = len(offset_list)
    test_result = []

    for i in range(PEG_NUM):
        for r in range(REPEAT_NUM):
            for k in range(test_num):
                logger.opt(colors=True).info(f"<blue>#### Test No. {len(test_result) + 1} ####</blue>")
                o, _ = env.reset(offset_list[k], peg_idx=i)
                initial_offset_of_current_episode = o["gt_offset"]
                logger.info(f"Initial offset: {initial_offset_of_current_episode}")
                d, ep_ret, ep_len = False, 0, 0
                while not d:
                    # Take deterministic actions at test time (noise_scale=0)
                    ep_len += 1
                    for obs_k, obs_v in o.items():
                        o[obs_k] = torch.from_numpy(obs_v)
                    action = model(o)
                    action = action.cpu().detach().numpy().flatten()
                    logger.info(f"Step {ep_len} Action: {action}")
                    o, r, terminated, truncated, info = env.step(action)
                    d = terminated or truncated
                    if 'gt_offset' in o.keys():
                        logger.info(f"Offset: {o['gt_offset']}")
                    ep_ret += r
                if info["is_success"]:
                    test_result.append([True, ep_len])
                    logger.opt(colors=True).info(f"<green>RESULT: SUCCESS</green>")
                else:
                    test_result.append([False, ep_len])
                    logger.opt(colors=True).info(f"<d>RESULT: FAIL</d>")

    env.close()
    success_rate = np.sum(np.array([int(v[0]) for v in test_result])) / (test_num * PEG_NUM * REPEAT_NUM)
    if success_rate > 0:
        avg_steps = np.mean(np.array([int(v[1]) if v[0] else 0 for v in test_result])) / success_rate
        logger.info(f"#SUCCESS_RATE: {success_rate*100.0:.2f}%")
        logger.info(f"#AVG_STEP: {avg_steps:.2f}")
    else:
        logger.info(f"#SUCCESS_RATE: 0")
        logger.info(f"#AVG_STEP: NA")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--key", type=str, required=True, help="use the key sent to you")
    parser.add_argument("--render_rgb",action="store_true")
    args = parser.parse_args()
    key = args.key
    # replace the model with your own policy
    import torch.nn as nn

    class ZeroAction(nn.Module):
        def forward(self, obs):
            return torch.zeros(3, dtype=torch.float32)

    # model = ZeroAction()

    policy_file = "../pretrain_weight/pretrain_peg_insertion/best_model.zip"
    data, params, _ = load_from_zip_file(policy_file)
    model = TD3PolicyForPointFlowEnv(observation_space=data["observation_space"],
                                    action_space=data["action_space"],
                                    lr_schedule=data["lr_schedule"],
                                    **data["policy_kwargs"],)
    model.load_state_dict(params["policy"])

    evaluate_policy(model, key, args.render_rgb)
