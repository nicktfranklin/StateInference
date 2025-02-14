import argparse
import logging
import os
import pickle
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml
from stable_baselines3.common.monitor import Monitor

from src.model.agents.lookahead_value_iteration import (
    LookaheadViAgent as ValueIterationAgent,
)
from src.model.training.callbacks import ThreadTheNeedleCallback
from src.model.training.rollout_data import RolloutBuffer as Buffer
from src.task.gridworld import CnnWrapper, GridWorldEnv
from src.task.gridworld import ThreadTheNeedleEnv as Environment
from src.utils.config_utils import parse_configs
from src.utils.pytorch_utils import DEVICE

logging.info(f"python {sys.version}")
logging.info(f"torch {torch.__version__}")
logging.info(f"device = {DEVICE}")


BASE_FILE_NAME = "thread_the_needle_cnn_vae_DEBUGGER"


project_root = Path(__file__).resolve().parents[1]
print(project_root)


### Configuration files
parser = argparse.ArgumentParser()


parser.add_argument("--vae_config", default=f"{project_root}/configs/vae_config.yml")
parser.add_argument("--task_config", default=f"{project_root}/configs/env_config.yml")
parser.add_argument(
    "--agent_config", default=f"{project_root}/configs/agent_config.yml"
)
parser.add_argument("--task_name", default="thread_the_needle")
parser.add_argument("--model_name", default="cnn_vae")
parser.add_argument("--results_dir", default=f"simulations/")
parser.add_argument("--log_dir", default=f"logs/{BASE_FILE_NAME}_{date.today()}/")
parser.add_argument("--n_training_samples", default=50000)  # 50000
parser.add_argument("--n_rollout_samples", default=10000)  # 50000
parser.add_argument("--n_batch", default=24)  # 24
parser.add_argument("--capacity", default=16384)


@dataclass
class Config:
    log_dir: str
    results_dir: str
    env_kwargs: Dict[str, Any]
    agent_config: Dict[str, Any]
    vae_config: Dict[str, Any]

    n_training_samples: int
    n_rollout_samples: int

    n_batch: int
    epsilon: float = 0.02

    capacity: Optional[int] = None

    @classmethod
    def construct(cls, args: argparse.Namespace):
        configs = parse_configs(args)
        return cls(
            log_dir=args.log_dir,
            env_kwargs=configs["env_kwargs"],
            vae_config=configs["vae_config"],
            agent_config=configs["agent_config"],
            n_training_samples=args.n_training_samples,
            n_rollout_samples=args.n_rollout_samples,
            results_dir=args.results_dir,
            n_batch=args.n_batch,
            capacity=args.capacity,
        )


def make_env(configs: Config) -> GridWorldEnv:
    # create the task
    task = CnnWrapper(Environment.create_env(**configs.env_kwargs))

    # create the monitor
    task = Monitor(task, configs.log_dir)

    return task


def train_agent(configs: Config):

    # create task
    task = make_env(configs)
    task = Monitor(task, configs.log_dir)  # not sure I use this

    callback = ThreadTheNeedleCallback()

    agent = ValueIterationAgent.make_from_configs(
        task, configs.agent_config, configs.vae_config, configs.env_kwargs
    )

    agent.learn(
        total_timesteps=configs.n_training_samples,
        progress_bar=True,
        callback=callback,
        capacity=configs.capacity,
        buffer_class="priority",
        buffer_kwargs=dict(aggregation=np.max),
    )

    data = {"rewards": callback.rewards, "evaluations": callback.evaluations}

    return agent, data


def main():
    config = Config.construct(parser.parse_args())

    config_record = f"logs/{BASE_FILE_NAME}_config_{date.today()}.yaml"
    with open(config_record, "w") as f:
        yaml.dump(config.__dict__, f)

    # Create log dir
    os.makedirs(config.log_dir, exist_ok=True)

    # train ppo
    batched_data = []
    for ii in range(config.n_batch):
        logging.info(f"running batch {ii}")
        agent, data = train_agent(config)
        data["batch"] = ii
        batched_data.append(data)

        with open(
            f"{config.results_dir}lookahead_priority_batched_data_{date.today()}.pkl",
            "wb",
        ) as f:
            pickle.dump(batched_data, f)

    rollout_buffer = Buffer()
    rollout_buffer = agent.collect_buffer(
        agent.env, rollout_buffer, n=1000, epsilon=config.epsilon
    )

    with open(
        f"{config.results_dir}lookahead_priority_rollouts_{date.today()}.pkl", "wb"
    ) as f:
        pickle.dump(rollout_buffer, f)


if __name__ == "__main__":
    main()
