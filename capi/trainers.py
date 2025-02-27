"""Trainer for Trade Comm and coordinator"""

from collections import defaultdict, deque
from pathlib import Path
from typing import List, Tuple
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from .agents import Agent
from .games import Game


class Trainer:
    def __init__(
        self, game: Game, agent: Agent, agent_type: str, directory: str = "results", jobnum: int = 0
    ):
        """Trainer for Trade Comm and coordinator

        Args:
            game: Trade Comm PuB-MDP
            agent: PuB-MDP coordinator
            directory: Directory to which to write date
            jobnum: Job identifier

        Attributes:
            See args
        """
        self.game = game
        self.agent = agent
        self.agent_type = agent_type
        self.directory = directory
        self.jobnum = jobnum
        Path(directory).mkdir(exist_ok=True)

    def play_episode(self, train: bool) -> float:
        """Play an episode

        Args:
            train: Whether the agent is training or being evaluated

        Returns:
            Expected return over public tree
        """
        decision_points = [(self.game.init_state(), 1)]
        er = torch.tensor(0.0)
        while len(decision_points) > 0:
            s, prod = decision_points.pop(0)
            prescription, action_dynamics, val, done = self.agent.act(s, train)
            if done:
                er += prod * val
            else:
                for a, p in enumerate(action_dynamics):
                    if p > 0:
                        s_ = s.clone()
                        s_.apply_action(prescription, a)
                        decision_points.append((s_, prod * p))
        return er.item()

    def run(self, num_episodes: int, write_every: int, seed: int, dim: int, epsilon: float = None) -> None:
        """Run the trainer

        Args:
            num_episodes: Number of episodes for which to train
            write_every: The period at which to save data
        """
        vals = []
        for t in tqdm(range(1, num_episodes+1)):
            self.play_episode(train=True)
            self.agent.train()
            if t % write_every == 0:
                vals.append((t, self.play_episode(train=False)))
                self.write(vals, seed, num_episodes, dim, epsilon)
        

    def write(self, vals: List[Tuple[int, float]], seed: int, num_episodes: int, dim: int, epsilon: float) -> None:
        """Write data

        Args:
            vals: list of (episode_num, expected_return) tuples
        """
        data = {}
        episode_nums, expected_returns = list(zip(*vals))
        data["episode"] = episode_nums
        data["expected_return"] = expected_returns
        data["jobnum"] = tuple(len(vals) * [self.jobnum])
        df = pd.DataFrame(data)
        # df.to_pickle(f"{self.directory}/job{self.jobnum}.pkl")
        
        sns.lineplot(data=df, x="episode", y="expected_return")
        plt.axhline(y=1.0, color="gray", linestyle="-")
        if epsilon is None:
            df.to_csv(f"{self.directory}/job{self.jobnum}_{self.agent_type}_{num_episodes}_dim_{dim}_{seed}.csv")
            plt.savefig(f"{self.directory}/job{self.jobnum}_{self.agent_type}_{num_episodes}_dim_{dim}_{seed}.png")
        else:
            df.to_csv(f"{self.directory}/job{self.jobnum}_{self.agent_type}_{num_episodes}_dim_{dim}_epsilon_{epsilon}_{seed}.csv")
            plt.savefig(f"{self.directory}/job{self.jobnum}_{self.agent_type}_{num_episodes}_dim_{dim}_epsilon_{epsilon}_{seed}.png")
        
        plt.close()
        
