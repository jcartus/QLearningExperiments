"""This script will run the agent in a 1 D environment and plot its learning 
progress

Author: Johannes Cartus, 22.12.2021
"""

import numpy as np
import matplotlib.pyplot as plt

from environment import DiscreteEnvironment1D
from agent import AgentBase
from analysis import AnalyzerEpisode

def generate_course():
    return np.array([0, 0, 1, 0, 0, 6, 0, -10, 0])

def main():

    game_map = generate_course()

    environment = DiscreteEnvironment1D(game_map=game_map)

    agent = AgentBase(
        environment=environment,
        discount_factor=1,
        learning_rate=0.3,
        epsilon_greedyness=0.4
    )

    agent.run_episode()

    analyzer = AnalyzerEpisode(episode_statistics=agent._episode_statistics)
    analyzer.plot_reward()

if __name__ == '__main__':
    main()